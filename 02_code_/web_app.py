"""Web interface for the DEG-DR (disease-entity graph + BGE-M3) medical QA pipeline.

Serves a single-page medical Q&A site (02_code/web/) and a small JSON API
backed by the same cached index used by run_experiment.py / chat.py. The
index is loaded in a background thread at startup so the page can show a
loading state instead of the server blocking for ~20-60s.

Usage:
    python 02_code/web_app.py                 # http://127.0.0.1:5000
    python 02_code/web_app.py --port 8080 --host 0.0.0.0
    python 02_code/web_app.py --max-docs 5000  # match a different cached build

Endpoints:
    GET  /             the web page
    GET  /api/status   index loading state + corpus/graph statistics
    POST /api/ask      {"question", "top_k", "use_graph", "use_llm"} -> answer + evidence
"""
import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, send_from_directory

import config
from qa_pipeline import generate_answer

WEB_DIR = config.BASE_DIR / "web"
MAX_QUESTION_CHARS = 1000

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/static")

_state = {
    "ready": False,
    "error": None,
    "started_at": time.time(),
    "load_time_s": None,
    "stats": {},
}
_retriever = None
# All model work (index loading and every query) runs on ONE worker thread.
# Flask serves each request on a new thread; running inference there would
# re-create per-thread CUDA/cuBLAS state on every request and would ignore
# faiss.omp_set_num_threads(), which only applies to the thread that set it.
# Together these roughly doubled per-query latency. A single worker also
# serializes access to the shared model.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deg-dr")


def _load_index(max_docs: int):
    global _retriever
    try:
        from retriever import build_or_load_index, graph_stats

        t0 = time.perf_counter()
        retriever, corpus, graph, _ = build_or_load_index(force_rebuild=False, max_docs=max_docs)
        stats = graph_stats(graph)
        _state["stats"] = {
            "corpus_size": int(len(corpus)),
            "num_entities": int(stats["num_entities"]),
            "num_edges": int(stats["num_edges"]),
            "num_communities": int(stats["num_communities"]),
            "num_sources": int(corpus["source"].nunique()),
            "device": getattr(retriever.embedder, "device", "cpu"),
        }
        _state["load_time_s"] = round(time.perf_counter() - t0, 1)
        _retriever = retriever
        _state["ready"] = True
        print(f"[web] index ready in {_state['load_time_s']}s -> open http://{app.config['HOST']}:{app.config['PORT']}")
    except Exception as exc:  # surface load failures in the UI instead of a dead server
        traceback.print_exc()
        _state["error"] = f"{type(exc).__name__}: {exc}"


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "ready": _state["ready"],
        "error": _state["error"],
        "elapsed_s": round(time.time() - _state["started_at"], 1),
        "load_time_s": _state["load_time_s"],
        "stats": _state["stats"],
        "llm_available": bool(os.environ.get("OPENAI_API_KEY")),
        "defaults": {"top_k": config.FINAL_TOP_K, "dense_top_k": config.DENSE_TOP_K,
                     "linked_entities": config.GRAPH_HOP_ENTITIES, "rrf_k": config.RRF_K},
    })


@app.post("/api/ask")
def ask():
    if not _state["ready"]:
        return jsonify({"error": "The index is still loading. Please wait a moment."}), 503

    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()[:MAX_QUESTION_CHARS]
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        top_k = max(1, min(int(payload.get("top_k", config.FINAL_TOP_K)), config.DENSE_TOP_K))
    except (TypeError, ValueError):
        top_k = config.FINAL_TOP_K
    use_graph = bool(payload.get("use_graph", True))
    use_llm = bool(payload.get("use_llm", False))

    def _run():
        retrieval = _retriever.retrieve(question, final_top_k=top_k, use_graph=use_graph)
        context_docs = _retriever.get_docs(retrieval.doc_ids)
        return retrieval, context_docs, generate_answer(question, context_docs, use_llm=use_llm)

    retrieval, context_docs, answer = _worker.submit(_run).result()

    contexts = []
    for rank, (row, score) in enumerate(zip(context_docs.itertuples(), retrieval.scores), start=1):
        contexts.append({
            "rank": rank,
            "doc_id": int(row.doc_id),
            "question": row.question,
            "answer": row.answer,
            "focus_area": row.focus_area,
            "source": row.source,
            "score": float(score),
        })

    return jsonify({
        "question": question,
        "answer": answer.text,
        "mode": answer.mode,
        "method": "graphrag_rrf" if use_graph else "dense_only",
        "linked_entities": retrieval.linked_entities,
        "retrieval_ms": round(retrieval.latency_ms, 2),
        "generation_ms": round(answer.latency_ms, 2),
        "contexts": contexts,
    })


def main():
    p = argparse.ArgumentParser(description="Web UI for the DEG-DR medical QA pipeline")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--max-docs", type=int, default=config.MAX_DOCS,
                   help="corpus size to load from cache (default: full dataset)")
    args = p.parse_args()

    app.config["HOST"], app.config["PORT"] = args.host, args.port
    _worker.submit(_load_index, args.max_docs)
    print(f"[web] server starting on http://{args.host}:{args.port} (index loads in background)")
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
