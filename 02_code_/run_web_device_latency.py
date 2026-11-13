"""CPU vs. GPU latency of the web-based QA system (web_app.py).

Runs the Flask app in-process and sends the seeded 100 evaluation questions
to POST /api/ask through Flask's test client, so the timing covers the same
path a browser request takes (JSON parsing, the single model worker thread,
retrieval, extractive answer selection and JSON serialization), minus the
network socket. The BGE-M3 query encoder is placed on the requested device;
FAISS search, entity linking, graph expansion and RRF always run on the CPU.

Usage (one device per process so CUDA state cannot leak into the CPU run):
    python 02_code/run_web_device_latency.py --device cuda
    python 02_code/run_web_device_latency.py --device cpu

Appends one row per (device, method) to 03_result/result-web-device-latency.csv.
"""
import argparse
import platform
import time

import numpy as np
import pandas as pd

import config

WARMUP_QUERIES = 10
REPEATS = 3


def summarize(device: str, method: str, retrieval: np.ndarray, request: np.ndarray, load_s: float) -> dict:
    return {
        "device": device,
        "method": method,
        "n_queries": len(retrieval),
        "index_load_s": round(load_s, 1),
        "retrieval_mean_ms": round(float(retrieval.mean()), 2),
        "retrieval_median_ms": round(float(np.median(retrieval)), 2),
        "retrieval_p95_ms": round(float(np.percentile(retrieval, 95)), 2),
        "request_mean_ms": round(float(request.mean()), 2),
        "request_median_ms": round(float(np.median(request)), 2),
        "request_p95_ms": round(float(np.percentile(request, 95)), 2),
        "throughput_qps": round(1000.0 / float(request.mean()), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cuda", "cpu"], required=True)
    args = p.parse_args()

    config.EMBED_DEVICE = args.device  # read by embedder._load_model at load time
    import torch
    import web_app

    web_app.app.config["HOST"], web_app.app.config["PORT"] = "127.0.0.1", 5000  # only used in the ready log line
    t0 = time.perf_counter()
    web_app._worker.submit(web_app._load_index, config.MAX_DOCS).result()
    load_s = time.perf_counter() - t0
    if not web_app._state["ready"]:
        raise SystemExit(f"index failed to load: {web_app._state['error']}")
    device = web_app._state["stats"]["device"]
    print(f"[bench] device={device} torch_threads={torch.get_num_threads()} load={load_s:.1f}s")

    corpus = web_app._retriever.corpus.reset_index(drop=True)
    questions = corpus.sample(n=config.NUM_EVAL_QUERIES, random_state=config.RANDOM_SEED)["question"].tolist()
    warmup = corpus.sample(n=WARMUP_QUERIES, random_state=config.RANDOM_SEED + 1)["question"].tolist()

    client = web_app.app.test_client()

    def ask(question: str, use_graph: bool):
        t = time.perf_counter()
        resp = client.post("/api/ask", json={"question": question, "top_k": config.FINAL_TOP_K,
                                             "use_graph": use_graph, "use_llm": False})
        request_ms = (time.perf_counter() - t) * 1000
        body = resp.get_json()
        assert resp.status_code == 200, body
        return body["retrieval_ms"], request_ms

    for q in warmup:
        ask(q, True)
        ask(q, False)

    timings = {True: ([], []), False: ([], [])}
    for q in questions:
        runs = {True: ([], []), False: ([], [])}
        for _ in range(REPEATS):
            for use_graph in (False, True):  # interleaved so both methods see the same device state
                r_ms, q_ms = ask(q, use_graph)
                runs[use_graph][0].append(r_ms)
                runs[use_graph][1].append(q_ms)
        for use_graph in (False, True):
            timings[use_graph][0].append(np.median(runs[use_graph][0]))
            timings[use_graph][1].append(np.median(runs[use_graph][1]))

    rows = [summarize(device, "Dense-only", *map(np.asarray, timings[False]), load_s),
            summarize(device, "DEG-DR", *map(np.asarray, timings[True]), load_s)]
    hw = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
    for row in rows:
        row["hardware"] = hw
        row["torch_threads"] = torch.get_num_threads()

    out = config.RESULT_DIR / "result-web-device-latency.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df = df.drop_duplicates(subset=["device", "method"], keep="last")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
