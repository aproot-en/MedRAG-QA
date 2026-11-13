"""Ablation: GraphRAG + BGE-M3 (proposed) vs. BGE-M3 dense-only (baseline).

Reuses the cached corpus/embeddings/graph built by run_experiment.py (no
re-embedding of the corpus) and re-runs retrieval + evaluation twice on the
same sampled eval questions: once with graph expansion enabled (the
proposed method) and once with it disabled (`use_graph=False`, a pure dense
BGE-M3 retriever). Writes 03_result/result-rag-comparison.csv, one row per
method, for the paper's "Comparison" section.

Reuses whatever corpus size is currently cached in 02_code/artifacts/ (see
meta.json) so it never triggers a re-embed; pass --max-docs to match a
different cached build if needed.
"""
import argparse

import config

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

import metrics  # noqa: E402
from qa_pipeline import generate_answer  # noqa: E402
from retriever import build_or_load_index  # noqa: E402


def evaluate_method(retriever, embedder, eval_sample, use_graph: bool) -> dict:
    rows = []
    for _, row in tqdm(eval_sample.iterrows(), total=len(eval_sample),
                        desc=f"use_graph={use_graph}"):
        question = row["question"]
        gold_answer = row["answer"]
        gold_doc_id = int(row["doc_id"])

        retrieval = retriever.retrieve(question, use_graph=use_graph)
        context_docs = retriever.get_docs(retrieval.doc_ids)
        answer = generate_answer(question, context_docs, use_llm=False)

        hit_metrics = metrics.retrieval_hit_metrics(gold_doc_id, retrieval.doc_ids)
        rouge_score = metrics.rouge_l_f1(answer.text, gold_answer)
        semantic_sim = metrics.semantic_cosine_similarity(embedder, answer.text, gold_answer)

        rows.append({
            "hit_at_1": hit_metrics["hit_at_1"],
            "hit_at_5": hit_metrics["hit_at_5"],
            "reciprocal_rank": hit_metrics["reciprocal_rank"],
            "rouge_l_f1": rouge_score,
            "semantic_cosine_sim": semantic_sim,
            "retrieval_latency_ms": retrieval.latency_ms,
        })
    df = pd.DataFrame(rows)
    return {
        "method": "GraphRAG + BGE-M3 (proposed)" if use_graph else "BGE-M3 dense-only (baseline)",
        "hit_rate_at_1_pct": round(df["hit_at_1"].mean() * 100, 2),
        "hit_rate_at_5_pct": round(df["hit_at_5"].mean() * 100, 2),
        "mrr_pct": round(df["reciprocal_rank"].mean() * 100, 2),
        "avg_rouge_l_f1_pct": round(df["rouge_l_f1"].mean() * 100, 2),
        "avg_semantic_cosine_sim_pct": round(df["semantic_cosine_sim"].mean() * 100, 2),
        "avg_retrieval_latency_ms": round(df["retrieval_latency_ms"].mean(), 3),
    }


def main():
    p = argparse.ArgumentParser(description="GraphRAG+BGE-M3 vs. BGE-M3 dense-only ablation")
    p.add_argument("--max-docs", type=int, default=None,
                    help="corpus size to reuse from cache (default: whatever is currently cached)")
    args = p.parse_args()

    if args.max_docs is not None:
        max_docs = args.max_docs
    else:
        cached_meta = config.ARTIFACT_DIR / "meta.json"
        import json
        max_docs = json.loads(cached_meta.read_text())["max_docs"] if cached_meta.exists() else config.MAX_DOCS

    retriever, corpus, graph, embedder = build_or_load_index(force_rebuild=False, max_docs=max_docs)
    eval_sample = corpus.sample(n=min(config.NUM_EVAL_QUERIES, len(corpus)),
                                 random_state=config.RANDOM_SEED).reset_index(drop=True)

    baseline = evaluate_method(retriever, embedder, eval_sample, use_graph=False)
    proposed = evaluate_method(retriever, embedder, eval_sample, use_graph=True)

    comparison_df = pd.DataFrame([baseline, proposed])
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULT_DIR / "result-rag-comparison.csv"
    comparison_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== Ablation: graph expansion on vs. off ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
