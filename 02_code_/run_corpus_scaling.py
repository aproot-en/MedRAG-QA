"""Corpus-size scaling study for the GraphRAG + BGE-M3 pipeline.

Builds independent indices (embeddings + entity graph) at several corpus
sizes and evaluates each with graph fusion on (proposed) and off (dense-only
baseline) on that corpus's own 100-question sample. Smaller-corpus caches
are written to a scratch directory (not 02_code/artifacts) so the main
full-corpus cache used by run_experiment.py / run_ablation.py is never
touched or rebuilt. The full corpus size (-1) reuses the main cache's
already-computed summary/comparison CSVs directly instead of rebuilding.
Writes 03_result/result-corpus-scaling.csv.
"""
import json
import time

import config
import pandas as pd

import metrics
from qa_pipeline import generate_answer
from retriever import build_or_load_index, graph_stats

SCRATCH_ARTIFACT_ROOT = config.BASE_DIR / "artifacts_scale"

SIZES = [1000, 2000, 5000, 8000]  # full corpus (16345) reuses the main cache's existing results


def evaluate_method(retriever, embedder, eval_sample, use_graph: bool) -> dict:
    rows = []
    for _, row in eval_sample.iterrows():
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
        "hit_rate_at_1_pct": round(df["hit_at_1"].mean() * 100, 2),
        "hit_rate_at_5_pct": round(df["hit_at_5"].mean() * 100, 2),
        "mrr_pct": round(df["reciprocal_rank"].mean() * 100, 2),
        "avg_rouge_l_f1_pct": round(df["rouge_l_f1"].mean() * 100, 2),
        "avg_semantic_cosine_sim_pct": round(df["semantic_cosine_sim"].mean() * 100, 2),
        "avg_retrieval_latency_ms": round(df["retrieval_latency_ms"].mean(), 3),
    }


def run_one_size(max_docs: int) -> dict:
    config.ARTIFACT_DIR = SCRATCH_ARTIFACT_ROOT / str(max_docs)
    t0 = time.perf_counter()
    retriever, corpus, graph, embedder = build_or_load_index(force_rebuild=False, max_docs=max_docs)
    build_time_s = time.perf_counter() - t0
    stats = graph_stats(graph)

    eval_sample = corpus.sample(n=min(config.NUM_EVAL_QUERIES, len(corpus)),
                                 random_state=config.RANDOM_SEED).reset_index(drop=True)

    dense_only = evaluate_method(retriever, embedder, eval_sample, use_graph=False)
    graph_rrf = evaluate_method(retriever, embedder, eval_sample, use_graph=True)

    return {
        "corpus_size": len(corpus),
        "num_entities": stats["num_entities"],
        "num_edges": stats["num_edges"],
        "num_communities": stats["num_communities"],
        "build_time_s": round(build_time_s, 1),
        "dense_hit_at_1_pct": dense_only["hit_rate_at_1_pct"],
        "dense_hit_at_5_pct": dense_only["hit_rate_at_5_pct"],
        "dense_mrr_pct": dense_only["mrr_pct"],
        "dense_rouge_l_f1_pct": dense_only["avg_rouge_l_f1_pct"],
        "dense_semsim_pct": dense_only["avg_semantic_cosine_sim_pct"],
        "dense_latency_ms": dense_only["avg_retrieval_latency_ms"],
        "graph_hit_at_1_pct": graph_rrf["hit_rate_at_1_pct"],
        "graph_hit_at_5_pct": graph_rrf["hit_rate_at_5_pct"],
        "graph_mrr_pct": graph_rrf["mrr_pct"],
        "graph_rouge_l_f1_pct": graph_rrf["avg_rouge_l_f1_pct"],
        "graph_semsim_pct": graph_rrf["avg_semantic_cosine_sim_pct"],
        "graph_latency_ms": graph_rrf["avg_retrieval_latency_ms"],
    }


def full_corpus_row_from_main_cache() -> dict:
    """Reuse the already-computed full-corpus results (result-rag-summary.csv /
    result-rag-comparison.csv) instead of re-embedding the full ~16.3k corpus."""
    summary = pd.read_csv(config.SUMMARY_CSV).iloc[0]
    comparison = pd.read_csv(config.RESULT_DIR / "result-rag-comparison.csv")
    dense_row = comparison[comparison["method"].str.contains("dense-only")].iloc[0]
    meta = json.loads((config.ARTIFACT_DIR / "meta.json").read_text())
    assert meta["max_docs"] == -1, "main cache is not the full-corpus build"

    return {
        "corpus_size": int(summary["corpus_size"]),
        "num_entities": int(summary["num_entities"]),
        "num_edges": int(summary["num_graph_edges"]),
        "num_communities": int(summary["num_communities"]),
        "build_time_s": round(float(summary["index_build_time_s"]), 1),
        "dense_hit_at_1_pct": dense_row["hit_rate_at_1_pct"],
        "dense_hit_at_5_pct": dense_row["hit_rate_at_5_pct"],
        "dense_mrr_pct": dense_row["mrr_pct"],
        "dense_rouge_l_f1_pct": dense_row["avg_rouge_l_f1_pct"],
        "dense_semsim_pct": dense_row["avg_semantic_cosine_sim_pct"],
        "dense_latency_ms": dense_row["avg_retrieval_latency_ms"],
        "graph_hit_at_1_pct": summary["hit_rate_at_1_pct"],
        "graph_hit_at_5_pct": summary["hit_rate_at_5_pct"],
        "graph_mrr_pct": summary["mrr_pct"],
        "graph_rouge_l_f1_pct": summary["avg_rouge_l_f1_pct"],
        "graph_semsim_pct": summary["avg_semantic_cosine_sim_pct"],
        "graph_latency_ms": summary["avg_retrieval_latency_ms"],
    }


def main():
    original_artifact_dir = config.ARTIFACT_DIR
    rows = []
    try:
        for size in SIZES:
            print(f"\n=== corpus scaling: max_docs={size} ===")
            rows.append(run_one_size(size))

        print("\n=== corpus scaling: max_docs=-1 (full cleaned corpus, reusing main cache) ===")
        config.ARTIFACT_DIR = original_artifact_dir
        rows.append(full_corpus_row_from_main_cache())
    finally:
        config.ARTIFACT_DIR = original_artifact_dir

    out_df = pd.DataFrame(rows)
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULT_DIR / "result-corpus-scaling.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n" + out_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
