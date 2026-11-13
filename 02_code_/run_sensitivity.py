"""Hyperparameter sensitivity sweep for the GraphRAG + BGE-M3 retriever.

Reuses the cached full-corpus artifacts built by run_experiment.py (no
re-embedding) and re-evaluates the same 100-question sample while varying
one retrieval hyperparameter at a time, holding the others at their
config.py defaults (K_d=20, H=2, K_f=5, RRF k=60). Writes
03_result/result-sensitivity.csv, one row per (sweep, value) pair, for the
paper's hyperparameter-sensitivity analysis.
"""
import config
import pandas as pd
from tqdm import tqdm

import metrics
from qa_pipeline import generate_answer
from retriever import build_or_load_index


def evaluate(retriever, embedder, eval_sample, use_graph=True, final_top_k=config.FINAL_TOP_K):
    rows = []
    for _, row in eval_sample.iterrows():
        question = row["question"]
        gold_answer = row["answer"]
        gold_doc_id = int(row["doc_id"])

        retrieval = retriever.retrieve(question, final_top_k=final_top_k, use_graph=use_graph)
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


def main():
    retriever, corpus, graph, embedder = build_or_load_index(force_rebuild=False, max_docs=config.MAX_DOCS)
    eval_sample = corpus.sample(n=min(config.NUM_EVAL_QUERIES, len(corpus)),
                                 random_state=config.RANDOM_SEED).reset_index(drop=True)

    sweeps = {
        "final_top_k (K_f)": [1, 3, 5, 10, 15, 20],
        "dense_top_k (K_d)": [10, 15, 20, 30, 50],
        "graph_hop_entities (H)": [1, 2, 3, 4, 5],
        "rrf_k": [10, 30, 60, 100, 200],
    }

    # Save original defaults so each sweep only varies its own parameter.
    defaults = dict(DENSE_TOP_K=config.DENSE_TOP_K, GRAPH_HOP_ENTITIES=config.GRAPH_HOP_ENTITIES,
                     RRF_K=config.RRF_K, FINAL_TOP_K=config.FINAL_TOP_K)

    rows = []
    for name, values in sweeps.items():
        for v in tqdm(values, desc=name):
            config.DENSE_TOP_K, config.GRAPH_HOP_ENTITIES, config.RRF_K = (
                defaults["DENSE_TOP_K"], defaults["GRAPH_HOP_ENTITIES"], defaults["RRF_K"])
            final_top_k = defaults["FINAL_TOP_K"]

            if name.startswith("final_top_k"):
                final_top_k = v
            elif name.startswith("dense_top_k"):
                config.DENSE_TOP_K = v
            elif name.startswith("graph_hop_entities"):
                config.GRAPH_HOP_ENTITIES = v
            elif name.startswith("rrf_k"):
                config.RRF_K = v

            metrics_row = evaluate(retriever, embedder, eval_sample, use_graph=True, final_top_k=final_top_k)
            metrics_row["sweep"] = name
            metrics_row["value"] = v
            rows.append(metrics_row)

    # restore defaults
    config.DENSE_TOP_K, config.GRAPH_HOP_ENTITIES, config.RRF_K = (
        defaults["DENSE_TOP_K"], defaults["GRAPH_HOP_ENTITIES"], defaults["RRF_K"])

    out_df = pd.DataFrame(rows)[[
        "sweep", "value", "hit_rate_at_1_pct", "hit_rate_at_5_pct", "mrr_pct",
        "avg_rouge_l_f1_pct", "avg_semantic_cosine_sim_pct", "avg_retrieval_latency_ms",
    ]]
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULT_DIR / "result-sensitivity.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
