"""Controlled latency measurement for the GraphRAG + BGE-M3 retriever.

The per-question latencies logged by run_experiment.py include a cold first
query (CUDA kernel/cache warm-up), and the other scripts each measured latency
in their own separate run, so the same configuration was reported with
slightly different numbers. This script measures both methods in one session
on the same 100 evaluation questions:

  - WARMUP_QUERIES untimed queries first, to exclude one-time warm-up cost;
  - each question is timed REPEATS times per method (methods interleaved so
    both see the same GPU/CPU state), and the per-question median is kept;
  - extractive answer selection is timed separately.

Writes 03_result/result-latency.csv (summary) and
03_result/result-latency-per-query.csv.
"""
import time

import numpy as np
import pandas as pd

import config
from qa_pipeline import generate_answer
from retriever import build_or_load_index

WARMUP_QUERIES = 10
REPEATS = 3


def summarize(name: str, values: np.ndarray) -> dict:
    return {
        "method": name,
        "n_queries": len(values),
        "mean_ms": round(float(values.mean()), 2),
        "median_ms": round(float(np.median(values)), 2),
        "p90_ms": round(float(np.percentile(values, 90)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "max_ms": round(float(values.max()), 2),
    }


def measure_latency(retriever, corpus):
    """Time dense-only retrieval, graph retrieval and answer selection on the
    corpus's seeded evaluation sample; returns per-query medians in ms."""
    eval_sample = corpus.sample(n=min(config.NUM_EVAL_QUERIES, len(corpus)),
                                random_state=config.RANDOM_SEED).reset_index(drop=True)
    questions = eval_sample["question"].tolist()

    for q in corpus.sample(n=WARMUP_QUERIES, random_state=config.RANDOM_SEED + 1)["question"]:
        retriever.retrieve(q, use_graph=True)
        retriever.retrieve(q, use_graph=False)

    dense, graph, answer = [], [], []
    for q in questions:
        d_runs, g_runs, a_runs = [], [], []
        for _ in range(REPEATS):
            d_runs.append(retriever.retrieve(q, use_graph=False).latency_ms)
            result = retriever.retrieve(q, use_graph=True)
            g_runs.append(result.latency_ms)
            context_docs = retriever.get_docs(result.doc_ids)
            a_runs.append(generate_answer(q, context_docs, use_llm=False).latency_ms)
        dense.append(np.median(d_runs))
        graph.append(np.median(g_runs))
        answer.append(np.median(a_runs))

    return eval_sample, np.asarray(dense), np.asarray(graph), np.asarray(answer)


def main():
    retriever, corpus, _, _ = build_or_load_index(force_rebuild=False, max_docs=config.MAX_DOCS)
    eval_sample, dense, graph, answer = measure_latency(retriever, corpus)
    summary = pd.DataFrame([
        summarize("BGE-M3 dense-only retrieval", dense),
        summarize("GraphRAG-lite + BGE-M3 retrieval", graph),
        summarize("Extractive answer selection", answer),
        summarize("GraphRAG-lite + BGE-M3 end-to-end", graph + answer),
    ])
    per_query = pd.DataFrame({"query_id": eval_sample["doc_id"], "dense_ms": dense,
                              "graph_ms": graph, "answer_ms": answer})

    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RESULT_DIR / "result-latency.csv", index=False, encoding="utf-8-sig")
    per_query.to_csv(config.RESULT_DIR / "result-latency-per-query.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nthroughput (end-to-end, sequential): {1000 / (graph + answer).mean():.1f} queries/s")


if __name__ == "__main__":
    main()
