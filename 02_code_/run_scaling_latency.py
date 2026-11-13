"""Corpus-size latency study with a common timing protocol.

run_corpus_scaling.py measured latency for each corpus size in a separate
session without warm-up, so its latencies are not comparable across sizes.
This script times every size in ONE session with the protocol of
run_latency.py (warm-up, three interleaved runs per query, per-query median).

Reduced-corpus indices are built into 02_code/artifacts_scale/<size>/ (the
main 02_code/artifacts cache is reused for the full corpus and never
rebuilt). Hit@1 and MRR are recomputed as a check that the rebuilt indices
reproduce result-corpus-scaling.csv.

Writes 03_result/result-scaling-latency.csv.
"""
import gc

import numpy as np
import pandas as pd

import config
import metrics
from retriever import build_or_load_index, graph_stats
from run_latency import measure_latency

SIZES = [1000, 2000, 5000, 8000, -1]  # -1 = full cleaned corpus (main cache)
SCALE_ARTIFACT_ROOT = config.BASE_DIR / "artifacts_scale"


def effectiveness(retriever, eval_sample, use_graph: bool) -> dict:
    hits, rrs = [], []
    for row in eval_sample.itertuples():
        result = retriever.retrieve(row.question, use_graph=use_graph)
        m = metrics.retrieval_hit_metrics(int(row.doc_id), result.doc_ids)
        hits.append(m["hit_at_1"])
        rrs.append(m["reciprocal_rank"])
    return {"hit_at_1_pct": round(100 * np.mean(hits), 2), "mrr_pct": round(100 * np.mean(rrs), 2)}


def main():
    main_artifact_dir = config.ARTIFACT_DIR
    rows = []
    try:
        for size in SIZES:
            config.ARTIFACT_DIR = main_artifact_dir if size == -1 else SCALE_ARTIFACT_ROOT / str(size)
            print(f"\n=== max_docs={size} ({config.ARTIFACT_DIR}) ===")
            retriever, corpus, graph, _ = build_or_load_index(force_rebuild=False, max_docs=size)
            stats = graph_stats(graph)

            eval_sample, dense, graph_ms, _ = measure_latency(retriever, corpus)
            dense_eff = effectiveness(retriever, eval_sample, use_graph=False)
            graph_eff = effectiveness(retriever, eval_sample, use_graph=True)

            rows.append({
                "corpus_size": len(corpus),
                "num_entities": stats["num_entities"],
                "num_edges": stats["num_edges"],
                "dense_hit_at_1_pct": dense_eff["hit_at_1_pct"],
                "dense_mrr_pct": dense_eff["mrr_pct"],
                "graph_hit_at_1_pct": graph_eff["hit_at_1_pct"],
                "graph_mrr_pct": graph_eff["mrr_pct"],
                "dense_mean_ms": round(float(dense.mean()), 2),
                "dense_median_ms": round(float(np.median(dense)), 2),
                "graph_mean_ms": round(float(graph_ms.mean()), 2),
                "graph_median_ms": round(float(np.median(graph_ms)), 2),
                "graph_p95_ms": round(float(np.percentile(graph_ms, 95)), 2),
            })
            print(rows[-1])

            del retriever, corpus, graph
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
    finally:
        config.ARTIFACT_DIR = main_artifact_dir

    out = pd.DataFrame(rows)
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULT_DIR / "result-scaling-latency.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n" + out.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
