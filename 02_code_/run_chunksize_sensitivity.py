"""Chunk-size (document character-limit) sensitivity study.

`data_prep.py` truncates each document's indexed text (focus_area + answer,
Eq. 1 in the paper) to `config.DOC_CHUNK_CHARS` characters before embedding --
this is the pipeline's "chunk size" knob. The main results use 3,000
characters. This script rebuilds independent indices at several other chunk
sizes on the full cleaned corpus (16,345 docs) and evaluates each on the same
100-question sample, to see how much truncating (or not truncating) document
text actually matters for retrieval quality.

Each chunk size needs its own embeddings (changing chunk_chars changes
doc_text, so nothing can be reused from the main cache), so each config is
built into its own scratch artifact directory -- the main 02_code/artifacts
cache used by run_experiment.py / run_ablation.py is never touched.
Writes 03_result/result-chunksize.csv.
"""
import time

import config
import pandas as pd

import metrics
from qa_pipeline import generate_answer
from retriever import build_or_load_index, graph_stats

SCRATCH_ARTIFACT_ROOT = config.BASE_DIR / "artifacts_chunksize"

CHUNK_SIZES = [500, 1000, 1500]  # 3000 (the default) reuses the main cache's existing results;
# 6000 was attempted but hit this machine's 6 GB VRAM limit (batches slowed to ~120s each) and was abandoned


def evaluate(retriever, embedder, eval_sample) -> dict:
    rows = []
    for _, row in eval_sample.iterrows():
        question = row["question"]
        gold_answer = row["answer"]
        gold_doc_id = int(row["doc_id"])

        retrieval = retriever.retrieve(question, use_graph=True)
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


def truncated_share(corpus: pd.DataFrame, chunk_chars: int) -> float:
    """Share of documents whose focus_area+answer text is actually longer than
    chunk_chars (i.e. would be cut off), for context in the results table."""
    raw_len = (corpus["focus_area"].astype(str) + ". " + corpus["answer"].astype(str)).str.len()
    return round(float((raw_len > chunk_chars).mean() * 100), 1)


def run_one(chunk_chars: int) -> dict:
    config.ARTIFACT_DIR = SCRATCH_ARTIFACT_ROOT / str(chunk_chars)
    t0 = time.perf_counter()
    retriever, corpus, graph, embedder = build_or_load_index(
        force_rebuild=False, max_docs=config.MAX_DOCS, chunk_chars=chunk_chars)
    build_time_s = time.perf_counter() - t0

    eval_sample = corpus.sample(n=min(config.NUM_EVAL_QUERIES, len(corpus)),
                                 random_state=config.RANDOM_SEED).reset_index(drop=True)
    result = evaluate(retriever, embedder, eval_sample)
    result["chunk_chars"] = chunk_chars
    result["pct_docs_truncated"] = truncated_share(corpus, chunk_chars)
    result["build_time_s"] = round(build_time_s, 1)
    return result


def main():
    original_artifact_dir = config.ARTIFACT_DIR
    rows = []
    try:
        for size in CHUNK_SIZES:
            print(f"\n=== chunk size: {size} chars ===")
            rows.append(run_one(size))
    finally:
        config.ARTIFACT_DIR = original_artifact_dir

    # 3,000-char default: reuse the already-computed main-cache results (Table II / III).
    main_summary = pd.read_csv(config.SUMMARY_CSV).iloc[0]
    corpus_for_share = None
    from data_prep import load_clean_corpus
    corpus_default = load_clean_corpus(max_docs=config.MAX_DOCS, chunk_chars=config.DOC_CHUNK_CHARS)
    rows.append({
        "hit_rate_at_1_pct": main_summary["hit_rate_at_1_pct"],
        "hit_rate_at_5_pct": main_summary["hit_rate_at_5_pct"],
        "mrr_pct": main_summary["mrr_pct"],
        "avg_rouge_l_f1_pct": main_summary["avg_rouge_l_f1_pct"],
        "avg_semantic_cosine_sim_pct": main_summary["avg_semantic_cosine_sim_pct"],
        "avg_retrieval_latency_ms": main_summary["avg_retrieval_latency_ms"],
        "chunk_chars": config.DOC_CHUNK_CHARS,
        "pct_docs_truncated": truncated_share(corpus_default, config.DOC_CHUNK_CHARS),
        "build_time_s": main_summary["index_build_time_s"],
    })

    out_df = pd.DataFrame(rows).sort_values("chunk_chars").reset_index(drop=True)
    out_df = out_df[["chunk_chars", "pct_docs_truncated", "build_time_s", "hit_rate_at_1_pct",
                      "hit_rate_at_5_pct", "mrr_pct", "avg_rouge_l_f1_pct",
                      "avg_semantic_cosine_sim_pct", "avg_retrieval_latency_ms"]]
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULT_DIR / "result-chunksize.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n" + out_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
