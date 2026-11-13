"""End-to-end experiment runner for the GraphRAG + BGE-M3 medical QA pipeline.

Usage (from the project root or from 02_code/):
    python 02_code/run_experiment.py                       # full cleaned dataset (~16.3k docs)
    python 02_code/run_experiment.py --max-docs 2000 --num-eval 50 --rebuild
    python 02_code/run_experiment.py --use-llm            # needs OPENAI_API_KEY

Writes:
    03_result/result-rag.csv          per-question retrieval + answer metrics
    03_result/result-rag-summary.csv  aggregated metrics for the whole run
"""
import argparse
import time

import pandas as pd
from tqdm import tqdm

import config
import metrics
from qa_pipeline import generate_answer
from retriever import build_or_load_index, graph_stats


def parse_args():
    p = argparse.ArgumentParser(description="Run the GraphRAG + BGE-M3 medical QA experiment")
    p.add_argument("--max-docs", type=int, default=config.MAX_DOCS,
                    help="cap corpus size (-1 for the full cleaned dataset)")
    p.add_argument("--num-eval", type=int, default=config.NUM_EVAL_QUERIES,
                    help="number of questions sampled for evaluation")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--rebuild", action="store_true", help="force rebuild of cached embeddings/graph")
    p.add_argument("--use-llm", action="store_true",
                    help="synthesize answers with an OpenAI chat model (requires OPENAI_API_KEY); "
                         "falls back to extractive answers otherwise")
    return p.parse_args()


def run(args):
    t_build0 = time.perf_counter()
    retriever, corpus, graph, embedder = build_or_load_index(force_rebuild=args.rebuild, max_docs=args.max_docs)
    build_time_s = time.perf_counter() - t_build0

    stats = graph_stats(graph)
    print(f"[run] corpus size: {len(corpus)} | entities: {stats['num_entities']} | "
          f"graph edges: {stats['num_edges']} | communities: {stats['num_communities']} | "
          f"index build time: {build_time_s:.1f}s")

    eval_sample = corpus.sample(n=min(args.num_eval, len(corpus)), random_state=args.seed).reset_index(drop=True)

    rows = []
    for _, row in tqdm(eval_sample.iterrows(), total=len(eval_sample), desc="evaluating"):
        question = row["question"]
        gold_answer = row["answer"]
        gold_doc_id = int(row["doc_id"])
        gold_focus_area = row["focus_area"]

        retrieval = retriever.retrieve(question)
        context_docs = retriever.get_docs(retrieval.doc_ids)

        answer = generate_answer(question, context_docs, use_llm=args.use_llm)

        hit_metrics = metrics.retrieval_hit_metrics(gold_doc_id, retrieval.doc_ids)
        rouge_score = metrics.rouge_l_f1(answer.text, gold_answer)
        semantic_sim = metrics.semantic_cosine_similarity(embedder, answer.text, gold_answer)

        retrieved_focus_areas = context_docs["focus_area"].tolist() if len(context_docs) else []

        # All 3 evaluation-technique scores are reported on a 0-100% scale
        # (binary hit flags become 0% / 100%, ROUGE-L and cosine similarity
        # are their natural [0, 1] score * 100), rounded to 2 decimals.
        rows.append({
            "query_id": int(row["doc_id"]),
            "question": question,
            "gold_focus_area": gold_focus_area,
            "gold_source": row["source"],
            "linked_entities": " | ".join(retrieval.linked_entities),
            "retrieved_doc_ids": " | ".join(map(str, retrieval.doc_ids)),
            "retrieved_focus_areas": " | ".join(retrieved_focus_areas),
            "hit_at_1_pct": round(hit_metrics["hit_at_1"] * 100, 2),
            "hit_at_5_pct": round(hit_metrics["hit_at_5"] * 100, 2),
            "reciprocal_rank_pct": round(hit_metrics["reciprocal_rank"] * 100, 2),
            "rouge_l_f1_pct": round(rouge_score * 100, 2),
            "semantic_cosine_sim_pct": round(semantic_sim * 100, 2),
            "retrieval_latency_ms": retrieval.latency_ms,
            "generation_latency_ms": answer.latency_ms,
            "generation_mode": answer.mode,
            "generated_answer": answer.text[: config.ANSWER_CHAR_LIMIT],
            "gold_answer": gold_answer[: config.ANSWER_CHAR_LIMIT],
        })

    results_df = pd.DataFrame(rows)
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.RESULT_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "corpus_size": len(corpus),
        "num_entities": stats["num_entities"],
        "num_graph_edges": stats["num_edges"],
        "num_communities": stats["num_communities"],
        "index_build_time_s": round(build_time_s, 2),
        "num_eval_queries": len(results_df),
        "hit_rate_at_1_pct": round(results_df["hit_at_1_pct"].mean(), 2),
        "hit_rate_at_5_pct": round(results_df["hit_at_5_pct"].mean(), 2),
        "mrr_pct": round(results_df["reciprocal_rank_pct"].mean(), 2),
        "avg_rouge_l_f1_pct": round(results_df["rouge_l_f1_pct"].mean(), 2),
        "avg_semantic_cosine_sim_pct": round(results_df["semantic_cosine_sim_pct"].mean(), 2),
        "avg_retrieval_latency_ms": round(results_df["retrieval_latency_ms"].mean(), 3),
        "avg_generation_latency_ms": round(results_df["generation_latency_ms"].mean(), 3),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(config.SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k:28s}: {v}")
    print(f"\nPer-question results -> {config.RESULT_CSV}")
    print(f"Summary               -> {config.SUMMARY_CSV}")


if __name__ == "__main__":
    run(parse_args())
