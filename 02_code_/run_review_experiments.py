"""Reviewer-driven, paired retrieval evaluation on the full MedQuAD corpus.

This script addresses the main validity concerns raised after the initial
100-query experiment:

* the original 100 queries are assigned to development and never reported as
  confirmatory test cases;
* H (the number of linked entities) is selected on development only;
* dense-only, entity-only, entity+neighbour graph fusion, BM25, and
  dense+BM25 RRF are evaluated on exactly the same held-out queries;
* paired win/loss/tie counts, exact McNemar tests, and bootstrap confidence
  intervals are reported;
* question-type, answer-leakage, and random/same-condition semantic baselines
  are measured for error analysis.

The full BGE-M3 document index is reused.  Query embeddings are encoded once
in batches, so the experiment is substantially faster than calling the online
retriever once per method and query.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import config
from retriever import build_or_load_index


SEED = config.RANDOM_SEED
DEV_FRACTION = 0.20
BOOTSTRAP_REPLICATES = 10_000
BM25_TOP_K = 20
H_CANDIDATES = (1, 2, 3, 4, 5)
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def rrf(rankings: list[list[int]], k: int = config.RRF_K, top_k: int = 20) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def rank_metrics(rankings: list[list[int]], gold: np.ndarray) -> dict[str, np.ndarray]:
    ranks = np.zeros(len(rankings), dtype=np.int32)
    for i, (ranking, target) in enumerate(zip(rankings, gold)):
        try:
            ranks[i] = ranking.index(int(target)) + 1
        except ValueError:
            pass
    return {
        "rank": ranks,
        "hit1": ranks == 1,
        "hit5": (ranks > 0) & (ranks <= 5),
        "hit10": (ranks > 0) & (ranks <= 10),
        "hit20": (ranks > 0) & (ranks <= 20),
        "rr": np.divide(1.0, ranks, out=np.zeros_like(ranks, dtype=float), where=ranks > 0),
    }


def summarize(name: str, values: dict[str, np.ndarray]) -> dict[str, float | str | int]:
    return {
        "method": name,
        "n_queries": len(values["rank"]),
        "hit_at_1_pct": 100 * values["hit1"].mean(),
        "hit_at_5_pct": 100 * values["hit5"].mean(),
        "hit_at_10_pct": 100 * values["hit10"].mean(),
        "hit_at_20_pct": 100 * values["hit20"].mean(),
        "mrr_at_20_pct": 100 * values["rr"].mean(),
    }


def bootstrap_ci(x: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        stop = min(start + 250, BOOTSTRAP_REPLICATES)
        sample = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = x[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_comparison(base_name: str, method_name: str, base: dict, method: dict) -> dict:
    b1, m1 = base["hit1"], method["hit1"]
    losses = int(np.sum(b1 & ~m1))
    wins = int(np.sum(~b1 & m1))
    discordant = wins + losses
    p = 1.0 if discordant == 0 else float(binomtest(min(wins, losses), discordant, 0.5).pvalue)
    hit_diff = method["hit1"].astype(float) - base["hit1"].astype(float)
    rr_diff = method["rr"] - base["rr"]
    hit_ci = bootstrap_ci(hit_diff)
    rr_ci = bootstrap_ci(rr_diff, seed=SEED + 1)
    return {
        "baseline": base_name,
        "method": method_name,
        "top1_wins": wins,
        "top1_losses": losses,
        "top1_ties": int(len(b1) - discordant),
        "hit1_difference_pp": 100 * hit_diff.mean(),
        "hit1_diff_ci95_low_pp": 100 * hit_ci[0],
        "hit1_diff_ci95_high_pp": 100 * hit_ci[1],
        "mcnemar_exact_p": p,
        "mrr_difference_pp": 100 * rr_diff.mean(),
        "mrr_diff_ci95_low_pp": 100 * rr_ci[0],
        "mrr_diff_ci95_high_pp": 100 * rr_ci[1],
    }


def dense_rankings(index: faiss.Index, doc_ids: np.ndarray, query_vecs: np.ndarray, top_k: int = 20):
    _, positions = index.search(query_vecs.astype(np.float32), top_k)
    return [doc_ids[row].astype(int).tolist() for row in positions]


def graph_rankings(retriever, query_vecs: np.ndarray, dense: list[list[int]], h: int,
                   include_neighbors: bool) -> list[list[int]]:
    results = []
    entity_matrix = retriever.entity_embeddings
    batch_size = 256
    for start in range(0, len(query_vecs), batch_size):
        stop = min(start + batch_size, len(query_vecs))
        link_scores = query_vecs[start:stop] @ entity_matrix.T
        linked_pos = np.argpartition(link_scores, -h, axis=1)[:, -h:]
        linked_pos = np.take_along_axis(
            linked_pos,
            np.argsort(np.take_along_axis(link_scores, linked_pos, axis=1), axis=1)[:, ::-1],
            axis=1,
        )
        for local_i, positions in enumerate(linked_pos):
            q = query_vecs[start + local_i]
            candidates: set[int] = set()
            for pos in positions:
                entity = retriever.entity_names[int(pos)]
                candidates.update(retriever.graph.nodes[entity].get("doc_ids", []))
                if include_neighbors:
                    for neighbor in retriever.graph.neighbors(entity):
                        candidates.update(retriever.graph.nodes[neighbor].get("doc_ids", []))
            candidate_ids = list(candidates)
            if candidate_ids:
                doc_pos = np.fromiter((retriever.doc_id_to_pos[d] for d in candidate_ids), dtype=np.int64)
                scores = retriever.doc_embeddings[doc_pos] @ q
                order = np.argsort(scores)[::-1]
                graph_list = [candidate_ids[i] for i in order]
            else:
                graph_list = []
            results.append(rrf([dense[start + local_i], graph_list], top_k=20))
    return results


def bm25_rankings(corpus: pd.DataFrame, questions: list[str], top_k: int = BM25_TOP_K) -> list[list[int]]:
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", min_df=2)
    counts = vectorizer.fit_transform(corpus["doc_text"].tolist()).astype(np.float32).tocsr()
    n_docs = counts.shape[0]
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    avg_len = float(doc_len.mean())
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    k1, b = 1.5, 0.75
    rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    denom = counts.data + k1 * (1.0 - b + b * doc_len[rows] / avg_len)
    counts.data = counts.data * (k1 + 1.0) / denom
    counts = counts.multiply(idf).tocsr()

    results: list[list[int]] = []
    doc_ids = corpus["doc_id"].to_numpy()
    for start in range(0, len(questions), 128):
        q = vectorizer.transform(questions[start:start + 128])
        q.data[:] = 1.0
        scores = (q @ counts.T).toarray()
        positions = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        for row_scores, row_pos in zip(scores, positions):
            ordered = row_pos[np.argsort(row_scores[row_pos])[::-1]]
            results.append(doc_ids[ordered].astype(int).tolist())
    return results


def infer_question_type(question: str) -> str:
    q = question.lower().strip()
    patterns = (
        ("symptoms", r"symptom|signs? of"), ("treatment", r"treat|therapy|management"),
        ("cause", r"cause|risk factor"), ("diagnosis", r"diagnos|test(?:ing)? for"),
        ("prevention", r"prevent"), ("inheritance", r"inherit|genetic"),
        ("definition", r"^what (?:is|are)"), ("outlook", r"prognosis|outlook"),
    )
    for label, pattern in patterns:
        if re.search(pattern, q):
            return label
    return "other"


def leakage_analysis(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in corpus.groupby("source"):
        exact = []
        token_overlap = []
        for row in group.itertuples():
            q = " ".join(TOKEN_RE.findall(str(row.question).lower()))
            a = " ".join(TOKEN_RE.findall(str(row.answer).lower()))
            exact.append(bool(q) and q in a)
            q_tokens, a_tokens = set(q.split()), set(a.split())
            token_overlap.append(len(q_tokens & a_tokens) / max(1, len(q_tokens)))
        rows.append({"source": source, "n": len(group), "exact_question_in_answer_pct": 100 * np.mean(exact),
                     "mean_question_token_recall_pct": 100 * np.mean(token_overlap)})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def semantic_baselines(embedder, corpus: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    eligible = corpus.groupby("focus_area").filter(lambda g: len(g) >= 2)
    anchors = eligible.sample(n=min(n, len(eligible)), random_state=SEED)
    random_answers = corpus.sample(n=len(anchors), random_state=SEED + 1)["answer"].tolist()
    same_answers = []
    for row in anchors.itertuples():
        pool = corpus[(corpus["focus_area"] == row.focus_area) & (corpus["doc_id"] != row.doc_id)]
        same_answers.append(pool.sample(n=1, random_state=SEED + int(row.doc_id))["answer"].iloc[0])
    # Match the indexed representation limit and use a small batch so a few
    # very long medical answers cannot exhaust a 6 GB GPU through padding.
    anchor_text = anchors["answer"].str.slice(0, config.DOC_CHUNK_CHARS).tolist()
    random_answers = [text[:config.DOC_CHUNK_CHARS] for text in random_answers]
    same_answers = [text[:config.DOC_CHUNK_CHARS] for text in same_answers]
    anchor_vec = embedder.encode(anchor_text, batch_size=4, show_progress=True)
    random_vec = embedder.encode(random_answers, batch_size=4, show_progress=True)
    same_vec = embedder.encode(same_answers, batch_size=4, show_progress=True)
    return pd.DataFrame([
        {"pair_type": "random answers", "n_pairs": len(anchors), "mean_cosine_pct": 100 * np.sum(anchor_vec * random_vec, axis=1).mean()},
        {"pair_type": "different answers, same focus_area", "n_pairs": len(anchors), "mean_cosine_pct": 100 * np.sum(anchor_vec * same_vec, axis=1).mean()},
    ])


def main() -> None:
    retriever, corpus, _, embedder = build_or_load_index(force_rebuild=False, max_docs=-1)
    original_100 = set(corpus.sample(n=100, random_state=SEED)["doc_id"].astype(int))
    remaining = corpus.loc[~corpus["doc_id"].isin(original_100)]
    extra_dev, test = train_test_split(
        remaining, test_size=1.0 - DEV_FRACTION, random_state=SEED,
        stratify=remaining["source"],
    )
    dev = pd.concat([corpus[corpus["doc_id"].isin(original_100)], extra_dev]).sample(frac=1, random_state=SEED)
    test = test.sort_values("doc_id")

    all_eval = pd.concat([dev.assign(split="dev"), test.assign(split="test")])
    query_vecs = embedder.encode(all_eval["question"].tolist(), show_progress=True)
    dense_all = dense_rankings(retriever.index, retriever.doc_ids, query_vecs)
    n_dev = len(dev)
    dev_gold = dev["doc_id"].to_numpy()
    test_gold = test["doc_id"].to_numpy()

    dev_scores = []
    graph_by_h: dict[int, list[list[int]]] = {}
    for h in H_CANDIDATES:
        rankings = graph_rankings(retriever, query_vecs, dense_all, h=h, include_neighbors=True)
        graph_by_h[h] = rankings
        metrics = rank_metrics(rankings[:n_dev], dev_gold)
        dev_scores.append({"h": h, **summarize(f"entity+neighbors H={h}", metrics)})
    dev_df = pd.DataFrame(dev_scores)
    selected_h = int(dev_df.sort_values(["mrr_at_20_pct", "hit_at_1_pct", "h"], ascending=[False, False, True]).iloc[0]["h"])

    dense_test = dense_all[n_dev:]
    entity_all = graph_rankings(retriever, query_vecs, dense_all, h=selected_h, include_neighbors=False)
    graph_test = graph_by_h[selected_h][n_dev:]
    entity_test = entity_all[n_dev:]
    bm25_test = bm25_rankings(corpus, test["question"].tolist())
    dense_bm25_test = [rrf([d, b], top_k=20) for d, b in zip(dense_test, bm25_test)]

    rankings = {
        "BGE-M3 dense": dense_test,
        "BM25": bm25_test,
        "Dense+BM25 RRF": dense_bm25_test,
        f"Dense+entity RRF (H={selected_h})": entity_test,
        f"Dense+entity+neighbors RRF (H={selected_h})": graph_test,
    }
    metric_arrays = {name: rank_metrics(ranks, test_gold) for name, ranks in rankings.items()}
    summary = pd.DataFrame([summarize(name, values) for name, values in metric_arrays.items()])

    dense_values = metric_arrays["BGE-M3 dense"]
    comparisons = pd.DataFrame([
        paired_comparison("BGE-M3 dense", name, dense_values, values)
        for name, values in metric_arrays.items() if name != "BGE-M3 dense"
    ])

    proposed_name = f"Dense+entity+neighbors RRF (H={selected_h})"
    proposed = metric_arrays[proposed_name]
    error_rows = []
    typed = test.assign(question_type=test["question"].map(infer_question_type)).reset_index(drop=True)
    for qtype, idx in typed.groupby("question_type").groups.items():
        pos = np.asarray(list(idx), dtype=int)
        error_rows.append({"question_type": qtype, "n": len(pos),
                           "dense_hit1_pct": 100 * dense_values["hit1"][pos].mean(),
                           "graph_hit1_pct": 100 * proposed["hit1"][pos].mean(),
                           "graph_hit5_pct": 100 * proposed["hit5"][pos].mean()})
    error_df = pd.DataFrame(error_rows).sort_values("n", ascending=False)

    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dev_df.to_csv(config.RESULT_DIR / "result-review-dev-h.csv", index=False)
    summary.to_csv(config.RESULT_DIR / "result-review-heldout.csv", index=False)
    comparisons.to_csv(config.RESULT_DIR / "result-review-paired-tests.csv", index=False)
    error_df.to_csv(config.RESULT_DIR / "result-review-question-types.csv", index=False)
    leakage_analysis(corpus).to_csv(config.RESULT_DIR / "result-review-leakage.csv", index=False)
    metadata = {
        "seed": SEED, "corpus_size": len(corpus), "development_queries": len(dev),
        "heldout_test_queries": len(test), "original_100_forced_into_development": 100,
        "selected_h": selected_h, "selection_rule": "highest development MRR@20; then Hit@1; then smaller H",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (config.RESULT_DIR / "result-review-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save the more expensive retrieval results before the auxiliary semantic
    # baseline so an interrupted GPU run does not discard the core experiment.
    semantic_baselines(embedder, corpus).to_csv(config.RESULT_DIR / "result-review-semantic-baselines.csv", index=False)

    print("\nDevelopment H selection\n", dev_df.to_string(index=False))
    print("\nHeld-out results\n", summary.to_string(index=False))
    print("\nPaired tests vs dense\n", comparisons.to_string(index=False))
    print("\nMetadata\n", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
