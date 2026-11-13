"""Three evaluation techniques for the GraphRAG + BGE-M3 pipeline.

1. Retrieval accuracy  - Hit Rate@1, Hit Rate@5 and Reciprocal Rank: does the
   gold document (the document the eval question was taken from) appear
   among the retrieved candidates, and how highly ranked is it.
2. Lexical answer quality - ROUGE-L F1 between the generated answer and the
   gold (reference) answer: measures textual/n-gram overlap.
3. Semantic answer quality - cosine similarity between the BGE-M3 embeddings
   of the generated answer and the gold answer: measures meaning overlap
   even when wording differs, complementing the purely lexical ROUGE-L score.
"""
from typing import List

import numpy as np
from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def retrieval_hit_metrics(gold_doc_id: int, retrieved_doc_ids: List[int]) -> dict:
    hit_at_1 = int(len(retrieved_doc_ids) > 0 and retrieved_doc_ids[0] == gold_doc_id)
    hit_at_5 = int(gold_doc_id in retrieved_doc_ids[:5])
    if gold_doc_id in retrieved_doc_ids:
        rank = retrieved_doc_ids.index(gold_doc_id) + 1
        reciprocal_rank = 1.0 / rank
    else:
        reciprocal_rank = 0.0
    return {"hit_at_1": hit_at_1, "hit_at_5": hit_at_5, "reciprocal_rank": reciprocal_rank}


def rouge_l_f1(generated_answer: str, gold_answer: str) -> float:
    if not generated_answer or not gold_answer:
        return 0.0
    return _rouge.score(gold_answer, generated_answer)["rougeL"].fmeasure


def semantic_cosine_similarity(embedder, generated_answer: str, gold_answer: str) -> float:
    if not generated_answer or not gold_answer:
        return 0.0
    vecs = embedder.encode([generated_answer, gold_answer], show_progress=False)
    return float(np.dot(vecs[0], vecs[1]))
