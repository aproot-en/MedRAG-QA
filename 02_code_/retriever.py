"""Hybrid GraphRAG + BGE-M3 retriever, with on-disk caching of built artifacts.

Retrieval pipeline for a query:
  1. Dense search: embed the query with BGE-M3, cosine-search all document
     embeddings (FAISS inner-product index over normalized vectors) for the
     top `DENSE_TOP_K` candidates -> a dense ranking.
  2. Entity linking: cosine-match the query embedding against entity (focus
     area) embeddings, take the top `GRAPH_HOP_ENTITIES` linked entities.
  3. Graph expansion: walk one hop from each linked entity over the
     similarity graph built in `graph_builder.py`, collect the documents
     belonging to the linked entities and their neighbors, then rank that
     set by cosine similarity to the query -> a graph ranking.
  4. Fuse the two rankings with Reciprocal Rank Fusion (RRF) and keep the
     top `FINAL_TOP_K`.

Note on why RRF instead of re-ranking the dense-union-graph pool by raw
cosine similarity: `dense_doc_ids` is, by construction, already the set of
`DENSE_TOP_K` documents with the globally highest cosine similarity to the
query. Any graph-only candidate therefore has cosine similarity no higher
than the weakest dense candidate, so a magnitude-based re-rank can never let
a graph-only candidate reach the final top-`FINAL_TOP_K` (since
`FINAL_TOP_K` < `DENSE_TOP_K`) -- the graph branch would be mathematically
inert. RRF instead fuses the two *rankings* rather than the two *score
scales*, so a document that ranks highly in the graph list can outscore a
document that ranks only mediocrely in the dense list, letting graph
provenance genuinely change the output (see `metrics`/Section 2.4 in the
paper for the formula).
"""
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd

import config
from data_prep import load_clean_corpus
from embedder import BgeM3Embedder
from graph_builder import build_entity_graph, graph_stats


@dataclass
class RetrievalResult:
    doc_ids: List[int]
    scores: List[float]
    latency_ms: float
    linked_entities: List[str] = field(default_factory=list)


class GraphRagRetriever:
    def __init__(self, corpus: pd.DataFrame, doc_embeddings: np.ndarray, graph,
                 entity_names: List[str], entity_embeddings: np.ndarray, embedder: BgeM3Embedder):
        self.corpus = corpus.set_index("doc_id", drop=False)
        self.doc_embeddings = doc_embeddings
        self.graph = graph
        self.entity_names = entity_names
        self.entity_embeddings = entity_embeddings
        self.embedder = embedder

        # Online retrieval issues one query at a time. For larger indices FAISS
        # parallelizes that single search with OpenMP, and the spinning worker
        # threads then compete with PyTorch during the next query encoding,
        # roughly doubling end-to-end latency on the full corpus (encode ~35 ms
        # vs ~17 ms). Single-threaded search is faster overall at this scale.
        faiss.omp_set_num_threads(config.FAISS_NUM_THREADS)
        self.index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        self.index.add(doc_embeddings)
        self.doc_ids = corpus["doc_id"].to_numpy()
        self.doc_id_to_pos = {int(d): i for i, d in enumerate(self.doc_ids)}

    def _link_entities(self, query_vec: np.ndarray, top_k: int) -> List[str]:
        sims = self.entity_embeddings @ query_vec
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.entity_names[i] for i in top_idx]

    def _graph_candidate_doc_ids(self, linked_entities: List[str]) -> List[int]:
        candidates = set()
        for entity in linked_entities:
            if entity not in self.graph:
                continue
            candidates.update(self.graph.nodes[entity].get("doc_ids", []))
            for neighbor in self.graph.neighbors(entity):
                candidates.update(self.graph.nodes[neighbor].get("doc_ids", []))
        return list(candidates)

    def _rank_by_similarity(self, doc_ids: List[int], query_vec: np.ndarray) -> List[int]:
        if not doc_ids:
            return []
        positions = [self.doc_id_to_pos[d] for d in doc_ids]
        scores = self.doc_embeddings[positions] @ query_vec
        order = np.argsort(scores)[::-1]
        return [doc_ids[i] for i in order]

    def retrieve(self, query: str, final_top_k: int = config.FINAL_TOP_K, use_graph: bool = True) -> RetrievalResult:
        """Set `use_graph=False` to run the pure dense (BGE-M3-only) baseline,
        i.e. skip entity linking, graph expansion and RRF fusion entirely --
        used by `run_ablation.py` to quantify the graph's contribution."""
        t0 = time.perf_counter()
        query_vec = self.embedder.encode_one(query)

        dense_scores, dense_idx = self.index.search(query_vec.reshape(1, -1), config.DENSE_TOP_K)
        dense_ranked_ids = self.doc_ids[dense_idx[0]].tolist()

        if not use_graph:
            final_doc_ids = dense_ranked_ids[:final_top_k]
            final_scores = dense_scores[0][:final_top_k].tolist()
            latency_ms = (time.perf_counter() - t0) * 1000
            return RetrievalResult(doc_ids=final_doc_ids, scores=final_scores,
                                    latency_ms=latency_ms, linked_entities=[])

        linked_entities = self._link_entities(query_vec, config.GRAPH_HOP_ENTITIES)
        graph_doc_ids = self._graph_candidate_doc_ids(linked_entities)
        graph_ranked_ids = self._rank_by_similarity(graph_doc_ids, query_vec)

        # Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):
        # fuse the dense ranking and the graph ranking by rank position
        # rather than by raw score magnitude (see module docstring for why).
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ranked_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (config.RRF_K + rank)
        for rank, doc_id in enumerate(graph_ranked_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (config.RRF_K + rank)

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:final_top_k]
        final_doc_ids = [doc_id for doc_id, _ in ranked]
        final_scores = [score for _, score in ranked]

        latency_ms = (time.perf_counter() - t0) * 1000
        return RetrievalResult(doc_ids=final_doc_ids, scores=final_scores,
                                latency_ms=latency_ms, linked_entities=linked_entities)

    def get_docs(self, doc_ids: List[int]) -> pd.DataFrame:
        return self.corpus.loc[doc_ids]


def _cache_paths():
    d = config.ARTIFACT_DIR
    return {
        "meta": d / "meta.json",
        "corpus": d / "corpus.parquet",
        "doc_emb": d / "doc_embeddings.npy",
        "entities": d / "entity_names.json",
        "entity_emb": d / "entity_embeddings.npy",
        "graph": d / "graph.pkl",
    }


def _cache_is_valid(paths, expected_meta: dict) -> bool:
    if not all(p.exists() for p in paths.values()):
        return False
    with open(paths["meta"]) as f:
        cached_meta = json.load(f)
    return cached_meta == expected_meta


def build_or_load_index(force_rebuild: bool = False, max_docs: int = config.MAX_DOCS,
                         chunk_chars: int = config.DOC_CHUNK_CHARS):
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths()
    expected_meta = {
        "embed_model": config.EMBED_MODEL_NAME,
        "max_docs": max_docs,
        "seed": config.RANDOM_SEED,
        "graph_topk": config.GRAPH_TOPK_NEIGHBORS,
        "graph_threshold": config.GRAPH_SIM_THRESHOLD,
        "chunk_chars": chunk_chars,
    }

    embedder = BgeM3Embedder()

    if not force_rebuild and _cache_is_valid(paths, expected_meta):
        print("[index] loading cached artifacts from", config.ARTIFACT_DIR)
        corpus = pd.read_parquet(paths["corpus"])
        doc_embeddings = np.load(paths["doc_emb"])
        entity_names = json.loads(paths["entities"].read_text())
        entity_embeddings = np.load(paths["entity_emb"])
        with open(paths["graph"], "rb") as f:
            graph = pickle.load(f)
    else:
        print("[index] building corpus, graph and embeddings from scratch...")
        corpus = load_clean_corpus(max_docs=max_docs, chunk_chars=chunk_chars)

        graph = build_entity_graph(corpus)
        print("[index] graph stats:", graph_stats(graph))

        doc_embeddings = embedder.encode(corpus["doc_text"].tolist())

        entity_names = sorted(corpus["focus_area"].unique().tolist())
        entity_embeddings = embedder.encode(entity_names)

        corpus.to_parquet(paths["corpus"])
        np.save(paths["doc_emb"], doc_embeddings)
        paths["entities"].write_text(json.dumps(entity_names))
        np.save(paths["entity_emb"], entity_embeddings)
        with open(paths["graph"], "wb") as f:
            pickle.dump(graph, f)
        paths["meta"].write_text(json.dumps(expected_meta))

    retriever = GraphRagRetriever(corpus, doc_embeddings, graph, entity_names, entity_embeddings, embedder)
    return retriever, corpus, graph, embedder


if __name__ == "__main__":
    retriever, corpus, graph, embedder = build_or_load_index()
    result = retriever.retrieve("What are the symptoms of diabetes?")
    print("linked entities:", result.linked_entities)
    print(retriever.get_docs(result.doc_ids)[["question", "focus_area"]])
