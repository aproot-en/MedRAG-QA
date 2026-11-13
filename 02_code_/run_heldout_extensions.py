"""Held-out extensions: document length and a biomedical encoder.

Re-runs the held-out protocol of run_review_experiments.py (same seed, same
development/test split, H fixed at the development-selected value) for:

  * BGE-M3 indexes rebuilt at 500, 1,000 and 1,500 characters per document
    (3,000 reuses the main cache in 02_code/artifacts);
  * a BioBERT-based sentence-embedding encoder at 3,000 characters, with the
    encoder's 512-token limit applied.

For every configuration it reports dense-only, entity-only and entity+neighbour
(DEG-DR) retrieval, plus paired tests of DEG-DR against (a) dense-only in the
same configuration and (b) the default BGE-M3 / 3,000-character DEG-DR.
New embeddings are cached under 02_code/artifacts_ext/<config>/ so an
interrupted run resumes where it stopped.

Writes 03_result/result-heldout-extensions.csv and
03_result/result-heldout-extensions-paired.csv.
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data_prep import load_clean_corpus
from graph_builder import build_entity_graph
from retriever import GraphRagRetriever
from run_review_experiments import (DEV_FRACTION, SEED, dense_rankings, graph_rankings,
                                    paired_comparison, rank_metrics, summarize)

EXT_DIR = config.BASE_DIR / "artifacts_ext"
BIO_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
SELECTED_H = json.loads((config.RESULT_DIR / "result-review-metadata.json").read_text())["selected_h"]

CONFIGS = [
    # name, encoder, chunk_chars
    ("BGE-M3 / 3000", "bge-m3", 3000),
    ("BGE-M3 / 1500", "bge-m3", 1500),
    ("BGE-M3 / 1000", "bge-m3", 1000),
    ("BGE-M3 / 500", "bge-m3", 500),
    ("BioBERT-ST / 3000", "biobert", 3000),
]


class StEmbedder:
    """Sentence-transformers encoder with the same interface as BgeM3Embedder."""

    def __init__(self, model_name: str, max_seq_length: int = 512):
        import transformers.modeling_utils as mu
        import transformers.utils.import_utils as iu
        iu.check_torch_load_is_safe = lambda *a, **k: None  # same .bin guard as embedder.py
        mu.check_torch_load_is_safe = lambda *a, **k: None
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length

    def encode(self, texts, batch_size: int = config.EMBED_BATCH_SIZE, show_progress: bool = True):
        return self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress, convert_to_numpy=True).astype(np.float32)


def make_embedder(encoder: str):
    if encoder == "bge-m3":
        from embedder import BgeM3Embedder
        return BgeM3Embedder()
    return StEmbedder(BIO_MODEL)


def load_index(name: str, encoder: str, chunk_chars: int, embedder):
    """Return (corpus, doc_emb, entity_names, entity_emb, graph), building and caching if needed."""
    corpus = load_clean_corpus(max_docs=-1, chunk_chars=chunk_chars)
    if encoder == "bge-m3" and chunk_chars == config.DOC_CHUNK_CHARS:
        d = config.ARTIFACT_DIR
    else:
        d = EXT_DIR / name.replace(" / ", "_").replace("-", "")
    d.mkdir(parents=True, exist_ok=True)
    paths = {k: d / f for k, f in [("doc", "doc_embeddings.npy"), ("ent", "entity_embeddings.npy"),
                                   ("names", "entity_names.json"), ("graph", "graph.pkl")]}
    if all(p.exists() for p in paths.values()):
        print(f"[{name}] loading cached index from {d}")
        doc_emb = np.load(paths["doc"])
        entity_emb = np.load(paths["ent"])
        entity_names = json.loads(paths["names"].read_text())
        with open(paths["graph"], "rb") as f:
            graph = pickle.load(f)
    else:
        print(f"[{name}] building index in {d}")
        t0 = time.perf_counter()
        graph = build_entity_graph(corpus)
        doc_emb = embedder.encode(corpus["doc_text"].tolist())
        entity_names = sorted(corpus["focus_area"].unique().tolist())
        entity_emb = embedder.encode(entity_names)
        np.save(paths["doc"], doc_emb)
        np.save(paths["ent"], entity_emb)
        paths["names"].write_text(json.dumps(entity_names))
        with open(paths["graph"], "wb") as f:
            pickle.dump(graph, f)
        print(f"[{name}] built in {time.perf_counter() - t0:.0f}s")
    return corpus, doc_emb, entity_names, entity_emb, graph


def heldout_split(corpus: pd.DataFrame) -> pd.DataFrame:
    """Identical to run_review_experiments.main(): returns the held-out test set."""
    original_100 = set(corpus.sample(n=100, random_state=SEED)["doc_id"].astype(int))
    remaining = corpus.loc[~corpus["doc_id"].isin(original_100)]
    _, test = train_test_split(remaining, test_size=1.0 - DEV_FRACTION, random_state=SEED,
                               stratify=remaining["source"])
    return test.sort_values("doc_id")


def main() -> None:
    only = set(sys.argv[1:])  # optional: restrict to config names
    summary_rows, paired_rows = [], []
    reference = None  # DEG-DR metrics of the default configuration
    embedders = {}
    for name, encoder, chunk_chars in CONFIGS:
        if only and name not in only and reference is not None:
            continue
        if encoder not in embedders:
            embedders[encoder] = make_embedder(encoder)
        embedder = embedders[encoder]
        corpus, doc_emb, entity_names, entity_emb, graph = load_index(name, encoder, chunk_chars, embedder)
        retriever = GraphRagRetriever(corpus, doc_emb, graph, entity_names, entity_emb, embedder)

        test = heldout_split(corpus)
        gold = test["doc_id"].to_numpy()
        query_vecs = embedder.encode(test["question"].tolist(), show_progress=False)
        dense = dense_rankings(retriever.index, retriever.doc_ids, query_vecs)
        entity = graph_rankings(retriever, query_vecs, dense, h=SELECTED_H, include_neighbors=False)
        degdr = graph_rankings(retriever, query_vecs, dense, h=SELECTED_H, include_neighbors=True)

        values = {"dense": rank_metrics(dense, gold), "entity": rank_metrics(entity, gold),
                  "degdr": rank_metrics(degdr, gold)}
        for key, label in [("dense", "Dense-only"), ("entity", "Dense+entity"), ("degdr", "DEG-DR")]:
            summary_rows.append({"config": name, "encoder": encoder, "chunk_chars": chunk_chars,
                                 **summarize(label, values[key])})
        paired_rows.append({"config": name, **paired_comparison("Dense-only", "DEG-DR",
                                                                values["dense"], values["degdr"])})
        if reference is None:
            reference = values["degdr"]
        else:
            paired_rows.append({"config": name, **paired_comparison("DEG-DR BGE-M3/3000", f"DEG-DR {name}",
                                                                    reference, values["degdr"])})
        print(pd.DataFrame(summary_rows[-3:]).to_string(index=False))

        out = pd.DataFrame(summary_rows)
        out.to_csv(config.RESULT_DIR / "result-heldout-extensions.csv", index=False)
        pd.DataFrame(paired_rows).to_csv(config.RESULT_DIR / "result-heldout-extensions-paired.csv", index=False)
        del retriever, doc_emb

    print("\nPaired tests\n", pd.DataFrame(paired_rows).to_string(index=False))


if __name__ == "__main__":
    main()
