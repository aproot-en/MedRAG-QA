"""GraphRAG-lite: build a disease/entity relationship graph from the corpus.

Microsoft's original GraphRAG extracts entities and relationships with an
LLM and summarizes graph communities, also with an LLM. That is heavyweight
and non-deterministic, which does not fit a small, reproducible, offline
study. Instead we build a graph deterministically:

  - one node per unique `focus_area` (the disease/condition entity)
  - each entity's text = concatenation of all its indexed texts
  - TF-IDF vectorize entity texts, connect an entity to its top-K most
    cosine-similar neighbor entities above a similarity threshold
  - run greedy-modularity community detection to group related entities

Retrieval then performs entity linking (query -> nearest entity) followed by
graph traversal (entity -> neighbor entities -> their documents) to expand
beyond what pure dense retrieval would find, which is the same "local
search" idea GraphRAG uses, implemented without an LLM in the loop.
"""
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config


def build_entity_graph(corpus: pd.DataFrame) -> nx.Graph:
    entity_docs = corpus.groupby("focus_area")["doc_text"].apply(lambda s: " ".join(s)).reset_index()
    entities = entity_docs["focus_area"].tolist()

    vectorizer = TfidfVectorizer(
        max_features=config.GRAPH_TFIDF_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(entity_docs["doc_text"])  # rows are L2-normalizable via cosine below
    sim = (tfidf @ tfidf.T).toarray()  # cosine similarity since TF-IDF rows are L2-normalized by default

    graph = nx.Graph()
    for idx, entity in enumerate(entities):
        doc_ids = corpus.loc[corpus["focus_area"] == entity, "doc_id"].tolist()
        graph.add_node(entity, doc_ids=doc_ids)

    n = len(entities)
    for i in range(n):
        sims = sim[i].copy()
        sims[i] = -1.0  # exclude self
        top_idx = sims.argsort()[::-1][: config.GRAPH_TOPK_NEIGHBORS]
        for j in top_idx:
            weight = float(sim[i, j])
            if weight >= config.GRAPH_SIM_THRESHOLD:
                graph.add_edge(entities[i], entities[j], weight=weight)

    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    for community_id, members in enumerate(communities):
        for member in members:
            graph.nodes[member]["community"] = community_id

    return graph


def graph_stats(graph: nx.Graph) -> dict:
    num_communities = len({graph.nodes[n].get("community", -1) for n in graph.nodes})
    return {
        "num_entities": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_communities": num_communities,
        "avg_degree": (2 * graph.number_of_edges() / graph.number_of_nodes()) if graph.number_of_nodes() else 0.0,
    }


if __name__ == "__main__":
    from data_prep import load_clean_corpus

    corpus = load_clean_corpus()
    g = build_entity_graph(corpus)
    print(graph_stats(g))
