"""
Node 2: RAG Retriever.

Searches the FAISS library index for chunks relevant to the query, and
(when trends are already in state) also pulls chunks relevant to each
trending topic label -- so downstream gap analysis has retrieval coverage
per topic, not just per raw query text.
"""
import logging
from typing import Dict, List

from src import config

logger = logging.getLogger(__name__)


def retrieve(query: str, library_index, top_k: int = None) -> List[Dict]:
    top_k = top_k or config.RAG_TOP_K
    results = library_index.search(query, k=top_k)
    return [
        {
            "article_id": meta["article_id"],
            "headline": meta["headline"],
            "chunk_text": meta["chunk_text"],
            "category": meta["category"],
            "seed_topic": meta["seed_topic"],
            "date": meta["date"],
            "score": score,
        }
        for meta, score in results
    ]


def retrieve_per_topic(trends: List[Dict], library_index, top_k: int = 5) -> Dict[str, List[Dict]]:
    per_topic = {}
    for trend in trends:
        topic_label = trend["topic"]
        per_topic[topic_label] = retrieve(topic_label, library_index, top_k=top_k)
    return per_topic


def make_node(library_index):
    def node(state: dict) -> dict:
        docs = retrieve(state["query"], library_index)

        per_topic_docs = {}
        if state.get("trends"):
            per_topic_docs = retrieve_per_topic(state["trends"], library_index)

        trace_entry = {
            "node": "rag_retriever",
            "output": f"{len(docs)} chunk(s) for query"
            + (f", per-topic retrieval for {len(per_topic_docs)} topic(s)" if per_topic_docs else ""),
        }
        return {
            "retrieved_docs": docs,
            "retrieved_docs_per_topic": per_topic_docs,
            "trace": state.get("trace", []) + [trace_entry],
        }

    return node
