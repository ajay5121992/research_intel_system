"""
Node 1: Trend Detector.

Given the query and the external signals store, surface the most relevant
trending topics. Uses the library index's embedder to compute cosine
similarity between the query and each topic label (works identically
whether the embedder backend is sentence-transformers or the hashing
fallback -- no extra dependency needed here).

Special case: for GAP-intent queries, the question wording usually isn't
about any one specific topic ("which topics are under-covered?"), so
similarity-filtering against the query text doesn't make sense. In that
case we broaden to scan the FULL topic universe, ranked by momentum, so
gap analysis considers every topic rather than only the few most
textually similar to the question.
"""
import logging
from typing import List, Dict

import numpy as np

from src import config

logger = logging.getLogger(__name__)


def _rank_by_similarity(query: str, signals: List[Dict], embedder, top_n: int) -> List[Dict]:
    topic_labels = [s["topic"] for s in signals]
    query_vec = embedder.encode([query])[0]
    topic_vecs = embedder.encode(topic_labels)
    sims = topic_vecs @ query_vec  # vectors are already L2-normalized

    ranked = sorted(zip(signals, sims), key=lambda pair: pair[1], reverse=True)
    return [dict(sig, relevance_score=float(score)) for sig, score in ranked[:top_n]]


def _rank_by_momentum(signals: List[Dict], top_n: int = None) -> List[Dict]:
    ranked = sorted(signals, key=lambda s: s.get("momentum", 0.0), reverse=True)
    ranked = ranked[:top_n] if top_n else ranked
    return [dict(sig, relevance_score=sig.get("momentum", 0.0)) for sig in ranked]


def detect_trends(query: str, signals: List[Dict], embedder, intent: str = None, top_n: int = None) -> List[Dict]:
    top_n = top_n or config.TREND_TOP_N

    if intent == "GAP":
        # Broad scan: gap analysis should consider the full topic universe,
        # not just topics that are textually similar to the gap question.
        return _rank_by_momentum(signals)

    try:
        results = _rank_by_similarity(query, signals, embedder, top_n)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Similarity ranking failed (%s); falling back to momentum ranking", exc)
        results = _rank_by_momentum(signals, top_n)

    if not results:
        results = _rank_by_momentum(signals, top_n)
    return results


def make_node(external_signals: List[Dict], embedder):
    def node(state: dict) -> dict:
        trends = detect_trends(
            query=state["query"],
            signals=external_signals,
            embedder=embedder,
            intent=state.get("intent"),
        )
        trace_entry = {
            "node": "trend_detector",
            "output": f"{len(trends)} topic(s) selected "
            f"({'full momentum scan' if state.get('intent') == 'GAP' else 'query-similarity ranked'})",
        }
        return {
            "trends": trends,
            "trace": state.get("trace", []) + [trace_entry],
        }

    return node
