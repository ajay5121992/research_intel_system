"""
Node 0: Intent Classifier -- first node, always runs.

Classifies the incoming query into one of four routes:
  TREND          -> trend_detector -> synthesiser
  ENTITY_RAG_KG  -> rag_retriever -> kg_query -> synthesiser
  GAP            -> trend_detector -> rag_retriever -> kg_query -> gap_analyser -> synthesiser
  HYBRID         -> trend_detector -> rag_retriever -> kg_query -> gap_analyser -> synthesiser

Uses the LLM for classification with a deterministic keyword-heuristic
mock fallback so classification still works with no LLM running.
"""
import json
import logging
import re

from src import llm_client

logger = logging.getLogger(__name__)

VALID_INTENTS = {"TREND", "ENTITY_RAG_KG", "GAP", "HYBRID"}

_SYSTEM_PROMPT = """You are an intent classifier for a research intelligence agent.
Classify the user's question into exactly one of: TREND, ENTITY_RAG_KG, GAP, HYBRID.

- TREND: asking what topics are emerging/trending/surging in the market.
- ENTITY_RAG_KG: asking a specific factual question about an entity, company, region, or event, best answered by retrieving library documents and graph context.
- GAP: asking to compare market interest/demand against internal library coverage, or asking what is under-covered.
- HYBRID: asking a question that requires both trend detection AND gap/coverage comparison AND specific retrieval.

Respond ONLY with JSON: {"intent": "<ONE_OF_ABOVE>", "confidence": <float 0-1>}"""

_GAP_KEYWORDS = ["under-covered", "under covered", "coverage gap", "gap", "versus our", "vs. our", "vs our"]
_HYBRID_KEYWORDS = ["and how well", "how well is our", "compare", "coverage vs"]
_TREND_KEYWORDS = ["emerging", "trending", "surged", "surge", "momentum", "what's new", "whats new", "recent trends"]
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_MULTIWORD_CAP_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b")


def _keyword_fallback(query: str) -> dict:
    q = query.lower()

    if any(kw in q for kw in _GAP_KEYWORDS):
        return {"intent": "GAP", "confidence": 0.7}

    if any(kw in q for kw in _HYBRID_KEYWORDS):
        return {"intent": "HYBRID", "confidence": 0.6}

    if any(kw in q for kw in _TREND_KEYWORDS):
        return {"intent": "TREND", "confidence": 0.65}

    # Entity-mention heuristics: multi-word capitalized phrase OR an
    # ALL-CAPS acronym (e.g. "APAC") both suggest a specific-entity question.
    if _MULTIWORD_CAP_RE.search(query) or _ACRONYM_RE.search(query):
        return {"intent": "ENTITY_RAG_KG", "confidence": 0.65}

    return {"intent": "HYBRID", "confidence": 0.5}


def _mock_fn(query: str) -> str:
    return json.dumps(_keyword_fallback(query))


def classify_intent(query: str) -> dict:
    from src.tools.synthesiser import _extract_json

    raw = llm_client.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f"Question: {query}",
        mock_fn=lambda: _mock_fn(query),
        json_mode=True,
    )
    try:
        parsed = _extract_json(raw)
        intent = str(parsed.get("intent", "")).upper()
        if intent not in VALID_INTENTS:
            raise ValueError(f"Invalid intent from LLM: {intent}")
        return {"intent": intent, "confidence": float(parsed.get("confidence", 0.5))}
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Failed to parse LLM intent output (%s); using keyword fallback", exc)
        return _keyword_fallback(query)


def make_node():
    """Factory returning a LangGraph node function: state -> partial state update."""

    def node(state: dict) -> dict:
        result = classify_intent(state["query"])
        trace_entry = {
            "node": "intent_classifier",
            "output": result,
        }
        return {
            "intent": result["intent"],
            "intent_confidence": result["confidence"],
            "trace": state.get("trace", []) + [trace_entry],
        }

    return node