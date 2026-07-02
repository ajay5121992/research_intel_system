"""
Agent orchestration graph.

Wires the five/six agent nodes into the intent-routed state machine defined
in the architecture:

  TREND:          intent_classifier -> trend_detector -> rag_retriever -> kg_query -> synthesiser
  ENTITY_RAG_KG:  intent_classifier -> rag_retriever -> kg_query -> synthesiser
  GAP:            intent_classifier -> trend_detector -> rag_retriever -> kg_query -> gap_analyser -> synthesiser
  HYBRID:         intent_classifier -> trend_detector -> rag_retriever -> kg_query -> gap_analyser -> synthesiser

  Note: TREND queries route through rag_retriever/kg_query too (not just
  trend_detector). Momentum scores alone tell you a topic is trending, not
  *why* -- that "why" has to come from grounded library evidence, so every
  route converges on the synthesiser with retrieved_docs populated.

Built with LangGraph when available. If the `langgraph` package is not
installed, falls back to a tiny dependency-free state-machine runner that
executes the exact same routing logic -- so the agent still works even in
a minimal environment. This mirrors the graceful-fallback pattern used
throughout the rest of the system (embedder, NER, LLM).
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, TypedDict, Optional

import networkx as nx

from src import config
from src.indexer import LibraryIndex
from src.tools import (
    intent_classifier,
    trend_detector,
    rag_retriever,
    kg_query,
    gap_analyser,
    synthesiser,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    intent_confidence: float
    trends: List[Dict]
    retrieved_docs: List[Dict]
    retrieved_docs_per_topic: Dict
    kg_coverage: Dict
    gap_scores: List[Dict]
    final_answer: str
    citations: List[str]
    trace: List[Dict]


@dataclass
class AgentContext:
    library_index: LibraryIndex
    graph: nx.DiGraph
    external_signals: List[Dict]


# ---------------------------------------------------------------------------
# Routing functions (shared by both the LangGraph build and the fallback
# runner, so behavior is identical either way)
# ---------------------------------------------------------------------------
def route_after_intent(state: AgentState) -> str:
    return "rag_retriever" if state["intent"] == "ENTITY_RAG_KG" else "trend_detector"


def route_after_trend(state: AgentState) -> str:
    # Previously TREND queries skipped straight to the synthesiser, so the
    # answer was only ever a formatted list of external_signals.json
    # momentum scores with no grounded evidence ("No grounded citations for
    # this route (expected for pure TREND queries)" in the UI). Trends alone
    # say a topic is hot, not *why* -- that requires retrieval. Every route
    # now continues on to rag_retriever so the synthesiser always has
    # library evidence to cite, whichever intent produced it.
    return "rag_retriever"


def route_after_rag(state: AgentState) -> str:
    # All routes that reach rag_retriever continue on to kg_query.
    return "kg_query"


def route_after_kg(state: AgentState) -> str:
    return "gap_analyser" if state["intent"] in ("GAP", "HYBRID") else "synthesiser"


def route_after_gap(state: AgentState) -> str:
    return "synthesiser"


# ---------------------------------------------------------------------------
# Build the compiled agent
# ---------------------------------------------------------------------------
def _build_nodes(ctx: AgentContext) -> Dict:
    return {
        "intent_classifier": intent_classifier.make_node(),
        "trend_detector": trend_detector.make_node(ctx.external_signals, ctx.library_index.embedder),
        "rag_retriever": rag_retriever.make_node(ctx.library_index),
        "kg_query": kg_query.make_node(ctx.graph),
        "gap_analyser": gap_analyser.make_node(),
        "synthesiser": synthesiser.make_node(),
    }


def build_agent(ctx: AgentContext):
    """Returns a compiled LangGraph app if langgraph is installed, otherwise
    a FallbackAgent with an identical `.invoke(state) -> state` interface."""
    nodes = _build_nodes(ctx)

    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)
        for name, fn in nodes.items():
            graph.add_node(name, fn)

        graph.set_entry_point("intent_classifier")
        graph.add_conditional_edges(
            "intent_classifier", route_after_intent,
            {"trend_detector": "trend_detector", "rag_retriever": "rag_retriever"},
        )
        graph.add_conditional_edges(
            "trend_detector", route_after_trend,
            {"synthesiser": "synthesiser", "rag_retriever": "rag_retriever"},
        )
        graph.add_conditional_edges(
            "rag_retriever", route_after_rag, {"kg_query": "kg_query"},
        )
        graph.add_conditional_edges(
            "kg_query", route_after_kg,
            {"gap_analyser": "gap_analyser", "synthesiser": "synthesiser"},
        )
        graph.add_conditional_edges(
            "gap_analyser", route_after_gap, {"synthesiser": "synthesiser"},
        )
        graph.add_edge("synthesiser", END)

        logger.info("Built agent using LangGraph")
        return graph.compile()

    except ImportError as exc:
        logger.warning(
            "langgraph not installed (%s); using dependency-free FallbackAgent runner",
            exc,
        )
        return FallbackAgent(nodes)


class FallbackAgent:
    """Executes the exact same routing table as the LangGraph build, without
    requiring the langgraph package. Provides a matching `.invoke()` API."""

    def __init__(self, nodes: Dict):
        self._nodes = nodes

    def invoke(self, state: AgentState) -> AgentState:
        state = dict(state)
        state.setdefault("trace", [])

        state.update(self._nodes["intent_classifier"](state))
        next_node = route_after_intent(state)

        if next_node == "trend_detector":
            state.update(self._nodes["trend_detector"](state))
            next_node = route_after_trend(state)

        if next_node == "rag_retriever":
            state.update(self._nodes["rag_retriever"](state))
            next_node = route_after_rag(state)

        if next_node == "kg_query":
            state.update(self._nodes["kg_query"](state))
            next_node = route_after_kg(state)

        if next_node == "gap_analyser":
            state.update(self._nodes["gap_analyser"](state))

        state.update(self._nodes["synthesiser"](state))
        return state


def run_query(agent, query: str) -> AgentState:
    initial_state: AgentState = {"query": query, "trace": []}
    return agent.invoke(initial_state)


def run_gap_scan(ctx: AgentContext) -> List[Dict]:
    """Standalone helper for the dedicated Gap Identification tab: runs the
    GAP route directly across the FULL topic universe, bypassing the intent
    classifier. Used by app.py so users don't have to phrase a question to
    see a full gap report."""
    nodes = _build_nodes(ctx)
    state: AgentState = {"query": "coverage gap scan", "intent": "GAP", "trace": []}

    state.update(nodes["trend_detector"](state))
    state.update(nodes["rag_retriever"](state))
    state.update(nodes["kg_query"](state))
    state.update(nodes["gap_analyser"](state))

    return state["gap_scores"]
