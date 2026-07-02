"""
Node 3: KG Query.

Given the trending topics (or the raw query for pure entity questions),
traverses the knowledge graph to compute structural coverage: how many
articles and entities are connected to each topic within KG_TRAVERSAL_DEPTH
hops. This "kg_density" measure feeds directly into gap_analyser.
"""
import logging
from typing import Dict, List

from src import config, knowledge_graph as kg

logger = logging.getLogger(__name__)


def query_topics(topics: List[str], graph) -> Dict[str, Dict]:
    coverage = {}
    for topic_label in topics:
        coverage[topic_label] = kg.query_topic_coverage(
            graph, topic_label, depth=config.KG_TRAVERSAL_DEPTH
        )
    return coverage


def make_node(graph):
    def node(state: dict) -> dict:
        if state.get("trends"):
            topic_labels = [t["topic"] for t in state["trends"]]
        else:
            # Entity-only route: no trends computed, so query the graph for
            # any topic whose label textually overlaps the raw query.
            query_lower = state["query"].lower()
            topic_labels = [
                data["label"]
                for node_id, data in graph.nodes(data=True)
                if data.get("type") == "Topic"
                and any(word in query_lower for word in data["label"].lower().split())
            ]

        kg_coverage = query_topics(topic_labels, graph)

        trace_entry = {
            "node": "kg_query",
            "output": f"traversed {len(topic_labels)} topic(s), depth={config.KG_TRAVERSAL_DEPTH}",
        }
        return {
            "kg_coverage": kg_coverage,
            "trace": state.get("trace", []) + [trace_entry],
        }

    return node
