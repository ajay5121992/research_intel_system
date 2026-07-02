"""
Layer 3: Knowledge Graph.

Builds a NetworkX DiGraph with three node types:
  - Category  (root taxonomy nodes, e.g. HEALTH_TECH, POLICY)
  - Topic     (from external_signals, e.g. "AI in Healthcare")
  - Entity    (organizations/institutions mentioned per topic, plus NER hits
               pulled from library article text)
  - Article   (each internal library article)

Edges:
  - sub_topic_of : Topic -> Category
  - related_to   : Entity -> Topic
  - covered_by   : Topic -> Article   (article covers this topic)
  - mentions     : Article -> Entity  (article text mentions this entity)

Entity extraction from article text uses spaCy NER if a model is available,
else falls back to a regex heuristic (capitalized multi-word sequences).
This mirrors the same graceful-fallback pattern used in the embedder.
"""
import logging
import pickle
import re
from typing import List, Dict, Set

import networkx as nx

from src import config

logger = logging.getLogger(__name__)

_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b")


# ---------------------------------------------------------------------------
# NER abstraction
# ---------------------------------------------------------------------------
class EntityExtractor:
    name = "base"

    def extract(self, text: str) -> Set[str]:
        raise NotImplementedError


class SpacyEntityExtractor(EntityExtractor):
    name = "spacy"

    def __init__(self, model_name: str = None):
        import spacy  # may raise

        model_name = model_name or config.SPACY_MODEL
        self._nlp = spacy.load(model_name)  # raises OSError if model absent

    def extract(self, text: str) -> Set[str]:
        doc = self._nlp(text)
        return {
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in ("ORG", "GPE", "PERSON", "PRODUCT", "NORP")
            and len(ent.text.strip()) > 2
        }


class RegexEntityExtractor(EntityExtractor):
    """Fallback NER: capitalized multi-word sequences as pseudo-entities.

    Not linguistically precise, but requires no model download and never
    fails -- keeps the knowledge graph buildable fully offline.
    """

    name = "regex-fallback"

    def extract(self, text: str) -> Set[str]:
        return {m.strip() for m in _CAP_PHRASE_RE.findall(text)}


def get_entity_extractor() -> EntityExtractor:
    if config.FORCE_REGEX_NER:
        logger.info("FORCE_REGEX_NER=true -> using RegexEntityExtractor")
        return RegexEntityExtractor()
    try:
        extractor = SpacyEntityExtractor()
        logger.info("Using SpacyEntityExtractor (%s)", config.SPACY_MODEL)
        return extractor
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "spaCy model unavailable (%s); falling back to RegexEntityExtractor",
            exc,
        )
        return RegexEntityExtractor()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_knowledge_graph(
    library_df, external_signals: List[Dict], extractor: EntityExtractor = None
) -> nx.DiGraph:
    extractor = extractor or get_entity_extractor()
    g = nx.DiGraph()

    # 1. Category nodes
    categories = set(library_df["category"].unique()) | {
        s["category"] for s in external_signals
    }
    for cat in categories:
        g.add_node(f"CATEGORY::{cat}", type="Category", label=cat)

    # 2. Topic nodes + sub_topic_of Category, + Entity nodes from signal defs
    for sig in external_signals:
        topic_id = f"TOPIC::{sig['topic']}"
        g.add_node(
            topic_id,
            type="Topic",
            label=sig["topic"],
            momentum=sig.get("momentum", 0.0),
            category=sig.get("category"),
        )
        cat_id = f"CATEGORY::{sig['category']}"
        if g.has_node(cat_id):
            g.add_edge(topic_id, cat_id, relation="sub_topic_of")

        for entity in sig.get("entities", []):
            entity_id = f"ENTITY::{entity}"
            g.add_node(entity_id, type="Entity", label=entity)
            g.add_edge(entity_id, topic_id, relation="related_to")

    topic_labels = {sig["topic"]: f"TOPIC::{sig['topic']}" for sig in external_signals}

    # 3. Article nodes + covered_by (Topic -> Article) + mentions (Article -> Entity)
    for _, row in library_df.iterrows():
        article_id = f"ARTICLE::{row['article_id']}"
        g.add_node(
            article_id,
            type="Article",
            label=row["headline"],
            category=row["category"],
            date=row["date"],
        )

        seed_topic = row.get("seed_topic")
        topic_node = topic_labels.get(seed_topic)
        if topic_node and g.has_node(topic_node):
            g.add_edge(topic_node, article_id, relation="covered_by")
        else:
            # Fallback: no matching topic signal for this article's seed_topic
            # (e.g. real-world data lacking a clean seed_topic tag). Skip
            # linking rather than guessing incorrectly.
            logger.debug(
                "Article %s has no matching topic signal for seed_topic=%s",
                row["article_id"], seed_topic,
            )

        entities = extractor.extract(row["full_text"])
        for entity in entities:
            entity_id = f"ENTITY::{entity}"
            if not g.has_node(entity_id):
                g.add_node(entity_id, type="Entity", label=entity)
            g.add_edge(article_id, entity_id, relation="mentions")

    logger.info(
        "Built knowledge graph: %d nodes, %d edges (extractor=%s)",
        g.number_of_nodes(), g.number_of_edges(), extractor.name,
    )
    return g


# ---------------------------------------------------------------------------
# KG query: given a topic label, traverse and collect coverage
# ---------------------------------------------------------------------------
def query_topic_coverage(g: nx.DiGraph, topic_label: str, depth: int = None) -> Dict:
    """Traverse from a Topic node to find the articles and entities that
    genuinely belong to it.

    IMPORTANT: this deliberately does NOT do a plain undirected BFS out to
    `depth` hops from the topic node. Because Entity nodes are shared
    across topics (e.g. "United States" gets mentioned in Politics,
    Business, and Travel articles alike), an undirected BFS at depth=2
    walks Topic -> Article -> Entity -> (every other article that
    mentions that entity, regardless of topic) -- which floods coverage
    to nearly the entire library for every topic. In practice this made
    kg_coverage_count identical (~= total_articles / num_topics) across
    ALL topics, so it carried zero signal for gap analysis.

    Instead:
      - "articles" = only articles directly linked to this topic via a
        `covered_by` edge (Topic -> Article). This is the actual,
        unambiguous definition of "this article is about this topic".
      - "entities" = entities reachable from those articles/the topic
        within `depth` hops, which is still useful context for the KG
        query tool/UI, but is kept separate from the coverage COUNT so it
        can't dilute the gap score.
    """
    depth = depth or config.KG_TRAVERSAL_DEPTH
    topic_id = f"TOPIC::{topic_label}"
    if not g.has_node(topic_id):
        return {
            "topic": topic_label,
            "found": False,
            "articles": [],
            "entities": [],
            "coverage_count": 0,
        }

    # Direct coverage: only articles with an explicit covered_by edge from
    # this topic. This is the signal that should drive gap_score.
    articles = [
        {
            "article_id": article_id.replace("ARTICLE::", ""),
            "headline": g.nodes[article_id].get("label"),
        }
        for article_id in g.successors(topic_id)
        if g.nodes[article_id].get("type") == "Article"
    ]

    # Entities: still traverse outward for context (used by KG Q&A / UI),
    # but scoped to the topic's own subgraph (topic + its direct articles
    # + their directly-mentioned entities) rather than a global undirected
    # flood, and capped at `depth` hops from the topic within that subgraph.
    scoped_nodes = {topic_id} | {f"ARTICLE::{a['article_id']}" for a in articles}
    for article in list(scoped_nodes):
        if g.has_node(article):
            scoped_nodes.update(g.successors(article))  # Article -> Entity (mentions)

    subgraph = g.subgraph(scoped_nodes).to_undirected(as_view=True)
    if topic_id in subgraph:
        visited = nx.single_source_shortest_path_length(subgraph, topic_id, cutoff=depth)
    else:
        visited = {}

    entities = [
        g.nodes[node_id].get("label")
        for node_id in visited
        if g.nodes[node_id].get("type") == "Entity"
    ]

    return {
        "topic": topic_label,
        "found": True,
        "articles": articles,
        "entities": sorted(set(entities)),
        "coverage_count": len(articles),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_graph(g: nx.DiGraph, path=None):
    path = path or config.GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(g, f)
    logger.info("Saved knowledge graph -> %s", path)


def load_graph(path=None) -> nx.DiGraph:
    path = path or config.GRAPH_PATH
    with open(path, "rb") as f:
        return pickle.load(f)
