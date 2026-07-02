"""
One-time bootstrap: loads data, builds (or loads a cached) FAISS index and
knowledge graph, and constructs the compiled agent.

Used by both the Streamlit app (via @st.cache_resource) and the smoke
test / pytest suite, so there is exactly one code path for "get me a ready
system" that everything else depends on.
"""
import logging

from src import config, data_loader, indexer, knowledge_graph as kg
from src.agent_graph import AgentContext, build_agent

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def bootstrap(force_rebuild: bool = False):
    """Returns (agent, AgentContext, library_df, external_signals)."""
    library_df = data_loader.load_internal_library()

    if config.USE_LIVE_GOOGLE_TRENDS:
        logger.info("USE_LIVE_GOOGLE_TRENDS=true -> fetching live Google Trends data")
        try:
            external_signals = data_loader.build_live_external_signals()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Live Google Trends fetch failed (%s); falling back to synthetic "
                "external_signals.json", exc,
            )
            external_signals = data_loader.load_external_signals()
    else:
        external_signals = data_loader.load_external_signals()

    index_exists = config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()
    if index_exists and not force_rebuild:
        try:
            library_index = indexer.load_index()
            logger.info("Loaded cached FAISS index")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load cached index (%s); rebuilding", exc)
            library_index = indexer.build_index(library_df)
            library_index.save()
    else:
        library_index = indexer.build_index(library_df)
        library_index.save()

    graph_exists = config.GRAPH_PATH.exists()
    if graph_exists and not force_rebuild:
        try:
            graph = kg.load_graph()
            logger.info("Loaded cached knowledge graph")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load cached graph (%s); rebuilding", exc)
            graph = kg.build_knowledge_graph(library_df, external_signals)
            kg.save_graph(graph)
    else:
        graph = kg.build_knowledge_graph(library_df, external_signals)
        kg.save_graph(graph)

    ctx = AgentContext(library_index=library_index, graph=graph, external_signals=external_signals)
    agent = build_agent(ctx)

    return agent, ctx, library_df, external_signals


if __name__ == "__main__":
    agent, ctx, library_df, external_signals = bootstrap()
    print(f"Bootstrap complete.")
    print(f"  Library articles: {len(library_df)}")
    print(f"  External signal topics: {len(external_signals)}")
    print(f"  FAISS index size: {ctx.library_index.index.ntotal} chunks (embedder={ctx.library_index.embedder.name})")
    print(f"  Knowledge graph: {ctx.graph.number_of_nodes()} nodes, {ctx.graph.number_of_edges()} edges")
