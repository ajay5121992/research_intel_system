
"""
Streamlit UI -- Layer 6 of the architecture.

Four tabs, matching the finalized (trimmed) scope exactly:
  1. Q&A with Citations
  2. Topic Insights
  3. Gap Identification
  4. Orchestration Trace

Run from the project root:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm_client
from src.agent_graph import run_query, run_gap_scan
from src.bootstrap import bootstrap

st.set_page_config(
    page_title="Research Intelligence System",
    page_icon="🧭",
    layout="wide",
)

_ABOUT_TEXT = """
This tool answers **"What topics are emerging in the market, and how well
does our library cover them?"** by combining two different kinds of data:

- 📡 **External Signals** — what the *market* is talking about right now
  (news volume + search interest, per topic). Can be live **Google Trends**
  data or a synthetic offline fallback.
- 📚 **Internal Library** — what *your organization* has already published
  or researched. Real articles from the **HuffPost News Category Dataset**,
  filtered to the topic taxonomy in `data/topics.py`.

A **Knowledge Graph** links topics, entities, and articles together, and an
**agent** decides — per question — whether to pull trend data, retrieve
library documents, check graph coverage, compute a gap score, or some
combination, then synthesizes a cited answer.
"""


def _data_sources_summary(library_df, external_signals):
    """Best-effort as-of dates + row counts for the sidebar 'About the
    data' panel. Never raises -- falls back to 'n/a' if fields are missing."""
    lib_dates = pd.to_datetime(library_df["date"], errors="coerce").dropna()
    lib_range = (
        f"{lib_dates.min().date()} → {lib_dates.max().date()}"
        if len(lib_dates) else "n/a"
    )
    signal_dates = [s.get("last_updated") for s in external_signals if s.get("last_updated")]
    signal_asof = max(signal_dates) if signal_dates else "n/a"
    return lib_range, signal_asof


@st.cache_resource(show_spinner="Bootstrapping index, knowledge graph, and agent...")
def get_system():
    return bootstrap()


def render_sidebar(ctx, library_df, external_signals):
    st.sidebar.title("🧭 System Status")

    health = llm_client.check_ollama_health()
    status = health["status"]
    if status == "online":
        st.sidebar.success(f"Ollama online ({health['model']})")
    elif status == "mock_forced":
        st.sidebar.info("FORCE_MOCK_LLM=true -- using deterministic mock LLM")
    elif status == "online_model_missing":
        st.sidebar.warning(
            f"Ollama online, but model '{health['model']}' not pulled. "
            f"Run: `ollama pull {health['model']}`. Falling back to mock."
        )
    else:
        st.sidebar.warning(
            "Ollama not reachable -- falling back to mock LLM.\n\n"
            "To use a real LLM: `ollama serve` then "
            f"`ollama pull {health['model']}`."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Embedder backend:** `{ctx.library_index.embedder.name}`")
    st.sidebar.markdown(f"**Library articles:** {len(library_df)}")
    st.sidebar.markdown(f"**FAISS chunks indexed:** {ctx.library_index.index.ntotal}")
    st.sidebar.markdown(f"**Topic signals:** {len(external_signals)}")
    st.sidebar.markdown(
        f"**Knowledge graph:** {ctx.graph.number_of_nodes()} nodes, "
        f"{ctx.graph.number_of_edges()} edges"
    )

    st.sidebar.markdown("---")
    lib_range, signal_asof = _data_sources_summary(library_df, external_signals)
    with st.sidebar.expander("ℹ️ About the data", expanded=False):
        st.markdown(
            f"**📚 Internal Library**\n"
            f"- What your org has already published/researched\n"
            f"- Source: real articles from the HuffPost News Category "
            f"Dataset, filtered to the 8 topics in `data/topics.py` "
            f"(`data/load_huffpost_by_category.py`)\n"
            f"- Article dates span: `{lib_range}`\n"
            f"- Refresh by re-running the loader script, then "
            f"'Rebuild index & graph' below\n"
            f"- No real data yet? `data/generate_sample_data.py` produces "
            f"a small synthetic fallback using the same topic labels"
        )
        st.markdown(
            f"**📡 External Signals**\n"
            f"- What the market/news/search is talking about right now\n"
            f"- Source: live Google Trends via `pytrends` "
            f"(`USE_LIVE_GOOGLE_TRENDS=true`), or a synthetic fallback "
            f"if offline\n"
            f"- Signals as of: `{signal_asof}` (end of the fetched "
            f"Trends window -- matches `GOOGLE_TRENDS_TIMEFRAME`)\n"
        )
        fallback_topics = [
            s["topic"] for s in external_signals
            if s.get("source") == "static_fallback_empty_live_result"
        ]
        if fallback_topics:
            st.warning(
                f"⚠️ {len(fallback_topics)} topic(s) had no live Google Trends "
                f"data and used a static fallback momentum instead: "
                f"{', '.join(fallback_topics)}"
            )
        st.caption(
            "Gap analysis compares real search-interest momentum against "
            "real library coverage per topic -- it's only as meaningful "
            "as how current/representative these two sources are."
        )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Rebuild index & graph"):
        st.cache_resource.clear()
        st.rerun()


def render_qa_tab(agent):
    st.subheader("Ask a question")
    st.caption(
        "Routes through: Intent Classifier → (Trend Detector) → (RAG Retriever) → "
        "(KG Query) → (Gap Analyser) → Synthesiser, depending on intent."
    )

    example_questions = [
        "What wellness topics have surged in search interest recently?",
        "What are the key stories in politics this quarter?",
        "Which topics are under-covered versus market interest?",
        "Summarize recent signals around business and the economy.",
    ]
    cols = st.columns(len(example_questions))
    for col, eq in zip(cols, example_questions):
        if col.button(eq, use_container_width=True):
            st.session_state["qa_query"] = eq

    query = st.text_input(
        "Your question",
        key="qa_query",
        placeholder="e.g. What topics are emerging in the market, and how well is our library covering them?",
    )

    if st.button("Ask", type="primary") and query.strip():
        with st.spinner("Running agent..."):
            result = run_query(agent, query)
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result:
        route_colors = {
            "TREND": "blue", "ENTITY_RAG_KG": "green", "GAP": "orange", "HYBRID": "violet",
        }
        color = route_colors.get(result["intent"], "gray")
        st.markdown(f":{color}[**Route: {result['intent']}**] (confidence {result.get('intent_confidence', 0):.2f})")

        st.markdown("### Answer")
        st.write(result["final_answer"])

        if result.get("citations"):
            st.markdown("### Citations")
            st.caption("📚 Sourced from your Internal Library (grounded — every citation below was actually retrieved).")
            for citation in result["citations"]:
                doc = next(
                    (d for d in result.get("retrieved_docs", []) if d["article_id"] == citation),
                    None,
                )
                if doc:
                    with st.expander(f"📄 {citation} — {doc['headline']}"):
                        st.write(doc["chunk_text"])
                        st.caption(f"Category: {doc['category']} · Date: {doc['date']} · Score: {doc['score']:.3f}")
                else:
                    st.write(f"📄 {citation}")
        else:
            st.caption("No grounded citations were returned for this query.")


def render_topic_insights_tab(external_signals):
    st.subheader("Topic Insights")
    st.caption(
        "📡 **External Signals** — what the market is talking about (news volume + "
        "search interest momentum, per topic). Not from your internal library."
    )
    signal_dates = [s.get("last_updated") for s in external_signals if s.get("last_updated")]
    if signal_dates:
        st.caption(f"Signals as of: `{max(signal_dates)}`")

    df = pd.DataFrame(external_signals)
    df_sorted = df.sort_values("momentum", ascending=False)

    st.markdown("##### Market momentum by topic")
    st.caption("Y-axis: momentum score (0-1, recent Google Trends search interest, averaged over the last few points of each topic's series). X-axis: topic.")
    st.bar_chart(df_sorted.set_index("topic")["momentum"])

    display_df = df_sorted[["topic", "category", "momentum"]].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("#### Weekly interest trend")
    selected_topic = st.selectbox("Select a topic", df_sorted["topic"].tolist())
    row = df[df["topic"] == selected_topic].iloc[0]

    raw_series = row["weekly_interest"]
    granularity = row.get("granularity", "unknown")

    # Support both the current [date, value] pair shape and, defensively,
    # a legacy bare-float list (e.g. an older cached signals file) so this
    # doesn't crash on stale cache/JSON from before this fix.
    if raw_series and isinstance(raw_series[0], (list, tuple)):
        dates = [pd.to_datetime(d) for d, _ in raw_series]
        values = [v for _, v in raw_series]
    else:
        dates = list(range(1, len(raw_series) + 1))
        values = raw_series
        granularity = "unknown (legacy data -- no dates available)"

    if not raw_series:
        st.info(f"No interest data available for **{selected_topic}** in this window.")
    else:
        label = {
            "daily": "Daily interest",
            "weekly": "Weekly interest",
        }.get(granularity, f"Interest ({granularity})")
        axis_start = dates[0]
        axis_end = dates[-1]
        st.caption(
            f"Y-axis: Google Trends search interest (0-1, normalized). "
            f"X-axis: date ({label.lower()} points, {axis_start} to {axis_end})."
        )
        trend_df = pd.DataFrame({"date": dates, "interest": values}).set_index("date")
        st.line_chart(trend_df)


def render_gap_tab(ctx):
    st.subheader("Gap Identification")
    st.caption(
        "Compares 📡 **external topic momentum** against 📚 **internal coverage** "
        "(knowledge-graph density + RAG retrieval density from your library) "
        "across the full topic universe."
    )

    if st.button("🔍 Run full gap scan", type="primary"):
        with st.spinner("Scanning all topics..."):
            gap_scores = run_gap_scan(ctx)
        st.session_state["gap_scores"] = gap_scores

    gap_scores = st.session_state.get("gap_scores")
    if gap_scores:
        gap_df = pd.DataFrame(gap_scores)
        st.bar_chart(gap_df.set_index("topic")["gap_score"])
        st.dataframe(
            gap_df[["topic", "momentum", "kg_coverage_count", "rag_relevant_chunks", "coverage_signal", "gap_score"]],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("What do these columns mean?"):
            st.markdown(
                "- **momentum** (📡 external) — market interest score for the topic\n"
                "- **kg_coverage_count** (📚 internal) — # articles linked to this "
                "topic in the knowledge graph\n"
                "- **rag_relevant_chunks** (📚 internal) — # library chunks the "
                "retriever found relevant to this topic\n"
                "- **coverage_signal** (📚 internal) — combined internal-coverage "
                "score (0–1) blending the two above\n"
                "- **gap_score** — `momentum − coverage_signal`. Higher = market "
                "interest is outpacing what your library covers."
            )
        st.caption("Higher gap_score = higher market momentum relative to internal coverage (bigger opportunity).")
    else:
        st.info("Click 'Run full gap scan' to compute coverage gaps across all topics.")


def render_trace_tab():
    st.subheader("Orchestration Trace")
    st.caption("Shows exactly which agent nodes ran, in order, for the last Q&A query.")

    result = st.session_state.get("last_result")
    if not result:
        st.info("Ask a question in the Q&A tab first to see its orchestration trace.")
        return

    st.markdown(f"**Query:** {result['query']}")
    st.markdown(f"**Route:** `{result['intent']}`")

    for i, step in enumerate(result["trace"], start=1):
        st.markdown(f"**{i}. `{step['node']}`**")
        st.code(str(step["output"]), language=None)

    with st.expander("Raw final state (debug)"):
        debug_state = {k: v for k, v in result.items() if k != "trace"}
        st.json(debug_state)


def main():
    st.title("Research Intelligence System")
    st.caption("Agentic AI + RAG + Knowledge Graph for Client Demand Sensing")

    with st.expander("ℹ️ What is this? (click to expand)", expanded=False):
        st.markdown(_ABOUT_TEXT)

    agent, ctx, library_df, external_signals = get_system()

    render_sidebar(ctx, library_df, external_signals)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Q&A with Citations", "📈 Topic Insights", "🔍 Gap Identification", "🧩 Orchestration Trace"]
    )
    with tab1:
        render_qa_tab(agent)
    with tab2:
        render_topic_insights_tab(external_signals)
    with tab3:
        render_gap_tab(ctx)
    with tab4:
        render_trace_tab()


if __name__ == "__main__":
    main()
