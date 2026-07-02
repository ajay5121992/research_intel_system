"""
Node 5: Synthesiser -- final node, all routes converge here.

Builds a prompt from the full state (trends / retrieved_docs / gap_scores,
whichever are populated for the given route) and asks the LLM to produce a
JSON answer with citations.

Guardrail: every citation is validated against the set of article_ids that
were actually retrieved in `retrieved_docs`. Any hallucinated citation is
silently dropped -- the answer text itself is never blocked, only
ungrounded citations are filtered out.
"""
import json
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a research intelligence assistant. Answer the user's question
using ONLY the provided context (trends, retrieved documents, gap scores, knowledge graph
coverage). Be concise and specific. Cite article IDs for any factual claim drawn from a
retrieved document.

Respond ONLY with a single JSON object with exactly two keys, "answer" and "citations".
Do not include any text before or after the JSON. Do not use placeholder text -- "answer"
must contain your real, complete answer to the question, and "citations" must be a list of
real article IDs from the retrieved documents (or an empty list if none apply).

Example of the exact shape required (do not reuse this example's content):
{"answer": "AI in Healthcare and Generative AI Regulation show the strongest momentum this quarter, both around 0.9.", "citations": ["ART00012", "ART00047"]}"""


def _build_context_block(state: Dict) -> str:
    parts = [f"Route: {state.get('intent')}", f"Question: {state['query']}"]

    if state.get("trends"):
        trend_lines = [
            f"- {t['topic']} (momentum={t.get('momentum', t.get('relevance_score', 0)):.2f})"
            for t in state["trends"]
        ]
        parts.append("Trending topics:\n" + "\n".join(trend_lines))

    if state.get("retrieved_docs"):
        doc_lines = [
            f"- [{d['article_id']}] {d['headline']}: {d['chunk_text'][:200]}"
            for d in state["retrieved_docs"]
        ]
        parts.append("Retrieved documents:\n" + "\n".join(doc_lines))

    if state.get("kg_coverage"):
        kg_lines = [
            f"- {topic}: {info.get('coverage_count', 0)} linked article(s), "
            f"entities: {', '.join(info.get('entities', [])[:5])}"
            for topic, info in state["kg_coverage"].items()
        ]
        parts.append("Knowledge graph coverage:\n" + "\n".join(kg_lines))

    if state.get("gap_scores"):
        gap_lines = [
            f"- {g['topic']}: gap_score={g['gap_score']:.2f} "
            f"(momentum={g['momentum']:.2f}, coverage_signal={g['coverage_signal']:.2f})"
            for g in state["gap_scores"]
        ]
        parts.append("Gap analysis (higher = more under-covered vs. market interest):\n" + "\n".join(gap_lines))

    return "\n\n".join(parts)


def _valid_article_ids(state: Dict) -> set:
    ids = {d["article_id"] for d in state.get("retrieved_docs", [])}
    for docs in state.get("retrieved_docs_per_topic", {}).values():
        ids |= {d["article_id"] for d in docs}
    return ids


def _mock_synthesis(state: Dict) -> str:
    """Deterministic templated answer built directly from state, used when
    no LLM is available. Produces a genuinely useful (if less fluent)
    answer rather than a placeholder."""
    intent = state.get("intent")
    lines = []
    citations = []

    if intent == "TREND" or state.get("trends"):
        top = state.get("trends", [])[:3]
        if top:
            topic_desc = "; ".join(
                f"{t['topic']} (momentum {t.get('momentum', t.get('relevance_score', 0)):.2f})"
                for t in top
            )
            lines.append(f"Top trending topics: {topic_desc}.")

    if state.get("retrieved_docs"):
        lines.append(
            f"Found {len(state['retrieved_docs'])} relevant library document(s)."
        )
        for d in state["retrieved_docs"][:3]:
            lines.append(f"- {d['headline']} ({d['article_id']})")
            citations.append(d["article_id"])

    if state.get("gap_scores"):
        gap_scores = state["gap_scores"]
        # Prefer the gap entry for the single MOST query-relevant trending
        # topic (trends[0], since trend_detector ranks by query similarity)
        # rather than the globally largest gap across the whole topic
        # universe -- otherwise a scoped question like "...supply-chain
        # resilience" could surface an unrelated topic's gap score just
        # because it happened to also be in the broad top-5 similarity set.
        trends = state.get("trends", [])
        gap_by_topic = {g["topic"]: g for g in gap_scores}
        top_gap = None
        if trends:
            top_gap = gap_by_topic.get(trends[0]["topic"])
        top_gap = top_gap or gap_scores[0]
        lines.append(
            f"Largest coverage gap: '{top_gap['topic']}' "
            f"(momentum {top_gap['momentum']:.2f} vs. coverage signal {top_gap['coverage_signal']:.2f}, "
            f"gap score {top_gap['gap_score']:.2f})."
        )

    if not lines:
        lines.append("No strong signal found in trends, retrieval, or gap analysis for this query.")

    return json.dumps({"answer": " ".join(lines), "citations": citations})


def _extract_json(raw: str) -> dict:
    """Robustly pull a {"answer": ..., "citations": [...]} object out of raw
    LLM output. Reasoning models (e.g. deepseek-r1) often wrap output in
    <think>...</think> blocks or markdown ```json fences before the actual
    JSON -- strip those before parsing."""
    if not isinstance(raw, str):
        raise ValueError("raw output is not a string")

    text = raw.strip()
    # Strip <think>...</think> reasoning blocks (deepseek-r1 and similar).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```).
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: grab the first {...} block in the text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise json.JSONDecodeError("No JSON object found", text, 0)


_PLACEHOLDER_ANSWERS = {"<answer text>", "answer text", "your answer here", ""}


def synthesise(state: Dict) -> Dict:
    from src import llm_client

    context_block = _build_context_block(state)
    valid_ids = _valid_article_ids(state)

    raw = llm_client.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=context_block,
        mock_fn=lambda: _mock_synthesis(state),
        json_mode=True,
    )

    answer = ""
    raw_citations: List[str] = []
    parse_ok = False
    try:
        parsed = _extract_json(raw)
        answer = str(parsed.get("answer", "")).strip()
        raw_citations = parsed.get("citations", [])
        if answer.lower() not in _PLACEHOLDER_ANSWERS:
            parse_ok = True
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    if not parse_ok:
        logger.warning(
            "LLM synthesiser output was empty, placeholder, or unparseable; "
            "falling back to deterministic mock synthesis"
        )
        mock_parsed = json.loads(_mock_synthesis(state))
        answer = mock_parsed["answer"]
        raw_citations = mock_parsed["citations"]

    # Guardrail: drop any citation not present in retrieved_docs.
    grounded_citations = [c for c in raw_citations if c in valid_ids]
    dropped = set(raw_citations) - set(grounded_citations)
    if dropped:
        logger.warning("Dropped ungrounded citation(s): %s", dropped)

    if not answer:
        answer = "I could not generate an answer for this query."

    return {"answer": answer, "citations": grounded_citations}


def make_node():
    def node(state: dict) -> dict:
        result = synthesise(state)
        trace_entry = {
            "node": "synthesiser",
            "output": f"answer generated, {len(result['citations'])} grounded citation(s)",
        }
        return {
            "final_answer": result["answer"],
            "citations": result["citations"],
            "trace": state.get("trace", []) + [trace_entry],
        }

    return node