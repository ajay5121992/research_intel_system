"""
Node 4: Gap Analyser.

Combines trend momentum + KG structural coverage + RAG retrieval density
into a single gap score per topic:

    coverage_signal = 0.5 * kg_density_norm + 0.5 * rag_ratio
    gap_score = momentum - coverage_signal   (higher = bigger gap)

Where:
  - kg_density_norm = min(kg_coverage_count / kg_target, 1.0), and
    kg_target = mean(kg_coverage_count across topics) * KG_COVERAGE_TARGET_MULTIPLIER.
    Normalized against an ABSOLUTE target (a multiple of the average
    topic's coverage), not the current run's own max -- otherwise
    whichever topic happens to be biggest is guaranteed 1.0 by
    construction, regardless of whether it's actually "fully covered".
  - rag_ratio = (# retrieved chunks for that topic scoring above a
                 relevance floor) / RAG_FULL_COVERAGE_CHUNKS.
    Scored against an absolute "what would comprehensive coverage look
    like" target, not RAG_TOP_K (the retriever's own fetch limit) --
    otherwise fetching exactly RAG_TOP_K chunks and having all of them
    pass trivially reads as maximum possible coverage.

Both normalizations are capped at 1.0 but, unlike the old self-relative
versions, are NOT guaranteed to be hit by any particular topic -- an
average or below-average topic will score meaningfully below 1.0, giving
coverage_signal (and therefore gap_score) real headroom to distinguish
genuinely over-covered topics from under-covered ones.

Topics are ranked descending by gap_score -- the biggest gaps (high market
momentum, low internal coverage) surface first.
"""
import logging
from typing import Dict, List

from src import config

logger = logging.getLogger(__name__)

# Minimum FAISS cosine similarity to count a retrieved chunk as "relevant"
# for a topic, used as a floor on top of the adaptive threshold below. See
# config.RAG_RELEVANCE_FLOOR docstring -- this used to be a hardcoded 0.05,
# trivially cleared by nearly any top-k result, which made
# rag_relevant_chunks == RAG_TOP_K for every topic and collapsed
# coverage_signal to a constant across all topics.
_ABSOLUTE_RELEVANCE_FLOOR = config.RAG_RELEVANCE_FLOOR

# A fixed absolute cosine cutoff behaves very differently depending on
# which embedder backend is active: sentence-transformers scores spread
# widely (often 0.1-0.7+), while the offline hashing-SVD fallback
# produces scores tightly clustered around ~0.4-0.45 regardless of true
# relevance. A single hardcoded floor either passes everything (fallback
# embedder) or is too strict (real embedder). To be robust to whichever
# backend loaded, ALSO require each chunk to clear a per-query adaptive
# threshold calibrated to that topic's own retrieved-score distribution,
# rather than one global number.
#
# Two earlier versions of this got the direction wrong in opposite ways:
#   - mean + k*stddev is >= the mean by construction, so it structurally
#     passes at most ~half of any topic's own scores no matter how good
#     they are -- a topic with five excellent, tightly-clustered chunks
#     still failed most of them against its own average, capping
#     rag_ratio near 0.5 for every topic regardless of true coverage.
#   - max_score - k*stddev fixes the direction but overcorrects for a
#     tight cluster: five nearly-identical strong scores have a tiny
#     absolute stddev, so the threshold sits just under the max and
#     rejects everything except the single top score -- exactly the
#     "genuinely well-covered topic scores near-zero rag_ratio" failure
#     this was meant to prevent, just via a different mechanism.
#
# What actually distinguishes "one strong hit + irrelevant tail" from
# "several genuinely similar strong matches" is RELATIVE dispersion
# (stddev relative to the mean), not absolute distance from the max.
# A real outlier tail has stddev comparable to or larger than the mean
# score itself; a tight cluster of similar-quality matches has stddev
# that's a small fraction of the mean. So: only tighten the cutoff when
# the coefficient of variation (stddev / mean) exceeds a threshold that
# signals real separation between "clearly relevant" and "clearly not" --
# otherwise treat the whole retrieved set as comparably relevant and let
# the absolute floor alone decide.
_CV_OUTLIER_THRESHOLD = 0.3  # stddev/mean above this signals a real split
_STDDEV_MULT = 0.75  # how far below the mean an outlier tail must sit


def _adaptive_threshold(scores: List[float]) -> float:
    if len(scores) < 2:
        return _ABSOLUTE_RELEVANCE_FLOOR
    mean = sum(scores) / len(scores)
    if mean <= 0:
        return _ABSOLUTE_RELEVANCE_FLOOR
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    stddev = variance ** 0.5
    coeff_of_variation = stddev / mean

    if coeff_of_variation < _CV_OUTLIER_THRESHOLD:
        # Scores are comparably relevant to each other (e.g. a topic with
        # several genuinely strong, similar matches) -- don't manufacture
        # a split among them. Fall back to the absolute floor only.
        return _ABSOLUTE_RELEVANCE_FLOOR

    # Real separation exists (e.g. one strong hit + a weak/irrelevant
    # tail) -- cut below the mean, not above it, so a topic's better-
    # than-average chunks pass and its worse-than-average ones don't.
    return max(_ABSOLUTE_RELEVANCE_FLOOR, mean - _STDDEV_MULT * stddev)


def analyse_gaps(
    trends: List[Dict],
    kg_coverage: Dict[str, Dict],
    retrieved_docs_per_topic: Dict[str, List[Dict]],
) -> List[Dict]:
    if not trends:
        return []

    # Absolute baseline instead of "this topic set's own max": the mean
    # article count across topics, scaled by KG_COVERAGE_TARGET_MULTIPLIER.
    # Using the current run's own max meant whichever topic happened to be
    # biggest was guaranteed kg_density_norm=1.0 by construction -- this
    # baseline instead asks "does this topic have at least
    # <multiplier>x the average topic's coverage", which a below-average
    # or even an average topic will NOT trivially satisfy.
    kg_counts = [
        kg_coverage.get(t["topic"], {}).get("coverage_count", 0) for t in trends
    ]
    mean_kg_count = (sum(kg_counts) / len(kg_counts)) if kg_counts else 0
    kg_target = max(mean_kg_count * config.KG_COVERAGE_TARGET_MULTIPLIER, 1)

    results = []
    for trend in trends:
        topic = trend["topic"]
        momentum = trend.get("momentum", trend.get("relevance_score", 0.0))

        kg_count = kg_coverage.get(topic, {}).get("coverage_count", 0)
        kg_density_norm = min(kg_count / kg_target, 1.0)

        topic_docs = retrieved_docs_per_topic.get(topic, [])
        topic_scores = [d.get("score", 0) for d in topic_docs]
        floor = _adaptive_threshold(topic_scores)
        relevant_docs = [d for d in topic_docs if d.get("score", 0) >= floor]
        # Scored against RAG_FULL_COVERAGE_CHUNKS (an absolute "what would
        # comprehensive coverage look like" target), not RAG_TOP_K (just
        # the retriever's fetch limit) -- otherwise fetching exactly
        # RAG_TOP_K chunks and having all of them pass the relevance floor
        # trivially reads as "maximum possible coverage" regardless of
        # whether the topic is actually covered comprehensively.
        rag_ratio = len(relevant_docs) / config.RAG_FULL_COVERAGE_CHUNKS

        coverage_signal = 0.5 * kg_density_norm + 0.5 * min(rag_ratio, 1.0)
        gap_score = round(momentum - coverage_signal, 4)

        results.append(
            {
                "topic": topic,
                "momentum": momentum,
                "kg_coverage_count": kg_count,
                "rag_relevant_chunks": len(relevant_docs),
                "coverage_signal": round(coverage_signal, 4),
                "gap_score": gap_score,
            }
        )

    return sorted(results, key=lambda r: r["gap_score"], reverse=True)


def make_node():
    def node(state: dict) -> dict:
        gap_scores = analyse_gaps(
            trends=state.get("trends", []),
            kg_coverage=state.get("kg_coverage", {}),
            retrieved_docs_per_topic=state.get("retrieved_docs_per_topic", {}),
        )
        trace_entry = {
            "node": "gap_analyser",
            "output": f"scored {len(gap_scores)} topic(s); "
            f"top gap: {gap_scores[0]['topic'] if gap_scores else 'n/a'}",
        }
        return {
            "gap_scores": gap_scores,
            "trace": state.get("trace", []) + [trace_entry],
        }

    return node
