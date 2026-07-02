"""
Central configuration for the Research Intelligence System.

All values can be overridden via environment variables (see .env.example).
Nothing here requires network access or an LLM to be running -- every
setting has a safe local default so the system boots offline.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed -- env vars can still be set manually

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", DATA_DIR / "index"))

LIBRARY_CSV = DATA_DIR / "internal_library.csv"
SIGNALS_JSON = DATA_DIR / "external_signals.json"

FAISS_INDEX_PATH = INDEX_DIR / "library.faiss"
METADATA_PATH = INDEX_DIR / "library_meta.pkl"
GRAPH_PATH = INDEX_DIR / "knowledge_graph.gpickle"

# ---------------------------------------------------------------------------
# LLM (Ollama) settings
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
# qwen2.5:7b-instruct is a large reasoning model that emits a long <think> block
# before its final answer -- on CPU or a modest GPU a single call can easily
# take 60-100s+. The previous 30s default was shorter than the model's own
# reasoning time for most machines, so nearly every real call hit the
# `requests` timeout, raised an exception, and silently fell back to
# `_mock_synthesis` (the templated "Found N relevant library document(s)..."
# text visible in the ENTITY_RAG_KG screenshot). Raise the default and make
# it easy to override per-machine via env var.
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

# If true, never even attempt to reach Ollama -- always use the deterministic
# mock fallback. Useful for offline grading / CI / first-run demos.
FORCE_MOCK_LLM = os.getenv("FORCE_MOCK_LLM", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------
# Preferred real embedder. Falls back automatically to a hashing embedder
# (scikit-learn HashingVectorizer + TruncatedSVD) if sentence-transformers
# or its model weights are unavailable (no internet / no disk space).
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
)
EMBEDDING_DIM_FALLBACK = int(os.getenv("EMBEDDING_DIM_FALLBACK", "128"))
FORCE_HASHING_EMBEDDER = os.getenv("FORCE_HASHING_EMBEDDER", "false").lower() in (
    "1", "true", "yes",
)

# ---------------------------------------------------------------------------
# NER settings
# ---------------------------------------------------------------------------
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_lg")
FORCE_REGEX_NER = os.getenv("FORCE_REGEX_NER", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Retrieval / graph tuning
# ---------------------------------------------------------------------------
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
TREND_TOP_N = int(os.getenv("TREND_TOP_N", "5"))
KG_TRAVERSAL_DEPTH = int(os.getenv("KG_TRAVERSAL_DEPTH", "2"))

# Minimum FAISS cosine similarity (IndexFlatIP on L2-normalized vectors,
# so this is a true cosine score in ~[-1, 1]) for a retrieved chunk to
# count as "relevant" in gap_analyser's rag_ratio. The old default of
# 0.05 was cleared by virtually any top-k result regardless of actual
# relevance, which made rag_relevant_chunks == RAG_TOP_K for every topic
# and removed all discriminative power from that half of coverage_signal.
# 0.05 is closer to "not orthogonal" than "actually relevant" for a
# sentence embedding space -- 0.3 is a more realistic floor for
# short-document, single-domain corpora like this one. Tune per corpus.
RAG_RELEVANCE_FLOOR = float(os.getenv("RAG_RELEVANCE_FLOOR", "0.3"))

# --- Gap-analysis coverage_signal normalization ----------------------------
# coverage_signal = 0.5*kg_density_norm + 0.5*rag_ratio previously normalized
# kg_density_norm against the CURRENT topic set's own max article count, and
# rag_ratio against RAG_TOP_K (the retriever's own fetch limit). Both are
# relative-to-self ceilings: whichever topic happens to have the most
# articles is *guaranteed* kg_density_norm=1.0 by construction (dividing by
# its own max), and any topic whose 5 fetched chunks are all relevant hits
# rag_ratio=1.0 simply because only 5 were ever fetched. On a real,
# well-populated library (e.g. 10k HuffPost articles) this pins
# coverage_signal near 1.0 for most/all topics, so gap_score = momentum -
# coverage_signal comes out negative almost everywhere and loses the
# ability to express genuine over- vs under-coverage in absolute terms.
#
# Fix: normalize both halves against an ABSOLUTE target instead of a
# self-relative maximum, so "average" coverage lands near the middle of the
# 0-1 range rather than at a ceiling that's trivially reachable.
#
# kg_density_norm = kg_count / (mean article count across topics * this
# multiplier). 1.0 means "at or above <multiplier>x the average topic's
# coverage" -- e.g. multiplier=2.0 means a topic needs DOUBLE the average
# article count to read as "fully covered", leaving real headroom below
# that for topics with average or below-average coverage.
KG_COVERAGE_TARGET_MULTIPLIER = float(
    os.getenv("KG_COVERAGE_TARGET_MULTIPLIER", "2.0")
)

# rag_ratio = relevant_chunks / RAG_FULL_COVERAGE_CHUNKS, instead of
# / RAG_TOP_K. Keeping RAG_TOP_K as the retrieval fetch count (a separate,
# performance-oriented knob) but scoring "full coverage" against a higher
# target means 5/5 relevant fetched chunks no longer automatically reads as
# "maximum possible coverage" -- it reads as 5 out of a larger expected
# target, leaving headroom for genuinely comprehensive topics to still
# outscore thinly-covered ones without both pinning at 1.0.
RAG_FULL_COVERAGE_CHUNKS = int(os.getenv("RAG_FULL_COVERAGE_CHUNKS", "15"))


# ---------------------------------------------------------------------------
# External signals source
# ---------------------------------------------------------------------------
# If true, bootstrap fetches live Google Trends data (via pytrends) instead
# of reading the synthetic data/external_signals.json file. Requires
# `pip install pytrends` and internet access. See src/data_loader.py
# build_live_external_signals() for the topic->keyword mapping used.
USE_LIVE_GOOGLE_TRENDS = os.getenv("USE_LIVE_GOOGLE_TRENDS", "true").lower() in (
    "1", "true", "yes",
)

# Fixed historical window matching the internal library's article dates
# (2012-01-28 -> 2022-09-23), so external "market momentum" signals are
# comparable to what the library actually covers, instead of a rolling
# "today 12-m" window that reports on a completely different period than
# the library (this was the root cause of "Signals as of: 2026-07-02"
# showing up next to a library that ends in 2022).
#
# Google Trends is a live service but DOES support genuinely historical
# fixed-date queries -- 'YYYY-MM-DD YYYY-MM-DD' is standard pytrends syntax,
# and Trends' underlying index covers 2004 onward, so this returns real
# historical search-interest for the window below, not today's data
# mislabeled.
#
# Window chosen: the quarter immediately following the library's last
# article (2022-09-23), i.e. "what emerged right after our library stops
# covering things" -- exactly the kind of gap the gap-analysis tab is
# meant to surface. Override via GOOGLE_TRENDS_TIMEFRAME if your library's
# date range changes.
#
# NOTE: Google Trends serves DAILY granularity for windows under ~9
# months (WEEKLY for longer windows). This 91-day window will therefore
# return ~91 daily points, not weekly ones -- the UI derives its axis
# label from the actual point count/date spacing rather than assuming
# "weekly", so this is handled correctly downstream.
GOOGLE_TRENDS_TIMEFRAME = os.getenv("GOOGLE_TRENDS_TIMEFRAME", "2022-10-01 2022-12-31")
GOOGLE_TRENDS_GEO = os.getenv("GOOGLE_TRENDS_GEO", "")  # "" = worldwide, e.g. "US"

# If a topic's live Google Trends series comes back empty or entirely
# flat-zero (batch failure, rate limit, or a keyword Trends has no data
# for), fall back to that topic's static momentum from data/topics.py
# instead of silently reporting momentum=0.0 as if it were a real signal.
GOOGLE_TRENDS_FALLBACK_ON_EMPTY = os.getenv(
    "GOOGLE_TRENDS_FALLBACK_ON_EMPTY", "true"
).lower() in ("1", "true", "yes")

# Minimum number of points a "successful" (no-exception) pytrends response
# must contain before it's trusted as real data. A request that returns
# without error but with far fewer points than the requested timeframe
# implies can happen when Google serves a degraded/truncated response
# while the IP is under rate-limit pressure -- this is different from a
# clean empty result and was previously trusted at face value, letting a
# single noisy/truncated last point become "momentum" for that topic.
# The default "2022-10-01 2022-12-31" window (~91 days, daily granularity)
# should return ~91 points; 8 is a conservative floor that still accepts
# much shorter explicit timeframes without false-positiving on those.
GOOGLE_TRENDS_MIN_SERIES_LENGTH = int(os.getenv("GOOGLE_TRENDS_MIN_SERIES_LENGTH", "8"))

# --- Rate-limit resilience -------------------------------------------------
# pytrends is unauthenticated and Google throttles it hard (HTTP 429),
# especially across repeated runs in a short window. Two mitigations:
#
# 1. Exponential backoff with jitter on 429s, applied both at the batch
#    level and the individual-keyword retry level.
# 2. A pause between batches even on success, so consecutive requests
#    don't trip the limiter in the first place.
#
# In practice, once Google starts 429-ing your IP it tends to keep doing
# so for a while (minutes), not just for one request -- so a high retry
# count mostly means burning several minutes before falling back anyway.
# 3 retries with a real inter-batch delay fails fast into the (perfectly
# fine) static momentum fallback rather than stalling a demo.
GOOGLE_TRENDS_MAX_RETRIES = int(os.getenv("GOOGLE_TRENDS_MAX_RETRIES", "3"))
GOOGLE_TRENDS_BACKOFF_BASE_SECONDS = float(os.getenv("GOOGLE_TRENDS_BACKOFF_BASE_SECONDS", "4"))
GOOGLE_TRENDS_BACKOFF_MAX_SECONDS = float(os.getenv("GOOGLE_TRENDS_BACKOFF_MAX_SECONDS", "60"))
GOOGLE_TRENDS_INTER_BATCH_DELAY_SECONDS = float(
    os.getenv("GOOGLE_TRENDS_INTER_BATCH_DELAY_SECONDS", "5")
)

# --- Disk cache for live signals -------------------------------------------
# bootstrap() previously re-fetched live Google Trends data on every single
# process start (every `streamlit run`, every test run), which is the
# fastest way to get rate-limited. Cache the fetched signals to disk with a
# TTL; within the TTL window, bootstrap reuses the cached signals instead
# of hitting pytrends again. Delete data/external_signals_live_cache.json
# or wait out the TTL to force a real refetch, or call
# data/refresh_external_signals.py directly (which always bypasses the
# cache, by design, since that script's whole purpose is refreshing).
GOOGLE_TRENDS_CACHE_PATH = DATA_DIR / "external_signals_live_cache.json"
GOOGLE_TRENDS_CACHE_TTL_SECONDS = int(os.getenv("GOOGLE_TRENDS_CACHE_TTL_SECONDS", str(6 * 3600)))


SAMPLE_DATA_SEED = int(os.getenv("SAMPLE_DATA_SEED", "42"))
SAMPLE_LIBRARY_SIZE = int(os.getenv("SAMPLE_LIBRARY_SIZE", "220"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")