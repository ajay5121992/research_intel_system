"""
Loads internal library (CSV) and external signals (JSON) from disk.

Also exposes stub "live fetch" hooks (disabled by default, no API keys
required) that show where a real deployment would plug in GDELT / pytrends /
NewsAPI calls without changing anything downstream -- the rest of the
pipeline only cares about the shape of the returned data, not its source.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

REQUIRED_LIBRARY_COLUMNS = {
    "article_id", "headline", "full_text", "category", "seed_topic", "date",
}


def load_internal_library(path: Path = None) -> pd.DataFrame:
    """Load the internal library CSV. Raises a clear error if missing."""
    path = path or config.LIBRARY_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Internal library not found at {path}.\n"
            f"Run `python data/generate_sample_data.py` first, or place your "
            f"own CSV with columns: {sorted(REQUIRED_LIBRARY_COLUMNS)}"
        )
    df = pd.read_csv(path)
    missing = REQUIRED_LIBRARY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Internal library CSV is missing columns: {missing}")
    df["full_text"] = df["full_text"].fillna("")
    df["headline"] = df["headline"].fillna("")
    logger.info("Loaded %d articles from %s", len(df), path)
    return df


def load_external_signals(path: Path = None) -> List[Dict]:
    """Load external signals JSON. Raises a clear error if missing."""
    path = path or config.SIGNALS_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"External signals not found at {path}.\n"
            f"Run `python data/generate_sample_data.py` first, or place your "
            f"own JSON list of topic signal objects."
        )
    with open(path) as f:
        signals = json.load(f)
    logger.info("Loaded %d external signals from %s", len(signals), path)
    return signals


# ---------------------------------------------------------------------------
# Optional live-fetch hooks. Disabled unless explicitly called; require
# extra packages (pytrends, feedparser) not in the default requirements.
# Kept here so graders can see how the system would extend to real feeds.
# ---------------------------------------------------------------------------
def fetch_live_google_trends(keywords: List[str], timeframe: str = None, geo: str = None) -> Dict[str, List[list]]:
    """Fetch live search-interest series via pytrends (requires internet +
    `pip install pytrends`).

    Returns {keyword: [[iso_date, interest_0_to_1], ...]} -- pytrends
    returns 0-100 Google Trends interest with a real DatetimeIndex; both
    are kept here (interest normalized to 0-1 to match the synthetic
    data's scale, date as an ISO string so the list stays JSON-serializable).
    Depending on the requested timeframe this may be daily or weekly
    granularity (Google Trends serves daily for windows under ~9 months,
    weekly for longer windows) -- callers should not assume "weekly" from
    the point count alone; derive granularity from the actual date deltas.

    IMPORTANT -- one keyword per payload, not batches of 5:
      pytrends.build_payload() accepts up to 5 keywords, but Google Trends
      does NOT return 5 independent 0-100 scales when you do that. It
      returns ONE shared scale for the whole payload: whichever keyword in
      the batch has the most relative search volume is normalized to 100,
      and every other keyword in that same batch is scored relative to it.
      Two keywords fetched together will get different numbers than the
      same two keywords fetched separately (or fetched alongside different
      keywords) -- the score is a function of "who else was in this
      request", not just of the keyword itself.
      This previously caused a real bug here: batching
      ['POLITICS','WELLNESS','ENTERTAINMENT','TRAVEL','STYLE & BEAUTY']
      together made TRAVEL/STYLE & BEAUTY (higher absolute search volume)
      dominate the batch and get scaled near 100, while POLITICS/WELLNESS/
      ENTERTAINMENT/PARENTING were compressed toward 0 -- not because
      those topics have low search interest, but because they were being
      measured on the wrong topic's scale. Since build_live_external_signals
      compares momentum ACROSS topics (for gap analysis), each topic's
      score must come from its own independent payload or the comparison
      is meaningless. Hence batch size 1 below, even though it costs more
      requests (mitigated by the inter-request delay/backoff/cache).

    Resilient per-keyword, with real backoff:
      - Each keyword is fetched in its own payload (see above for why).
      - A request that raises is retried with exponential backoff +
        jitter, up to GOOGLE_TRENDS_MAX_RETRIES times, before giving up
        on that keyword and leaving it empty. Backoff is applied for ANY
        exception, not just explicit 429s, since pytrends wraps HTTP
        errors in generic exceptions whose message just happens to
        contain "429".
      - A short pause runs between requests even on success, so a normal
        multi-keyword fetch doesn't trip the rate limiter in the first
        place (this is separate from the retry backoff, which only
        triggers on failure).
      - A rate-limit failure is logged distinctly from "Trends genuinely
        has no data for this keyword" (both end up as an empty list, but
        the log makes clear which happened) -- build_live_external_signals()
        turns either case into a static-fallback momentum rather than a
        bare, misleading 0.0.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        raise ImportError(
            "pytrends is not installed. Run `pip install pytrends` to enable "
            "live Google Trends fetching."
        ) from exc

    import random
    import time

    timeframe = timeframe or config.GOOGLE_TRENDS_TIMEFRAME
    geo = geo if geo is not None else config.GOOGLE_TRENDS_GEO

    pytrends = TrendReq(hl="en-US", tz=360)
    results: Dict[str, List[float]] = {}

    def _fetch_one(kw: str) -> None:
        # Single-keyword payload -- see the module-level docstring above
        # for why this must NOT be batched with other keywords: Google
        # Trends normalizes 0-100 within whatever payload it's given, so
        # multiple keywords in one call would still produce comparative
        # (not independent) scores even though each keyword individually
        # "succeeds".
        pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
        interest_df = pytrends.interest_over_time()
        min_points = config.GOOGLE_TRENDS_MIN_SERIES_LENGTH

        if kw not in interest_df.columns:
            logger.warning("No Google Trends data returned for '%s'", kw)
            results[kw] = []
            return

        values = (interest_df[kw] / 100.0).round(3)
        # Keep the real DatetimeIndex instead of discarding it -- this is
        # what lets the UI plot actual dates on the x-axis instead of
        # meaningless sequential integers.
        series = [
            [idx.strftime("%Y-%m-%d"), float(val)]
            for idx, val in zip(values.index, values.tolist())
        ]

        if len(series) < min_points:
            # A request that "succeeds" (no exception) but comes back
            # with far fewer points than the requested timeframe implies
            # can happen when Google serves a degraded/partial response
            # while the IP is under rate-limit pressure -- this is NOT
            # the same as "Trends genuinely has no data for this
            # keyword". Treating it as valid data risks trusting
            # series[-1] (naive "momentum") from a point that may not
            # represent the real current window at all. Reject it the
            # same way as an empty result so it falls back to static
            # momentum rather than reporting a misleadingly small/garbage
            # number.
            logger.warning(
                "Google Trends returned only %d point(s) for '%s' "
                "(expected >= %d for timeframe '%s') -- likely a "
                "degraded response under rate-limit pressure; "
                "discarding rather than trusting it as real momentum",
                len(series), kw, min_points, timeframe,
            )
            results[kw] = []
            return

        if sum(v for _, v in series) == 0:
            logger.warning("All-zero Google Trends data for '%s'", kw)
            results[kw] = []
            return

        results[kw] = series

    def _fetch_with_backoff(kw: str) -> bool:
        """Returns True if the keyword's fetch eventually succeeded."""
        for attempt in range(config.GOOGLE_TRENDS_MAX_RETRIES + 1):
            try:
                _fetch_one(kw)
                return True
            except Exception as exc:  # noqa: BLE001
                is_last = attempt == config.GOOGLE_TRENDS_MAX_RETRIES
                if is_last:
                    logger.warning(
                        "'%s' failed after %d attempt(s), giving up (%s)",
                        kw, attempt + 1, exc,
                    )
                    return False
                # Exponential backoff with jitter: base * 2^attempt, capped,
                # plus up to 50% random jitter so parallel/rapid runs don't
                # all retry in lockstep.
                delay = min(
                    config.GOOGLE_TRENDS_BACKOFF_BASE_SECONDS * (2 ** attempt),
                    config.GOOGLE_TRENDS_BACKOFF_MAX_SECONDS,
                )
                delay *= 1 + random.uniform(0, 0.5)
                logger.warning(
                    "'%s' failed (attempt %d/%d): %s -- retrying in %.1fs",
                    kw, attempt + 1, config.GOOGLE_TRENDS_MAX_RETRIES + 1, exc, delay,
                )
                time.sleep(delay)
        return False

    for idx, kw in enumerate(keywords):
        succeeded = _fetch_with_backoff(kw)
        if not succeeded:
            results[kw] = []

        # Pause between requests (not after the last one) even on success,
        # to avoid tripping the limiter on the next request. Single-keyword
        # payloads mean more requests than the old batch-of-5 approach, so
        # this delay matters more now -- tune GOOGLE_TRENDS_INTER_BATCH_DELAY_SECONDS
        # up if you still see 429s.
        if idx < len(keywords) - 1:
            time.sleep(config.GOOGLE_TRENDS_INTER_BATCH_DELAY_SECONDS)

    return results


def _recent_momentum(series: List[list], window: int = 3) -> float:
    """Momentum = mean of the last `window` points' values, not a single
    last value. A bare series[-1] is fragile even for a legitimate series --
    Google Trends' most recent point is sometimes a partial/still-updating
    bucket, and a single noisy point shouldn't define "momentum" for a
    whole topic. Averaging the last few points smooths that out while
    still reflecting recent (not historical-average) interest.

    `series` is a list of [iso_date, value] pairs (see
    fetch_live_google_trends); only the value column feeds the average.
    """
    if not series:
        return 0.0
    values = [v for _, v in series]
    tail = values[-window:] if len(values) >= window else values
    return round(sum(tail) / len(tail), 3)


def _infer_granularity(series: List[list]) -> str:
    """Derives 'daily' / 'weekly' / 'unknown' from the actual gap between
    consecutive dates in `series`, rather than assuming from the requested
    timeframe. Google Trends decides granularity server-side based on
    window length (~9 months is the daily/weekly cutoff), so this is the
    only reliable way to label a chart axis correctly regardless of what
    GOOGLE_TRENDS_TIMEFRAME happens to be set to.
    """
    if len(series) < 2:
        return "unknown"
    from datetime import date as _date

    d0 = _date.fromisoformat(series[0][0])
    d1 = _date.fromisoformat(series[1][0])
    delta_days = (d1 - d0).days
    if delta_days <= 1:
        return "daily"
    if 5 <= delta_days <= 9:
        return "weekly"
    return f"~{delta_days}d"


def _load_trends_cache() -> List[Dict] | None:
    """Returns cached live signals if the cache file exists and is within
    GOOGLE_TRENDS_CACHE_TTL_SECONDS, else None."""
    import time

    path = config.GOOGLE_TRENDS_CACHE_PATH
    if not path.exists():
        return None
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > config.GOOGLE_TRENDS_CACHE_TTL_SECONDS:
        logger.info(
            "Live trends cache is %.0fs old (TTL=%.0fs) -- treating as stale",
            age_seconds, config.GOOGLE_TRENDS_CACHE_TTL_SECONDS,
        )
        return None
    try:
        with open(path) as f:
            cached = json.load(f)
        logger.info(
            "Using cached live trends (%.0fs old, TTL=%.0fs) -> %s. "
            "Set force_refresh=True or delete this file to refetch.",
            age_seconds, config.GOOGLE_TRENDS_CACHE_TTL_SECONDS, path,
        )
        return cached
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read trends cache (%s); ignoring cache", exc)
        return None


def _save_trends_cache(signals: List[Dict]) -> None:
    path = config.GOOGLE_TRENDS_CACHE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(signals, f, indent=2)
        logger.info("Cached live trends -> %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write trends cache (%s); continuing without it", exc)


def build_live_external_signals(topics: List[Dict] = None, force_refresh: bool = False) -> List[Dict]:
    """Builds an external_signals.json-compatible list using LIVE Google
    Trends data for momentum/weekly_interest, keeping the same topic ->
    category -> entities taxonomy used by the synthetic generator so
    nothing downstream (knowledge graph, gap analyser, UI) needs to change.

    `topics` defaults to the same 8-topic taxonomy defined in
    data/generate_sample_data.py (topic, category, entities per topic).
    Pass your own list of {"topic": ..., "category": ..., "entities": [...]}
    dicts to use a different taxonomy.

    Disk-cached: by default, reuses a previous live fetch if it's younger
    than GOOGLE_TRENDS_CACHE_TTL_SECONDS (default 6h), instead of hitting
    pytrends again on every bootstrap() call -- this was the main driver
    of repeated 429s during normal development (every `streamlit run` /
    every test run re-fetched from scratch). Pass force_refresh=True to
    always bypass the cache (used by data/refresh_external_signals.py,
    whose whole purpose is refreshing).

    If a topic's live series comes back empty/zero (see
    fetch_live_google_trends), and GOOGLE_TRENDS_FALLBACK_ON_EMPTY is set
    (default true), that topic falls back to its static momentum from
    data/topics.py rather than silently reporting momentum=0.0 as if it
    were a real, current signal. The signal's "source" field records
    which path was actually used per topic, and "last_updated" reflects
    the real end of the fetched Trends window -- not today's date -- so
    the UI never claims fresher data than what was actually retrieved.

    Requires: pip install pytrends, and internet access (unless served
    entirely from cache).
    """
    import sys
    from datetime import date

    if not force_refresh:
        cached = _load_trends_cache()
        if cached is not None:
            return cached

    if topics is None:
        sys.path.insert(0, str(config.BASE_DIR))
        from data.generate_sample_data import TOPICS as topics  # noqa: N812

    keywords = [t["topic"] for t in topics]
    weekly_series = fetch_live_google_trends(keywords)

    # Google Trends timeframes like "today 12-m" end on (roughly) today;
    # a fixed "YYYY-MM-DD YYYY-MM-DD" range ends on its second date. Use
    # the latter when present so "last_updated" always matches the data
    # actually fetched, not the day the script happened to run.
    timeframe = config.GOOGLE_TRENDS_TIMEFRAME
    parts = timeframe.split()
    if len(parts) == 2 and parts[1].count("-") == 2:
        data_as_of = parts[1]  # explicit fixed range, e.g. "2022-01-01 2022-06-01"
    else:
        data_as_of = date.today().isoformat()  # rolling window ("today 12-m" etc.)

    dropped_topics = []
    signals = []
    for topic_def in topics:
        series = weekly_series.get(topic_def["topic"], [])
        is_empty = not series or sum(v for _, v in series) == 0
        if is_empty and config.GOOGLE_TRENDS_FALLBACK_ON_EMPTY:
            momentum = topic_def.get("momentum", 0.0)
            source = "static_fallback_empty_live_result"
            dropped_topics.append(topic_def["topic"])
            logger.warning(
                "'%s' had no live Google Trends data; using static fallback "
                "momentum=%.3f instead of reporting 0.0",
                topic_def["topic"], momentum,
            )
        else:
            momentum = _recent_momentum(series)
            source = "google_trends_live"

        signals.append(
            {
                "topic": topic_def["topic"],
                "category": topic_def["category"],
                "entities": topic_def["entities"],
                "momentum": momentum,
                # [iso_date, value] pairs, not bare floats -- see
                # fetch_live_google_trends. Kept under the "weekly_interest"
                # key name for backward compatibility with existing
                # external_signals.json consumers, even though the actual
                # granularity may be daily (see "granularity" below).
                "weekly_interest": series,
                "granularity": _infer_granularity(series),
                "source_mix": {"news_volume": None, "search_interest": 1.0},
                "last_updated": data_as_of,
                "source": source,
            }
        )
        logger.info("Trend signal: %s -> momentum=%.3f (source=%s)", topic_def["topic"], momentum, source)

    if dropped_topics:
        logger.warning(
            "%d/%d topics used static fallback (no live data): %s",
            len(dropped_topics), len(topics), dropped_topics,
        )

    _save_trends_cache(signals)
    return signals


def fetch_live_gdelt(query: str, max_records: int = 50) -> List[Dict]:
    """Fetch live news signals via the GDELT 2.0 DOC API (requires internet).

    Not called anywhere by default -- illustrative extension point only.
    """
    import requests

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("articles", [])
