"""
Deterministic synthetic-data generator.

Produces:
  1. internal_library.csv   -- a simulated internal research library
  2. external_signals.json  -- simulated external demand signals
                                (news volume + search interest, per topic)

Both are fully synthetic (no scraping, no API keys needed) so the repo
runs end-to-end offline, immediately after `pip install -r requirements.txt`.

To swap in a real dataset later (e.g. HuffPost News Category Dataset for
the library, or GDELT/pytrends for signals), just replace these two output
files -- keep the same column/field names and everything downstream
(indexer, knowledge graph, agent tools) works unchanged.

Run:
    python data/generate_sample_data.py
"""
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from data.topics import TOPICS  # noqa: E402

random.seed(config.SAMPLE_DATA_SEED)

# ---------------------------------------------------------------------------
# Topic universe now lives in data/topics.py (single source of truth, shared
# with the real-data loader and the live-signals fetcher). It defaults to
# HuffPost's own top categories -- see that file's docstring for why.
# ---------------------------------------------------------------------------

# Coverage bias controls how many library articles we generate per topic --
# "low" == market interest is outrunning what the library has, i.e. a gap.
_COVERAGE_COUNTS = {"low": 2, "medium": 6, "high": 14}

_SENTENCE_TEMPLATES = [
    "{topic} continues to reshape how {entity} approaches strategic planning.",
    "Analysts note that {entity} is central to recent developments in {topic}.",
    "A new report examines the impact of {topic} on {entity} and its peers.",
    "Executives at {entity} are recalibrating roadmaps in response to {topic}.",
    "Regulatory attention on {topic} has increased scrutiny of {entity}.",
    "Market participants are watching {entity} closely amid shifts in {topic}.",
    "Industry observers say {topic} could redefine competitive dynamics for {entity}.",
    "Recent data suggests {topic} is accelerating faster than {entity} anticipated.",
]

_HEADLINE_TEMPLATES = [
    "{topic}: What {entity} Needs to Know",
    "{entity} Faces New Pressure From {topic}",
    "Inside {topic}: A Closer Look at {entity}",
    "{topic} Trends Reshape Outlook for {entity}",
    "Q&A: How {entity} Is Responding to {topic}",
]


def _random_date(days_back: int = 180) -> str:
    d = datetime(2026, 3, 30) - timedelta(days=random.randint(0, days_back))
    return d.strftime("%Y-%m-%d")


def _make_article(article_id: int, topic_def: dict) -> dict:
    # The real-data-derived taxonomy (data/topics.py) has no seed entities
    # (HuffPost categories aren't tied to specific named orgs the way the
    # old invented B2B topics were) -- fall back to a generic phrase rather
    # than crashing on an empty random.choice().
    entity = random.choice(topic_def["entities"]) if topic_def["entities"] else "industry observers"
    topic = topic_def["topic"]
    n_sentences = random.randint(3, 6)
    body = " ".join(
        random.choice(_SENTENCE_TEMPLATES).format(topic=topic, entity=entity)
        for _ in range(n_sentences)
    )
    headline = random.choice(_HEADLINE_TEMPLATES).format(topic=topic, entity=entity)
    return {
        "article_id": f"ART{article_id:05d}",
        "headline": headline,
        "full_text": body,
        "category": topic_def["category"],
        "seed_topic": topic,
        "date": _random_date(),
    }


def generate_internal_library() -> pd.DataFrame:
    rows = []
    article_counter = 1

    # Deliberate coverage bias per topic (drives realistic gap analysis)
    for topic_def in TOPICS:
        n = _COVERAGE_COUNTS[topic_def["coverage_bias"]]
        for _ in range(n):
            rows.append(_make_article(article_counter, topic_def))
            article_counter += 1

    # Pad out with more generic articles distributed across topics at random
    # until we hit the configured library size (keeps the corpus realistic).
    while len(rows) < config.SAMPLE_LIBRARY_SIZE:
        topic_def = random.choice(TOPICS)
        rows.append(_make_article(article_counter, topic_def))
        article_counter += 1

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=config.SAMPLE_DATA_SEED).reset_index(drop=True)


def generate_external_signals() -> list:
    """Synthetic fallback used when live Google Trends isn't available.

    Dates are anchored to the end of config.GOOGLE_TRENDS_TIMEFRAME (when
    it's a fixed 'YYYY-MM-DD YYYY-MM-DD' range) so the synthetic path
    reports the same time period the live path would, rather than a
    hardcoded date baked in separately from the rest of the config -- the
    same "today-relative"/stale-date mismatch that affected the live path
    was also present here, just statically (a hardcoded 2026-06-30).
    """
    end_date = _signals_end_date()
    signals = []
    for topic_def in TOPICS:
        # Simulate a simple 8-week news-volume + search-interest time series
        base = topic_def["momentum"]
        series = []
        for week in range(8):
            noise = random.uniform(-0.05, 0.05)
            value = max(0.0, min(1.0, base - (7 - week) * 0.015 + noise))
            point_date = end_date - timedelta(weeks=(7 - week))
            series.append([point_date.strftime("%Y-%m-%d"), round(value, 3)])
        signals.append(
            {
                "topic": topic_def["topic"],
                "category": topic_def["category"],
                "entities": topic_def["entities"],
                "momentum": topic_def["momentum"],
                "weekly_interest": series,
                "granularity": "weekly",
                "source_mix": {
                    "news_volume": round(random.uniform(0.4, 0.9), 2),
                    "search_interest": round(random.uniform(0.4, 0.9), 2),
                },
                "last_updated": end_date.strftime("%Y-%m-%d"),
            }
        )
    return signals


def _signals_end_date() -> datetime:
    """End date for synthetic signals: the second date of
    config.GOOGLE_TRENDS_TIMEFRAME if it's a fixed range, else today.
    Mirrors the same parsing data_loader.build_live_external_signals()
    uses for 'last_updated', so both paths agree.
    """
    parts = config.GOOGLE_TRENDS_TIMEFRAME.split()
    if len(parts) == 2 and parts[1].count("-") == 2:
        return datetime.strptime(parts[1], "%Y-%m-%d")
    return datetime.today()


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    library_df = generate_internal_library()
    library_df.to_csv(config.LIBRARY_CSV, index=False)
    print(f"Wrote {len(library_df)} articles -> {config.LIBRARY_CSV}")

    signals = generate_external_signals()
    with open(config.SIGNALS_JSON, "w") as f:
        json.dump(signals, f, indent=2)
    print(f"Wrote {len(signals)} topic signals -> {config.SIGNALS_JSON}")


if __name__ == "__main__":
    main()
