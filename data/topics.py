"""
Single source of truth for the topic taxonomy.

This version instead uses the real dataset's OWN top-level categories as
the topic taxonomy -- specifically HuffPost News Category Dataset's most
common categories. That makes the mapping from raw data -> internal
library exact (a column filter, not a fuzzy match), while still giving
gap analysis genuine variation to work with (these categories have very
different real Google Trends search-interest levels).

Every other script in this project (generate_sample_data.py's offline
synthetic fallback, load_huffpost_by_category.py's real-data loader,
data_loader.py's live Google Trends fetcher) imports TOPICS from here, so
changing the taxonomy only ever requires editing this one file.
"""
# `momentum` is a reasonable static fallback used only if live Google
# Trends fetching isn't available (offline / no pytrends / no internet) --
# see data_loader.build_live_external_signals() for the real path, which
# overwrites this with actual search-interest data.

TOPICS = [
    {"topic": "POLITICS", "category": "POLITICS", "entities": [], "momentum": 0.62, "coverage_bias": "high"},
    {"topic": "WELLNESS", "category": "WELLNESS", "entities": [], "momentum": 0.71, "coverage_bias": "high"},
    {"topic": "ENTERTAINMENT", "category": "ENTERTAINMENT", "entities": [], "momentum": 0.68, "coverage_bias": "high"},
    {"topic": "TRAVEL", "category": "TRAVEL", "entities": [], "momentum": 0.55, "coverage_bias": "medium"},
    {"topic": "STYLE & BEAUTY", "category": "STYLE & BEAUTY", "entities": [], "momentum": 0.58, "coverage_bias": "medium"},
    {"topic": "PARENTING", "category": "PARENTING", "entities": [], "momentum": 0.49, "coverage_bias": "medium"},
    {"topic": "HEALTHY LIVING", "category": "HEALTHY LIVING", "entities": [], "momentum": 0.53, "coverage_bias": "low"},
    {"topic": "BUSINESS", "category": "BUSINESS", "entities": [], "momentum": 0.66, "coverage_bias": "low"},
]

TOPIC_LABELS = [t["topic"] for t in TOPICS]
