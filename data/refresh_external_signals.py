"""
Refresh data/external_signals.json with REAL Google Trends search-interest
data for the topics in data/topics.py, instead of synthetic momentum
scores.

Usage:
    pip install pytrends
    python data/refresh_external_signals.py

Falls back to the synthetic generator (data/generate_sample_data.py's
generate_external_signals()) if pytrends isn't installed or the network
call fails -- so this is always safe to run, even offline.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import build_live_external_signals  # noqa: E402


def main():
    print("Fetching live Google Trends data for:", end=" ")
    from data.topics import TOPIC_LABELS
    print(TOPIC_LABELS)

    try:
        signals = build_live_external_signals(force_refresh=True)
        source = "google_trends_live"
    except Exception as exc:  # noqa: BLE001
        print(f"\nLive fetch failed ({exc}); falling back to synthetic signals.")
        from data.generate_sample_data import generate_external_signals
        signals = generate_external_signals()
        source = "synthetic_fallback"

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SIGNALS_JSON, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"\nWrote {len(signals)} topic signals ({source}) -> {config.SIGNALS_JSON}")
    for s in sorted(signals, key=lambda s: s.get("momentum", 0), reverse=True):
        print(f"  {s['topic']}: momentum={s.get('momentum', 0):.3f}")


if __name__ == "__main__":
    main()
