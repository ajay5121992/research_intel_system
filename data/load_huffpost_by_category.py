"""
Build internal_library.csv from the REAL HuffPost News Category Dataset,
using HuffPost's own category labels as the topic taxonomy (see
data/topics.py). Because the taxonomy IS the dataset's own categories,
this is an exact column filter -- no keyword/entity matching, no risk of
"0 articles matched", no dataset-mismatch problem.

Usage:
    python data/load_huffpost_by_category.py \\
        --input /path/to/News_Category_Dataset_v3.json \\
        --out data/internal_library.csv \\
        --max-total 5000

--max-total is a total cap across ALL topics combined, split evenly
per-topic (default 5000, i.e. ~625 articles per topic across the 8
categories in data/topics.py) -- large enough for a real, dense RAG
corpus, capped so indexing/embedding stays fast on a laptop.

After running this, rebuild the index + knowledge graph:
    python -m src.bootstrap
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.topics import TOPIC_LABELS  # noqa: E402


def load_huffpost_json(path: Path) -> pd.DataFrame:
    """HuffPost's News_Category_Dataset_v3.json is newline-delimited JSON
    with fields: category, headline, authors, link, short_description, date."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    required = {"category", "headline", "short_description", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing expected HuffPost columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to News_Category_Dataset_v3.json")
    parser.add_argument("--out", default="data/internal_library.csv")
    parser.add_argument(
        "--max-total", type=int, default=5000,
        help="Total article cap across all topics combined (default: 5000)",
    )
    parser.add_argument(
        "--topics", nargs="*", default=None,
        help="Override which HuffPost categories to keep (default: data/topics.py TOPIC_LABELS)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling when a category has more articles than its per-topic cap",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    wanted_topics = args.topics or TOPIC_LABELS
    print(f"Filtering to {len(wanted_topics)} categories: {wanted_topics}")

    raw_df = load_huffpost_json(input_path)
    print(f"Loaded {len(raw_df)} total raw articles from {input_path}")

    # Exact match against the dataset's own category column -- no fuzzy
    # logic, so this can't silently drop everything the way keyword
    # matching against an unrelated taxonomy did.
    filtered = raw_df[raw_df["category"].isin(wanted_topics)].copy()
    print(f"{len(filtered)} articles match the selected categories (before capping).")

    print("\nPer-category counts (available / after cap):")
    per_topic_cap = max(args.max_total // max(len(wanted_topics), 1), 1)
    kept_parts = []
    for topic in wanted_topics:
        group = filtered[filtered["category"] == topic]
        available = len(group)
        kept = group.sample(n=min(available, per_topic_cap), random_state=args.seed) if available else group
        print(f"  {topic}: {available} available -> keeping {len(kept)}")
        kept_parts.append(kept)

    result = pd.concat(kept_parts, ignore_index=True) if kept_parts else filtered.iloc[0:0]
    if result.empty:
        print(
            "\nERROR: no articles matched any selected category. Check that "
            "--topics values exactly match the category strings used in "
            "your input file's 'category' column (case-sensitive).",
            file=sys.stderr,
        )
        sys.exit(1)

    result = result.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle

    out_df = pd.DataFrame({
        "article_id": [f"ART{i+1:05d}" for i in range(len(result))],
        "headline": result["headline"].fillna(""),
        "full_text": result["short_description"].fillna(""),
        "category": result["category"],
        "seed_topic": result["category"],  # topic == category by design (see data/topics.py)
        "date": result["date"].fillna(""),
    })

    # Drop rows with no usable text at all -- an empty chunk can't be
    # embedded/retrieved meaningfully.
    before = len(out_df)
    out_df = out_df[(out_df["headline"].str.strip() != "") | (out_df["full_text"].str.strip() != "")]
    dropped_empty = before - len(out_df)
    if dropped_empty:
        print(f"\nDropped {dropped_empty} article(s) with no headline or description text.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} real articles -> {out_path}")
    print("\nNext: rebuild the FAISS index and knowledge graph:")
    print("  python -m src.bootstrap")


if __name__ == "__main__":
    main()
