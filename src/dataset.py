"""
dataset.py

Loads the depression / neutral CSVs, assigns fixed, reproducible IDs, and
returns a single ordered list of items to run through the models.

Each item is a dict:
    {
        "id": "1depression" | "1neutral" | ...,
        "label": "depression" | "neutral",
        "post": "<raw text>",
    }

Fixed IDs and fixed row order are preserved across runs as long as the
input CSVs don't change, so re-running the pipeline (e.g. for a second
or third trial) always sends the exact same items in the exact same
order to every model.
"""

import csv
import os
from pathlib import Path


def _load_csv(path: Path, label: str):
    """Load one class's CSV. Returns [] if the file doesn't exist or is empty."""
    if not path.exists():
        return []

    items = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post = (row.get("post") or "").strip()
            if not post:
                continue
            items.append(
                {
                    "id": row.get("id") or f"{len(items) + 1}{label}",
                    "label": row.get("label") or label,
                    "post": post,
                }
            )
    return items


def load_dataset(data_dir: str = "data", limit_per_class: int | None = None):
    """
    Load depression.csv and neutral.csv from data_dir.

    limit_per_class: if set, truncate each class to this many rows
    (useful for a cheap smoke test before running the full sample).

    Returns a flat list of items, depression rows first, then neutral rows,
    each keeping the fixed 'id' column from the CSV.
    """
    data_dir = Path(data_dir)

    depression = _load_csv(data_dir / "depression.csv", "depression")
    neutral = _load_csv(data_dir / "neutral.csv", "neutral")

    if limit_per_class is not None:
        depression = depression[:limit_per_class]
        neutral = neutral[:limit_per_class]

    if not depression and not neutral:
        raise FileNotFoundError(
            f"No data found in {data_dir}. Expected depression.csv and/or "
            f"neutral.csv with columns: id,label,author,date,post"
        )

    if not neutral:
        print(
            "[dataset] Warning: neutral.csv has no rows. Running on the "
            "depression class only — per-class agreement comparisons and "
            "any depression-vs-neutral analysis will be skipped."
        )

    return depression + neutral


if __name__ == "__main__":
    items = load_dataset(os.path.join(os.path.dirname(__file__), "..", "data"))
    print(f"Loaded {len(items)} items.")
    for item in items[:3]:
        print(item["id"], item["label"], item["post"][:60], "...")
