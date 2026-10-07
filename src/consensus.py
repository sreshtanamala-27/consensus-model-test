"""
consensus.py

Turns raw model text into normalized labels, computes per-item consensus
across the 4 models, and produces the summary statistics described in the
project plan:

  - unanimous agreement rate (4/4)
  - 3/4 majority rate
  - 2-2 tie rate
  - per-model agreement with the consensus label
  - agreement broken out by depression / neutral subset
  - agreement stability across trials
  - Fleiss' kappa
  - Krippendorff's alpha
"""

import re
from collections import Counter

CATEGORIES = ["Low Risk", "Medium Risk", "High Risk"]
UNPARSEABLE = "UNPARSEABLE"

_CLASSIFICATION_LINE = re.compile(
    r"^\s*Classification\s*:\s*(Low Risk|Medium Risk|High Risk)\b",
    re.IGNORECASE | re.MULTILINE,
)

_PATTERNS = [
    (re.compile(r"\bhigh\b", re.IGNORECASE), "High Risk"),
    (re.compile(r"\bmedium\b|\bmoderate\b", re.IGNORECASE), "Medium Risk"),
    (re.compile(r"\blow\b", re.IGNORECASE), "Low Risk"),
]


def normalize_label(raw_response: str | None) -> str:
    """
    Map a model's free-text reply to one of Low Risk / Medium Risk / High Risk.

    Prefer the explicit Classification line so category names in the
    reasoning cannot override the model's stated classification. If that
    line is absent, accept only an unambiguous category mention; otherwise
    mark the response UNPARSEABLE rather than guessing.
    """
    if not raw_response:
        return UNPARSEABLE

    classification_match = _CLASSIFICATION_LINE.search(raw_response)
    if classification_match:
        return next(
            category
            for category in CATEGORIES
            if category.casefold() == classification_match.group(1).casefold()
        )

    matches = {label for pattern, label in _PATTERNS if pattern.search(raw_response)}

    if len(matches) == 1:
        return matches.pop()
    return UNPARSEABLE


def compute_item_consensus(labels: list[str]) -> tuple[str, str]:
    """
    labels: the 4 normalized labels (one per model) for a single item.

    Returns (consensus_label, agreement_str):
      - ("High Risk", "4/4")       all four agree
      - ("High Risk", "3/4")       three of four agree
      - ("TIE", "2/4")             a genuine 2-2 split between two categories
      - ("NO_MAJORITY", "2/4")     top count is 2 but not a clean 2-2 split
                                   (e.g. one UNPARSEABLE muddies it)
      - ("NO_MAJORITY", "1/4")     no repeated label at all
    """
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]

    if top_count >= 3:
        return top_label, f"{top_count}/4"

    if top_count == 2:
        two_count_labels = [label for label, c in counts.items() if c == 2]
        if len(two_count_labels) == 2:
            return "TIE", "2/4"
        return "NO_MAJORITY", "2/4"

    return "NO_MAJORITY", f"{top_count}/4"


def build_consensus_rows(raw_predictions: list[dict]) -> list[dict]:
    """
    raw_predictions: rows shaped like
        {"trial": 1, "id": "1depression", "label": "depression",
         "model": "openai", "raw_response": "...", "normalized_label": "High Risk"}
    as written to results/raw_predictions.csv by run.py.

    Groups by (trial, round, id) and returns one consensus row per item,
    trial, and round.
    """
    grouped: dict[tuple[int, int, str], dict] = {}
    for row in raw_predictions:
        round_number = row.get("round", 1)
        key = (row["trial"], round_number, row["id"])
        grouped.setdefault(
            key,
            {
                "trial": row["trial"],
                "round": round_number,
                "id": row["id"],
                "label": row["label"],
            },
        )
        grouped[key][row["model"]] = row["normalized_label"]

    consensus_rows = []
    for (_trial, _round, _id), row in grouped.items():
        model_labels = [
            row.get(m, UNPARSEABLE) for m in ("openai", "deepseek", "claude", "qwen")
        ]
        consensus, agreement = compute_item_consensus(model_labels)
        row["consensus"] = consensus
        row["agreement"] = agreement
        consensus_rows.append(row)

    return consensus_rows


def fleiss_kappa(consensus_rows: list[dict]) -> float | None:
    """
    Fleiss' kappa across the 4 models, treating every (trial, item) as one
    rated unit and UNPARSEABLE as its own category (so every unit still has
    exactly 4 ratings). Returns None if there's fewer than 2 units.
    """
    categories = CATEGORIES + [UNPARSEABLE]
    units = []
    for row in consensus_rows:
        labels = [row.get(m, UNPARSEABLE) for m in ("openai", "deepseek", "claude", "qwen")]
        units.append([labels.count(c) for c in categories])

    n_units = len(units)
    if n_units < 2:
        return None

    n_raters = 4
    p_j = [sum(unit[j] for unit in units) / (n_units * n_raters) for j in range(len(categories))]
    P_i = [
        (sum(count * count for count in unit) - n_raters) / (n_raters * (n_raters - 1))
        for unit in units
    ]
    P_bar = sum(P_i) / n_units
    P_e = sum(p * p for p in p_j)

    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha(consensus_rows: list[dict]) -> float | None:
    """
    Krippendorff's alpha (nominal) across the 4 models, using the
    `krippendorff` package. Returns None if the package isn't installed or
    there's not enough data.
    """
    try:
        import krippendorff
    except ImportError:
        print(
            "[consensus] 'krippendorff' package not installed — skipping "
            "alpha. Install it with: pip install krippendorff"
        )
        return None

    if len(consensus_rows) < 2:
        return None

    categories = CATEGORIES + [UNPARSEABLE]
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # reliability_data: one row per rater (model), one column per unit (item)
    reliability_data = []
    for model in ("openai", "deepseek", "claude", "qwen"):
        reliability_data.append(
            [cat_to_idx.get(row.get(model, UNPARSEABLE)) for row in consensus_rows]
        )

    return krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement="nominal"
    )


def summarize(
    consensus_rows: list[dict], raw_predictions: list[dict] | None = None
) -> dict:
    """Build the summary.json content from the per-item consensus rows."""
    n = len(consensus_rows)
    if n == 0:
        return {"n_items": 0}

    unanimous = sum(1 for r in consensus_rows if r["agreement"] == "4/4")
    majority = sum(1 for r in consensus_rows if r["agreement"] == "3/4" and r["consensus"] != "NO_MAJORITY")
    ties = sum(1 for r in consensus_rows if r["consensus"] == "TIE")
    no_majority = sum(1 for r in consensus_rows if r["consensus"] == "NO_MAJORITY")

    # Per-model agreement with the item's consensus label (only counted on
    # items that actually reached a consensus label, i.e. not TIE/NO_MAJORITY).
    decided_rows = [r for r in consensus_rows if r["consensus"] in CATEGORIES]
    per_model_agreement = {}
    for model in ("openai", "deepseek", "claude", "qwen"):
        if decided_rows:
            matches = sum(1 for r in decided_rows if r.get(model) == r["consensus"])
            per_model_agreement[model] = matches / len(decided_rows)
        else:
            per_model_agreement[model] = None

    # Agreement by class (depression vs neutral), if both are present.
    by_class = {}
    for cls in ("depression", "neutral"):
        cls_rows = [r for r in consensus_rows if r["label"] == cls]
        if cls_rows:
            by_class[cls] = {
                "n_items": len(cls_rows),
                "unanimous_rate": sum(1 for r in cls_rows if r["agreement"] == "4/4") / len(cls_rows),
                "tie_rate": sum(1 for r in cls_rows if r["consensus"] == "TIE") / len(cls_rows),
            }

    # Stability of the consensus label for the same item across trials.
    by_item: dict[str, list[str]] = {}
    for r in consensus_rows:
        by_item.setdefault(r["id"], []).append(r["consensus"])
    items_with_multiple_trials = {k: v for k, v in by_item.items() if len(v) > 1}
    if items_with_multiple_trials:
        stable = sum(1 for v in items_with_multiple_trials.values() if len(set(v)) == 1)
        cross_trial_stability_rate = stable / len(items_with_multiple_trials)
    else:
        cross_trial_stability_rate = None

    # Per-model stability uses only valid final-round classifications and
    # excludes items with fewer than two valid trials.
    per_model_cross_trial_stability = {}
    for model in ("openai", "deepseek", "claude", "qwen"):
        model_by_item: dict[str, list[str]] = {}
        for row in consensus_rows:
            classification = row.get(model, UNPARSEABLE)
            if classification in CATEGORIES:
                model_by_item.setdefault(row["id"], []).append(classification)
        comparable_items = [
            labels for labels in model_by_item.values() if len(labels) >= 2
        ]
        if comparable_items:
            stable_items = sum(1 for labels in comparable_items if len(set(labels)) == 1)
            per_model_cross_trial_stability[model] = stable_items / len(comparable_items)
        else:
            per_model_cross_trial_stability[model] = None

    per_model_round_changes = {}
    if raw_predictions is not None:
        for model in ("openai", "deepseek", "claude", "qwen"):
            by_trial_item: dict[tuple[int, str], dict[int, str]] = {}
            for row in raw_predictions:
                if row["model"] != model:
                    continue
                by_trial_item.setdefault((row["trial"], row["id"]), {})[
                    row.get("round", 1)
                ] = row.get("normalized_label", UNPARSEABLE)

            changes_by_transition: dict[str, dict[str, int]] = {}
            for rounds in by_trial_item.values():
                ordered_rounds = sorted(rounds)
                for previous_round, current_round in zip(
                    ordered_rounds, ordered_rounds[1:]
                ):
                    previous = rounds[previous_round]
                    current = rounds[current_round]
                    if previous not in CATEGORIES or current not in CATEGORIES:
                        continue
                    transition = f"{previous_round}_to_{current_round}"
                    counts = changes_by_transition.setdefault(
                        transition, {"changes": 0, "valid_comparisons": 0}
                    )
                    counts["valid_comparisons"] += 1
                    if previous != current:
                        counts["changes"] += 1

            total_changes = sum(
                counts["changes"] for counts in changes_by_transition.values()
            )
            total_comparisons = sum(
                counts["valid_comparisons"]
                for counts in changes_by_transition.values()
            )
            per_model_round_changes[model] = {
                "changes": total_changes,
                "valid_comparisons": total_comparisons,
                "change_rate": (
                    total_changes / total_comparisons if total_comparisons else None
                ),
                "by_transition": {
                    transition: {
                        **counts,
                        "change_rate": (
                            counts["changes"] / counts["valid_comparisons"]
                            if counts["valid_comparisons"]
                            else None
                        ),
                    }
                    for transition, counts in changes_by_transition.items()
                },
            }

    return {
        "n_items": n,
        "unanimous_rate": unanimous / n,
        "majority_3_of_4_rate": majority / n,
        "tie_2_2_rate": ties / n,
        "no_majority_rate": no_majority / n,
        "per_model_agreement_with_consensus": per_model_agreement,
        "agreement_by_class": by_class,
        "cross_trial_consensus_stability_rate": cross_trial_stability_rate,
        "per_model_cross_trial_stability": per_model_cross_trial_stability,
        "per_model_round_changes": per_model_round_changes,
        "fleiss_kappa": fleiss_kappa(consensus_rows),
        "krippendorff_alpha": krippendorff_alpha(consensus_rows),
    }
