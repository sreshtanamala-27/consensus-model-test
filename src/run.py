"""
run.py

Orchestrates the full pipeline:

  1. load the dataset (fixed IDs, fixed order)
  2. classify independently, then optionally run iterative debate rounds
  3. save every response from every round — nothing is
     thrown away, including failed calls and unparseable responses
  4. compute final-round consensus for analysis only
  5. compute and save aggregate statistics (results/summary.json)

Usage:
    python src/run.py
    python src/run.py --trials 3 --debate-rounds 2 --models openai claude
    python src/run.py --sleep 1.0
"""

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.consensus import build_consensus_rows, normalize_label, summarize  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.models import DEBATE_FUNCS, MODEL_FUNCS  # noqa: E402

ALL_MODELS = ["openai", "deepseek", "claude", "qwen"]
RAW_FIELDS = [
    "trial",
    "round",
    "id",
    "label",
    "model",
    "raw_response",
    "normalized_label",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the multi-model consensus pipeline.")
    parser.add_argument("--data-dir", default="data", help="Directory with depression.csv / neutral.csv")
    parser.add_argument("--output-dir", default="results", help="Where to write results")
    parser.add_argument("--trials", type=int, default=3, help="Number of independent trials")
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=2,
        help="Number of debate rounds after independent Round 1 (default: 2)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="Which models to run (default: all four)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N items per class (for a cheap smoke test)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between model calls (basic rate limiting)",
    )
    return parser.parse_args()


def _record_prediction(
    raw_predictions: list[dict],
    trial: int,
    round_number: int,
    item: dict,
    model_name: str,
    result: dict,
) -> dict:
    raw_response = result.get("raw_response")
    prediction = {
        "trial": trial,
        "round": round_number,
        "id": item["id"],
        "label": item["label"],
        "model": model_name,
        "raw_response": raw_response,
        "normalized_label": normalize_label(raw_response),
        "error": result.get("error"),
    }
    raw_predictions.append(prediction)
    return prediction


def main():
    args = parse_args()
    if args.debate_rounds < 0:
        raise SystemExit("--debate-rounds must be zero or greater")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_dataset(args.data_dir, limit_per_class=args.limit)
    print(
        f"[run] Loaded {len(items)} items. Running {args.trials} trial(s), "
        f"{args.debate_rounds} debate round(s), with models: {args.models}"
    )

    raw_predictions = []
    total_calls = (
        len(items) * args.trials * len(args.models) * (args.debate_rounds + 1)
    )
    done = 0

    for trial in range(1, args.trials + 1):
        for item in items:
            previous_round = {}
            for model_name in args.models:
                try:
                    result = MODEL_FUNCS[model_name](item["post"])
                except Exception as error:  # noqa: BLE001 - log and continue the experiment
                    result = {"raw_response": None, "error": str(error)}
                prediction = _record_prediction(
                    raw_predictions, trial, 1, item, model_name, result
                )
                previous_round[model_name] = prediction
                done += 1
                if done % 20 == 0 or done == total_calls:
                    print(f"[run] {done}/{total_calls} calls complete")

                if args.sleep:
                    time.sleep(args.sleep)

            for round_number in range(2, args.debate_rounds + 2):
                current_round = {}
                for model_name in args.models:
                    own_previous = previous_round[model_name]["raw_response"]
                    other_responses = [
                        previous_round[other_model]["raw_response"]
                        for other_model in args.models
                        if other_model != model_name
                    ]
                    try:
                        result = DEBATE_FUNCS[model_name](
                            item["post"], own_previous, other_responses
                        )
                    except Exception as error:  # noqa: BLE001 - log and continue the experiment
                        result = {"raw_response": None, "error": str(error)}
                    prediction = _record_prediction(
                        raw_predictions,
                        trial,
                        round_number,
                        item,
                        model_name,
                        result,
                    )
                    current_round[model_name] = prediction
                    done += 1
                    if done % 20 == 0 or done == total_calls:
                        print(f"[run] {done}/{total_calls} calls complete")

                    if args.sleep:
                        time.sleep(args.sleep)
                previous_round = current_round

    # --- Save raw predictions (nothing discarded) ---
    raw_path = output_dir / "raw_predictions.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(raw_predictions)
    print(f"[run] Wrote {raw_path}")

    # --- Build consensus for the final response round (analysis only) ---
    final_round = args.debate_rounds + 1
    consensus_rows = [
        row
        for row in build_consensus_rows(raw_predictions)
        if row["round"] == final_round
    ]
    consensus_path = output_dir / "consensus_results.csv"
    if consensus_rows:
        fieldnames = [
            "trial",
            "round",
            "id",
            "label",
        ] + ALL_MODELS + ["consensus", "agreement"]
        with open(consensus_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in consensus_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"[run] Wrote {consensus_path}")

    # --- Save summary statistics + reproducibility metadata ---
    summary = summarize(consensus_rows, raw_predictions)
    summary["reproducibility"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "trials": args.trials,
        "debate_rounds": args.debate_rounds,
        "total_rounds": args.debate_rounds + 1,
        "models_run": args.models,
        "n_items": len(items),
        "dataset_dir": str(Path(args.data_dir).resolve()),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] Wrote {summary_path}")

    print("[run] Done.")


if __name__ == "__main__":
    main()
