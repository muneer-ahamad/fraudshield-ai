"""Compare models at a maximum 1% validation false-positive rate."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACTS = Path("artifacts")
MAX_FPR = 0.01

MODEL_NAMES = ["logistic_regression", "xgboost"]


def find_threshold(labels, scores, max_fpr):
    # Combine transactions with identical scores so ties are flagged together.
    grouped = (
        pd.DataFrame({"score": scores, "fraud": labels})
        .groupby("score")["fraud"]
        .agg(["sum", "count"])
        .sort_index(ascending=False)
    )

    # Lowering the threshold includes progressively more transactions.
    true_positives = grouped["sum"].cumsum()
    false_positives = (grouped["count"] - grouped["sum"]).cumsum()

    total_fraud = int(labels.sum())
    total_legitimate = int((labels == 0).sum())

    candidates = pd.DataFrame({
        "threshold": grouped.index.to_numpy(),
        "true_positives": true_positives.to_numpy(),
        "false_positives": false_positives.to_numpy(),
    })

    # Include the option to flag nothing.
    no_flags = pd.DataFrame([{
        "threshold": float(np.nextafter(scores.max(), np.inf)),
        "true_positives": 0,
        "false_positives": 0,
    }])
    candidates = pd.concat([no_flags, candidates], ignore_index=True)

    candidates["recall"] = candidates["true_positives"] / total_fraud
    candidates["false_positive_rate"] = (
        candidates["false_positives"] / total_legitimate
    )

    flagged = candidates["true_positives"] + candidates["false_positives"]
    candidates["precision"] = (
        candidates["true_positives"]
        .div(flagged.replace(0, np.nan))
        .fillna(0)
    )

    eligible = candidates[
        candidates["false_positive_rate"] <= max_fpr
    ]

    # Maximize recall; break ties by fewer false positives,
    # then by the higher threshold.
    best = eligible.sort_values(
        ["recall", "false_positives", "threshold"],
        ascending=[False, True, False],
    ).iloc[0]

    return {
        "threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "false_positive_rate": float(best["false_positive_rate"]),
        "fraud_caught": int(best["true_positives"]),
        "fraud_missed": total_fraud - int(best["true_positives"]),
        "false_alarms": int(best["false_positives"]),
    }


def main():
    path = ARTIFACTS / "validation_predictions.csv.gz"
    df = pd.read_csv(path)
    labels = df["TX_FRAUD"]

    if set(labels.unique()) != {0, 1}:
        raise ValueError("Validation data must contain both classes: 0 and 1.")

    results = {}
    for name in MODEL_NAMES:
        scores = df[name]
        if not np.isfinite(scores).all() or not scores.between(0, 1).all():
            raise ValueError(f"Invalid prediction scores for {name}.")

        results[name] = find_threshold(labels, scores, MAX_FPR)

    comparison = pd.DataFrame(results).T
    print(f"\nVALIDATION comparison: maximum FPR = {MAX_FPR:.1%}\n")
    print(comparison.to_string(float_format=lambda x: f"{x:.6f}"))

    selected = max(
        results,
        key=lambda name: (
            results[name]["recall"],
            -results[name]["false_positive_rate"],
        ),
    )

    policy = {
        "selection_split": "validation",
        "selection_rule": "Highest recall with FPR <= 0.01",
        "max_false_positive_rate": MAX_FPR,
        "selected_model": selected,
        "selected_threshold": results[selected]["threshold"],
        "results": results,
    }

    destination = ARTIFACTS / "threshold_selection.json"
    destination.write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    comparison.to_csv(ARTIFACTS / "threshold_comparison.csv")

    print(f"\nSelected model: {selected}")
    print(f"Selected threshold: {policy['selected_threshold']:.8f}")
    print(f"Saved: {destination}")
    print("Test set remains untouched.")


if __name__ == "__main__":
    main()