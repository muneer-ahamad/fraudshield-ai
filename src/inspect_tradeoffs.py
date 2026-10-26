"""Explore validation trade-offs without changing the saved model or policy."""

from pathlib import Path

import pandas as pd

from src.select_threshold import find_threshold


ARTIFACTS = Path("artifacts")


def main():
    df = pd.read_csv(
        ARTIFACTS / "validation_predictions.csv.gz"
    )

    labels = df["TX_FRAUD"]
    rows = []

    for model_name in ["logistic_regression", "xgboost"]:
        scores = df[model_name]

        for fpr_limit in [0.001, 0.0025, 0.005, 0.01]:
            result = find_threshold(labels, scores, fpr_limit)

            rows.append({
                "model": model_name,
                "FPR_limit_pct": fpr_limit * 100,
                "threshold": result["threshold"],
                "precision_pct": result["precision"] * 100,
                "recall_pct": result["recall"] * 100,
                "actual_FPR_pct": (
                    result["false_positive_rate"] * 100
                ),
                "fraud_caught": result["fraud_caught"],
                "false_alarms": result["false_alarms"],
            })

    comparison = pd.DataFrame(rows)

    print("\nVALIDATION THRESHOLD TRADE-OFFS\n")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    destination = ARTIFACTS / "validation_tradeoffs.csv"
    comparison.to_csv(destination, index=False)

    print(f"\nSaved: {destination}")
    print("Saved models and selected threshold were NOT changed.")
    print("Test data was NOT loaded.")


if __name__ == "__main__":
    main()