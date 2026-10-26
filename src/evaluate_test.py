"""Evaluate the frozen model and threshold on the held-out test set."""

import json
from pathlib import Path

import joblib

from src.features import MODEL_FEATURES
from src.train import load_split, evaluate


ARTIFACTS = Path("artifacts")


def main():
    output_path = ARTIFACTS / "test_metrics.json"

    # Prevent accidentally overwriting the final evaluation.
    if output_path.exists():
        raise FileExistsError(
            "test_metrics.json already exists. Keep the original evaluation; "
            "do not repeatedly tune against the test set."
        )

    policy = json.loads(
        (ARTIFACTS / "threshold_selection.json").read_text(
            encoding="utf-8"
        )
    )

    model_name = policy["selected_model"]
    threshold = policy["selected_threshold"]

    # Load our own saved model. Never load untrusted joblib files.
    model = joblib.load(ARTIFACTS / f"{model_name}.joblib")
    test = load_split(Path("data/processed/test.csv.gz"))

    labels = test["TX_FRAUD"]
    scores = model.predict_proba(test[MODEL_FEATURES])[:, 1]

    metrics = evaluate(labels, scores, threshold)

    report = {
        "evaluation_split": "test",
        "model": model_name,
        "threshold": threshold,
        "transactions": len(test),
        "fraud_cases": int(labels.sum()),
        "metrics": metrics,
        "note": (
            "Synthetic dataset. Model and threshold selected on validation "
            "data before test evaluation."
        ),
    }

    output_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nFINAL TEST RESULTS — {model_name}")
    print(f"Threshold:          {threshold:.8f}")
    print(f"Transactions:       {len(test):,}")
    print(f"Actual fraud cases: {int(labels.sum()):,}")
    print()
    print(f"Average Precision:  {metrics['average_precision']:.4f}")
    print(f"Precision:          {metrics['precision']:.2%}")
    print(f"Recall:             {metrics['recall']:.2%}")
    print(f"F1:                 {metrics['f1']:.4f}")
    print(f"False-positive rate:{metrics['false_positive_rate']: .2%}")
    print()
    print(f"Fraud caught:       {metrics['true_positives']:,}")
    print(f"Fraud missed:       {metrics['false_negatives']:,}")
    print(f"False alarms:       {metrics['false_positives']:,}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()