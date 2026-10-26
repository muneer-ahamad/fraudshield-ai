"""Train two models and compare validation results; never open the test set."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_score,
    recall_score, f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import MODEL_FEATURES


def load_split(path):
    df = pd.read_csv(path)
    required = [*MODEL_FEATURES, "TX_FRAUD", "TX_DATETIME"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if set(df["TX_FRAUD"].unique()) != {0, 1}:
        raise ValueError(f"{path} must contain both binary classes, 0 and 1.")
    if not np.isfinite(df[MODEL_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains missing or infinite model inputs.")
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"], errors="raise")
    if df["TX_DATETIME"].isna().any():
        raise ValueError("Missing transaction timestamps.")
    return df


def evaluate(y, scores, threshold=0.5):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "threshold": threshold,
    }


def main(data_dir, output_dir):
    train = load_split(data_dir / "train.csv.gz")
    val = load_split(data_dir / "val.csv.gz")
    if train["TX_DATETIME"].max() >= val["TX_DATETIME"].min():
        raise ValueError("Training must end strictly before validation begins.")
    X_train, y_train = train[MODEL_FEATURES], train["TX_FRAUD"]
    X_val, y_val = val[MODEL_FEATURES], val["TX_FRAUD"]
    negative_to_positive = float((y_train == 0).sum() / (y_train == 1).sum())
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=negative_to_positive,
            tree_method="hist", eval_metric="logloss", n_jobs=4, random_state=42,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    predictions = val[["TX_DATETIME", "TX_FRAUD"]].copy()
    for name, model in models.items():
        print(f"Training {name} on {len(train):,} transactions...", flush=True)
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        scores = model.predict_proba(X_val)[:, 1]
        results[name] = evaluate(y_val, scores)
        results[name]["fit_seconds"] = fit_seconds
        predictions[name] = scores
        joblib.dump(model, output_dir / f"{name}.joblib")
    comparison = pd.DataFrame(results).T
    print("\nVALIDATION results (threshold = 0.5; scores are not calibrated):")
    print(comparison[["average_precision", "precision", "recall", "f1", "false_positive_rate"]].to_string(float_format=lambda v: f"{v:.4f}"))
    best = max(results, key=lambda name: results[name]["average_precision"])
    report = {
        "evaluation_split": "validation", "selection_metric": "average_precision",
        "provisional_best_model": best, "features": MODEL_FEATURES,
        "train_rows": len(train), "validation_rows": len(val),
        "validation_fraud_rate": float(y_val.mean()), "models": results,
        "note": "Synthetic data. Test set untouched. Threshold 0.5 is a starting point.",
    }
    (output_dir / "validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    comparison.to_csv(output_dir / "model_comparison.csv")
    predictions.to_csv(output_dir / "validation_predictions.csv.gz", index=False)
    print(f"\nProvisional best by validation Average Precision: {best}")
    print(f"Saved models and validation reports to {output_dir}")
    print("Test data has NOT been loaded. Do not report these as final test metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
