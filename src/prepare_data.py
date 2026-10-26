"""Load a small slice of the official simulated dataset and prepare ML splits.

Run from the project root:
    python -m src.prepare_data --days 90
"""

import argparse
from pathlib import Path

import pandas as pd

from src.features import MODEL_FEATURES, build_features

DATA_DIR = Path("data/raw/simulated-data-raw/data")
OUTPUT_DIR = Path("data/processed")


def load_transactions(days: int = 90, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load daily pickle files from the Fraud Detection Handbook repository."""
    paths = sorted(data_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(
            f"No .pkl files found in {data_dir}.\n"
            "First run: git clone https://github.com/Fraud-Detection-Handbook/"
            "simulated-data-raw data/raw/simulated-data-raw"
        )
    if days < 10:
        raise ValueError("Use at least 10 days so we can make meaningful splits.")
    if days > len(paths):
        raise ValueError(f"Requested {days} days, but only {len(paths)} files exist.")

    print(f"Loading {days} daily files...")
    return pd.concat((pd.read_pickle(path) for path in paths[:days]), ignore_index=True)


def split_by_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split whole days: first ~70% train, next ~15% val, last ~15% test."""
    dates = df["TX_DATETIME"].dt.normalize()
    unique_days = dates.drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_days) < 10:
        raise ValueError("Need at least 10 distinct transaction dates.")

    val_start = unique_days.iloc[int(len(unique_days) * 0.70)]
    test_start = unique_days.iloc[int(len(unique_days) * 0.85)]

    return {
        "train": df.loc[dates < val_start].copy(),
        "val": df.loc[(dates >= val_start) & (dates < test_start)].copy(),
        "test": df.loc[dates >= test_start].copy(),
    }


def main(days: int) -> None:
    raw = load_transactions(days)
    print(f"Loaded {len(raw):,} transactions")
    prepared = build_features(raw)
    splits = split_by_date(prepared)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Store IDs and customer history for the eventual investigation dashboard,
    # but train only on MODEL_FEATURES (never on IDs or fraud scenario labels).
    output_columns = [
        "TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID",
        *MODEL_FEATURES, "CUSTOMER_AVG_AMOUNT_PRIOR", "TX_FRAUD",
    ]

    for name, part in splits.items():
        destination = OUTPUT_DIR / f"{name}.csv.gz"
        part[output_columns].to_csv(destination, index=False, compression="gzip")
        count = int(part["TX_FRAUD"].sum())
        rate = 100 * part["TX_FRAUD"].mean()
        print(
            f"{name:5s}  rows={len(part):>8,}  fraud={count:>6,}  "
            f"fraud_rate={rate:5.2f}%  "
            f"dates={part['TX_DATETIME'].min().date()}..{part['TX_DATETIME'].max().date()}"
        )
        print(f"       Saved {destination}")

    print("\nSuccess! Next stage: train Logistic Regression and XGBoost.")
    print("Model input columns:", ", ".join(MODEL_FEATURES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FraudShield ML dataset")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of daily files to load (default: 90)")
    arguments = parser.parse_args()
    main(arguments.days)
