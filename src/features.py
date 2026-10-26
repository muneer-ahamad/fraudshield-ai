"""Create interpretable, past-only features for fraud detection."""

import numpy as np
import pandas as pd

# Only these five numeric columns will be inputs to our ML models.
MODEL_FEATURES = [
    "TX_AMOUNT",
    "TX_HOUR",
    "IS_WEEKEND",
    "AMOUNT_VS_AVG",
    "TX_COUNT_24H",
]

REQUIRED_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
]


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transactions with behavioral features built from earlier events only.

    Feature meanings:
      - TX_HOUR: hour (0-23) of the transaction
      - IS_WEEKEND: 1 if Saturday/Sunday, otherwise 0
      - AMOUNT_VS_AVG: current amount / customer's prior average amount
      - TX_COUNT_24H: number of customer's transactions in the PREVIOUS 24 hours

    First-time customers have AMOUNT_VS_AVG = 1 (neutral, no history).
    TX_FRAUD is retained for evaluation but is NEVER used to build features.
    """
    missing = set(REQUIRED_COLUMNS) - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing expected dataset columns: {sorted(missing)}")

    df = transactions[REQUIRED_COLUMNS].copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    # The order within each customer's history is essential.
    df = df.sort_values(
        ["CUSTOMER_ID", "TX_DATETIME", "TRANSACTION_ID"], kind="stable"
    ).reset_index(drop=True)

    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["IS_WEEKEND"] = (df["TX_DATETIME"].dt.dayofweek >= 5).astype(int)

    customer = df.groupby("CUSTOMER_ID", sort=True)

    # cumsum includes the current transaction, so subtract its amount.
    # This avoids using the current amount when calculating past averages.
    prior_sum = customer["TX_AMOUNT"].cumsum() - df["TX_AMOUNT"]
    prior_count = customer.cumcount()
    prior_avg = prior_sum.div(prior_count.replace(0, np.nan))
    df["CUSTOMER_AVG_AMOUNT_PRIOR"] = prior_avg
    df["AMOUNT_VS_AVG"] = (
        df["TX_AMOUNT"].div(prior_avg.replace(0, np.nan)).fillna(1.0)
    )

    # closed='left' excludes the current transaction (and any equal-time
    # transactions), ensuring strictly earlier timestamps in the window.
    counts = (
        df.groupby("CUSTOMER_ID", sort=True)
        .rolling("24h", on="TX_DATETIME", closed="left")["TX_AMOUNT"]
        .count()
        .fillna(0)
        .to_numpy()
    )
    df["TX_COUNT_24H"] = counts.astype(int)

    # Put results back into chronological order for time-based splitting.
    return df.sort_values(
        ["TX_DATETIME", "TRANSACTION_ID"], kind="stable"
    ).reset_index(drop=True)
