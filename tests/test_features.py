"""Unit tests for time-aware features and data splitting."""

import pandas as pd
import pytest

from src.features import MODEL_FEATURES, build_features
from src.prepare_data import split_by_date


def example_transactions():
    return pd.DataFrame([
        (2, "2020-01-02 11:00:00", 1, 40, 50.0, 1),
        (0, "2020-01-01 10:00:00", 1, 40, 10.0, 0),
        (1, "2020-01-01 12:00:00", 1, 41, 20.0, 0),
        (3, "2020-01-01 15:00:00", 2, 42, 100.0, 1),
    ], columns=[
        "TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID",
        "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD"
    ])


def test_prior_average_and_24h_window():
    data = build_features(example_transactions()).set_index("TRANSACTION_ID")
    assert pd.isna(data.loc[0, "CUSTOMER_AVG_AMOUNT_PRIOR"])
    assert data.loc[0, "AMOUNT_VS_AVG"] == 1.0
    assert data.loc[1, "CUSTOMER_AVG_AMOUNT_PRIOR"] == pytest.approx(10.0)
    assert data.loc[1, "AMOUNT_VS_AVG"] == pytest.approx(2.0)
    assert data.loc[1, "TX_COUNT_24H"] == 1
    assert data.loc[2, "CUSTOMER_AVG_AMOUNT_PRIOR"] == pytest.approx(15.0)
    # 10:00 yesterday is 25h ago, while 12:00 yesterday is 23h ago.
    assert data.loc[2, "TX_COUNT_24H"] == 1
    assert data.loc[3, "TX_COUNT_24H"] == 0  # separate customer


def test_labels_do_not_affect_features():
    a = example_transactions()
    b = a.copy()
    b["TX_FRAUD"] = 1 - b["TX_FRAUD"]
    pd.testing.assert_frame_equal(
        build_features(a)[MODEL_FEATURES],
        build_features(b)[MODEL_FEATURES],
    )


def test_splits_are_chronological_and_disjoint():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame({"TX_DATETIME": dates, "TX_FRAUD": [0] * 20})
    s = split_by_date(data)
    assert [len(s[x]) for x in ("train", "val", "test")] == [14, 3, 3]
    assert s["train"].TX_DATETIME.max() < s["val"].TX_DATETIME.min()
    assert s["val"].TX_DATETIME.max() < s["test"].TX_DATETIME.min()
