"""Contract tests use a tiny fixture, not the user's trained model or test data."""
import json
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import create_app, Features
from src.features import MODEL_FEATURES

PAYLOAD = dict(TX_AMOUNT=100.0, TX_HOUR=12, IS_WEEKEND=0,
               AMOUNT_VS_AVG=2.0, TX_COUNT_24H=3)


class FixtureModel:
    feature_names_in_ = np.array(MODEL_FEATURES)
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        assert list(frame.columns) == MODEL_FEATURES
        score = .75 if frame.iloc[0].TX_AMOUNT >= 100 else .25
        return np.array([[1-score, score]])


@pytest.fixture
def client(tmp_path):
    joblib.dump(FixtureModel(), tmp_path / "xgboost.joblib")
    (tmp_path / "threshold_selection.json").write_text(json.dumps({
        "selected_model": "xgboost", "selected_threshold": .75}))
    with TestClient(create_app(tmp_path)) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.parametrize("amount,decision", [(100., "flag_for_review"), (10., "not_flagged")])
def test_decisions_and_threshold_boundary(client, amount, decision):
    response = client.post("/predict", json={**PAYLOAD, "TX_AMOUNT": amount})
    assert response.status_code == 200
    assert response.json()["decision"] == decision
    assert response.json()["threshold"] == .75
    assert response.json()["calibrated"] is False


@pytest.mark.parametrize("field,value", [("TX_AMOUNT", -1), ("TX_HOUR", 24),
    ("TX_HOUR", 1.5), ("IS_WEEKEND", 2), ("IS_WEEKEND", True),
    ("TX_COUNT_24H", -1), ("AMOUNT_VS_AVG", -1), ("TX_AMOUNT", "100"),
    ("TX_FRAUD", 1)])
def test_invalid_inputs(client, field, value):
    assert client.post("/predict", json={**PAYLOAD, field: value}).status_code == 422


def test_missing_input(client):
    assert client.post("/predict", json={}).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_input(value):
    with pytest.raises(ValueError):
        Features(**{**PAYLOAD, "TX_AMOUNT": value})


def test_missing_artifacts_fail_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path)):
            pass
