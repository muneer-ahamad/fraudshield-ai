"""Serve the original saved policy over prepared transaction features."""
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from src.features import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    TX_AMOUNT: float = Field(ge=0, description="Amount in dataset units")
    TX_HOUR: int = Field(ge=0, le=23)
    IS_WEEKEND: int = Field(ge=0, le=1)
    AMOUNT_VS_AVG: float = Field(ge=0, description="Amount divided by prior customer average; use training fallback 1.0 if no usable history")
    TX_COUNT_24H: int = Field(ge=0, description="Customer transactions strictly before this event within the previous 24h")


class Prediction(BaseModel):
    score: float = Field(ge=0, le=1)
    threshold: float
    decision: Literal["flag_for_review", "not_flagged"]
    model: str
    model_sha256: str
    policy: str = "original_validation_selected"
    calibrated: bool = False


def create_app(artifact_dir: Path = ROOT / "artifacts"):
    @asynccontextmanager
    async def lifespan(app):
        policy = json.loads((artifact_dir / "threshold_selection.json").read_text(encoding="utf-8"))
        name = policy["selected_model"]
        if name not in {"xgboost", "logistic_regression"}:
            raise ValueError("Unsupported saved model name")
        threshold = float(policy["selected_threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Saved threshold must be finite and between 0 and 1")
        path = artifact_dir / f"{name}.joblib"
        fitted = joblib.load(path)
        if list(fitted.feature_names_in_) != MODEL_FEATURES:
            raise ValueError("Saved model feature order does not match MODEL_FEATURES")
        if list(fitted.classes_) != [0, 1]:
            raise ValueError("Expected binary classes [0, 1]")
        app.state.model = fitted
        app.state.model_name = name
        app.state.threshold = threshold
        app.state.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        yield

    app = FastAPI(title="FraudShield AI", version="1.0.0", lifespan=lifespan,
        description="Synthetic-data portfolio demo. Accepts prepared features, not raw customer histories. Scores are uncalibrated.")

    @app.get("/health")
    def health(request: Request):
        state = request.app.state
        return {"status": "ready", "model": state.model_name,
                "threshold": state.threshold, "model_sha256": state.model_sha256}

    @app.post("/predict", response_model=Prediction)
    def predict(features: Features, request: Request):
        state = request.app.state
        frame = pd.DataFrame([features.model_dump()], columns=MODEL_FEATURES)
        score = float(state.model.predict_proba(frame)[0, 1])
        return Prediction(score=score, threshold=state.threshold,
            decision="flag_for_review" if score >= state.threshold else "not_flagged",
            model=state.model_name, model_sha256=state.model_sha256)

    return app


app = create_app()
