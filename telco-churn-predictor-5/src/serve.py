"""
Batch inference service for the Telco Churn model.

Exposes a /predict endpoint that accepts a list of customer records (as the
raw, uncleaned schema matching telco_churn.csv minus customerID/Churn) and
returns churn probability scores. Intended to be invoked by the nightly
batch job described in the design document's Model Deployment section --
not a true low-latency real-time endpoint, but a thin HTTP wrapper around
the same Pipeline object used in training, so behavior is guaranteed
identical between training-time and inference-time preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from preprocess import clean, add_derived_features

MODEL_PATH = Path("models/best_model.joblib")

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Batch scoring endpoint for customer churn risk",
    version="1.0.0",
)

_model = None  # lazy-loaded singleton


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model artifact not found at {MODEL_PATH}. Run train.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class CustomerRecord(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  # kept as str to match raw schema; coerced downstream


class PredictRequest(BaseModel):
    customers: list[CustomerRecord]


class PredictionResult(BaseModel):
    churn_probability: float
    churn_risk_flag: bool


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    model_version: str = "1.0.0"


@app.get("/health")
def health():
    """Liveness probe used by the CI/CD smoke test step."""
    try:
        get_model()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Score a batch of customer records.

    Records are run through the exact same clean() + add_derived_features()
    functions used at training time before being passed into the fitted
    Pipeline, ensuring preprocessing parity between training and serving.
    """
    model = get_model()

    raw_df = pd.DataFrame([r.model_dump() for r in request.customers])
    # clean() expects a Churn column to encode; inject a placeholder since
    # we are not predicting on labeled data, then drop it before scoring.
    raw_df["Churn"] = "No"
    raw_df["customerID"] = "placeholder"

    cleaned = clean(raw_df)
    featured = add_derived_features(cleaned)

    from features import FEATURE_COLS

    X = featured[FEATURE_COLS]

    probabilities = model.predict_proba(X)[:, 1]

    results = [
        PredictionResult(churn_probability=float(p), churn_risk_flag=bool(p >= 0.5))
        for p in probabilities
    ]
    return PredictResponse(predictions=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
