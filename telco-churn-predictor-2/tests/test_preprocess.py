"""
Unit tests for src/preprocess.py and src/features.py.

Run with: PYTHONPATH=src pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import clean, add_derived_features, run_pipeline
from features import (
    build_preprocessing_pipeline,
    get_feature_target_split,
    FEATURE_COLS,
)


@pytest.fixture
def raw_sample():
    return pd.DataFrame(
        [
            {
                "customerID": "0001-TEST",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.0,
                "TotalCharges": " ",
                "Churn": "No",
            },
            {
                "customerID": "0002-TEST",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 90.5,
                "TotalCharges": "2172.0",
                "Churn": "Yes",
            },
        ]
    )


def test_clean_coerces_blank_total_charges_to_zero(raw_sample):
    out = clean(raw_sample)
    # row 0 had a blank TotalCharges and tenure == 0
    assert out.loc[0, "TotalCharges"] == 0.0


def test_clean_drops_customer_id(raw_sample):
    out = clean(raw_sample)
    assert "customerID" not in out.columns


def test_clean_binary_encodes_churn(raw_sample):
    out = clean(raw_sample)
    assert set(out["Churn"].unique()).issubset({0, 1})
    assert out.loc[0, "Churn"] == 0
    assert out.loc[1, "Churn"] == 1


def test_clean_binary_encodes_gender(raw_sample):
    out = clean(raw_sample)
    assert out.loc[0, "gender"] == 0  # Female -> 0
    assert out.loc[1, "gender"] == 1  # Male -> 1


def test_derived_tenure_group_buckets_correctly(raw_sample):
    cleaned = clean(raw_sample)
    out = add_derived_features(cleaned)
    assert out.loc[0, "tenure_group"] == "0-12"
    assert out.loc[1, "tenure_group"] == "13-24"


def test_derived_has_support_services(raw_sample):
    cleaned = clean(raw_sample)
    out = add_derived_features(cleaned)
    # row 0 has OnlineSecurity=Yes and TechSupport=Yes -> 1
    assert out.loc[0, "has_support_services"] == 1
    # row 1 has neither -> 0
    assert out.loc[1, "has_support_services"] == 0


def test_no_nulls_after_full_pipeline(raw_sample, tmp_path):
    csv_path = tmp_path / "sample.csv"
    raw_sample.to_csv(csv_path, index=False)
    out = run_pipeline(str(csv_path))
    assert out.isnull().sum().sum() == 0


def test_feature_target_split_shapes(raw_sample, tmp_path):
    csv_path = tmp_path / "sample.csv"
    raw_sample.to_csv(csv_path, index=False)
    df = run_pipeline(str(csv_path))
    X, y = get_feature_target_split(df)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y) == 2


def test_preprocessing_pipeline_fits_and_transforms(raw_sample, tmp_path):
    csv_path = tmp_path / "sample.csv"
    raw_sample.to_csv(csv_path, index=False)
    df = run_pipeline(str(csv_path))
    X, y = get_feature_target_split(df)
    pre = build_preprocessing_pipeline()
    X_transformed = pre.fit_transform(X)
    assert X_transformed.shape[0] == 2


def test_one_hot_handles_unseen_category_at_inference(raw_sample, tmp_path):
    """
    Regression test for the handle_unknown='ignore' design decision in
    features.py -- an unseen PaymentMethod at inference time should not
    raise, and should simply produce all-zero indicators for that column.
    """
    csv_path = tmp_path / "sample.csv"
    raw_sample.to_csv(csv_path, index=False)
    df = run_pipeline(str(csv_path))
    X, y = get_feature_target_split(df)
    pre = build_preprocessing_pipeline()
    pre.fit(X, y)

    X_new = X.iloc[[0]].copy()
    X_new["PaymentMethod"] = "Crypto (new method)"  # unseen category
    # Should not raise
    transformed = pre.transform(X_new)
    assert transformed.shape[0] == 1
