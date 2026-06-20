"""
SageMaker Processing job entry point for the fairness audit.

Loads the trained XGBoost model artifact and the test set produced by
preprocess_job.py, generates predictions, and computes the same Fairlearn
metrics as the local pipeline (src/fairness_audit.py): demographic parity
difference and equalized odds difference across the gender and
SeniorCitizen subgroups.

Sensitive feature values are read from test_sensitive_features.csv (also
written by preprocess_job.py) rather than from the encoded test.csv, since
the encoded feature matrix no longer has these columns in a fixed, named
position. The two files are aligned by row order.

Writes fairness_audit.json in the same schema the local pipeline produces,
so both implementations are directly comparable.
"""

from __future__ import annotations

import json
import os
import tarfile

import pandas as pd
import xgboost as xgb
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import precision_score, recall_score

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/fairness"

SENSITIVE_COLUMNS = ["gender", "SeniorCitizen"]


def main():
    model_tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
    if os.path.exists(model_tar_path):
        with tarfile.open(model_tar_path) as tar:
            tar.extractall(path=MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, "xgboost-model")
    booster = xgb.Booster()
    booster.load_model(model_path)

    test_df = pd.read_csv(os.path.join(TEST_DIR, "test.csv"), header=None)
    sensitive_df = pd.read_csv(os.path.join(TEST_DIR, "test_sensitive_features.csv"))

    if len(test_df) != len(sensitive_df):
        raise ValueError(
            f"Row count mismatch between test.csv ({len(test_df)}) and "
            f"test_sensitive_features.csv ({len(sensitive_df)}) -- these "
            f"must stay aligned. Check preprocess_job.py."
        )

    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    d_test = xgb.DMatrix(X_test)
    y_proba = booster.predict(d_test)
    y_pred = (y_proba >= 0.5).astype(int)

    results = {}
    for col in SENSITIVE_COLUMNS:
        sensitive_features = sensitive_df[col]

        dp_diff = demographic_parity_difference(
            y_test, y_pred, sensitive_features=sensitive_features
        )
        eo_diff = equalized_odds_difference(
            y_test, y_pred, sensitive_features=sensitive_features
        )

        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "recall": recall_score,
                "precision": precision_score,
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        results[col] = {
            "demographic_parity_difference": float(dp_diff),
            "equalized_odds_difference": float(eo_diff),
            "by_group": mf.by_group.to_dict(),
        }

        print(f"=== {col} ===")
        print(f"Demographic parity difference: {dp_diff:.4f}")
        print(f"Equalized odds difference:     {eo_diff:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "fairness_audit.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Fairness audit written to {OUTPUT_DIR}/fairness_audit.json")


if __name__ == "__main__":
    main()
