"""
SageMaker Processing job entry point for model evaluation.

Loads the trained XGBoost model artifact (model.tar.gz, unpacked by
SageMaker into /opt/ml/processing/model) and the test set CSV produced by
preprocess_job.py, computes the same metric set used in the local pipeline
(src/train.py's evaluate() function: AUC-ROC, F1, precision, recall,
confusion matrix), and writes evaluation.json in the
SageMaker-Pipelines-compatible format that build_pipeline.py's
ConditionStep reads via JsonGet.

The output schema (binary_classification_metrics.auc.value) follows the
convention used by SageMaker's own example pipelines so that the
ConditionGreaterThanOrEqualTo step in build_pipeline.py can read it
directly with a JsonGet property file reference.
"""

from __future__ import annotations

import json
import os
import tarfile

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def main():
    # SageMaker Training jobs package the model artifact as model.tar.gz
    model_tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
    if os.path.exists(model_tar_path):
        with tarfile.open(model_tar_path) as tar:
            tar.extractall(path=MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, "xgboost-model")
    booster = xgb.Booster()
    booster.load_model(model_path)

    test_csv_path = os.path.join(TEST_DIR, "test.csv")
    test_df = pd.read_csv(test_csv_path, header=None)

    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    d_test = xgb.DMatrix(X_test)
    y_proba = booster.predict(d_test)
    y_pred = (y_proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_proba))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(
        f"Test AUC-ROC: {auc:.4f}  F1: {f1:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}"
    )

    # Schema matches SageMaker's own example pipelines' evaluation report
    # convention, which is what ConditionStep / JsonGet expects to parse.
    report = {
        "binary_classification_metrics": {
            "auc": {"value": auc, "standard_deviation": "NaN"},
            "f1": {"value": f1, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "confusion_matrix": {"value": cm},
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation report written to {OUTPUT_DIR}/evaluation.json")


if __name__ == "__main__":
    main()
