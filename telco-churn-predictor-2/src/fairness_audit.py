"""
Fairness audit for the selected churn model.

Computes demographic parity difference and equalized odds difference across
the gender and SeniorCitizen subgroups, as committed to in the design
document's Security Checklist section. Run after train.py has produced
models/best_model.joblib.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import recall_score, precision_score

from preprocess import run_pipeline
from features import get_feature_target_split
from train import make_splits

MODELS_DIR = Path("models")


def run_fairness_audit():
    df = run_pipeline()
    X, y = get_feature_target_split(df)
    # Need the same split as training; also carry sensitive features through
    df_indexed = df.reset_index(drop=True)
    X_indexed = X.reset_index(drop=True)
    y_indexed = y.reset_index(drop=True)

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X_indexed, y_indexed)

    model = joblib.load(MODELS_DIR / "best_model.joblib")
    y_pred = model.predict(X_test)

    results = {}
    for sensitive_col in ["gender", "SeniorCitizen"]:
        sensitive_features = df_indexed.loc[X_test.index, sensitive_col]

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

        results[sensitive_col] = {
            "demographic_parity_difference": float(dp_diff),
            "equalized_odds_difference": float(eo_diff),
            "by_group": mf.by_group.to_dict(),
        }

    with open(MODELS_DIR / "fairness_audit.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_fairness_audit()
