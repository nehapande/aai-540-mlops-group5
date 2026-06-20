"""
Feature engineering pipeline for the Telco Churn model.

Wraps one-hot encoding and scaling in a scikit-learn ColumnTransformer so
that the exact same transformation logic is used at training time and at
inference time -- this is the leakage-prevention mechanism described in the
design document's Feature Engineering section. Fitting happens only on the
training split; transform is then applied to validation/test/inference data.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET_COL = "Churn"

# Already binary 0/1 after preprocess.clean() -- pass through unchanged
BINARY_PASSTHROUGH_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "has_support_services",
]

# Multi-category string columns requiring one-hot encoding
CATEGORICAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "tenure_group",
]

# Continuous columns requiring scaling
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services"]

FEATURE_COLS = BINARY_PASSTHROUGH_COLS + CATEGORICAL_COLS + NUMERIC_COLS


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Construct the ColumnTransformer used inside the full model Pipeline.

    Design rationale:
    - OneHotEncoder(handle_unknown="ignore") so that a category unseen at
      training time (e.g., a new PaymentMethod added in production) does not
      crash inference -- it simply produces an all-zero indicator row rather
      than raising, which is the safer failure mode for a batch job.
    - StandardScaler on continuous features only; one-hot and passthrough
      binary columns are already on a 0/1 scale and don't benefit from
      scaling (and scaling them would distort tree-based feature importances
      less, but actively hurts the Logistic Regression baseline).
    """
    return ColumnTransformer(
        transformers=[
            ("binary", "passthrough", BINARY_PASSTHROUGH_COLS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_COLS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )


def get_feature_target_split(df):
    """Split a cleaned, feature-engineered dataframe into X, y."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y
