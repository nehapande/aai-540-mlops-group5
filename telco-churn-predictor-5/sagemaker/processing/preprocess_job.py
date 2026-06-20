"""
SageMaker Processing job entry point for the Telco Churn dataset.

Runs inside a SageMaker Processing container (e.g. the sklearn-processing
built-in image). Reads raw CSV from the input path SageMaker mounts at
/opt/ml/processing/input, applies the same cleaning + feature engineering
logic used in the local pipeline (src/preprocess.py, src/features.py), and
writes train/validation/test CSVs to the output paths SageMaker expects
for a downstream XGBoost Training job (label column first, no header,
no index -- this matches the SageMaker built-in XGBoost container's
expected input format for CSV).

Invoked by sagemaker/pipeline/build_pipeline.py as a ProcessingStep.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

# ---- Mirrors src/preprocess.py exactly; duplicated here (not imported) ----
# because a SageMaker Processing container only has this script and its
# requirements.txt on board -- it does not have the rest of the repo's
# src/ package available unless explicitly packaged as a dependency. For
# a real production system, src/ would be packaged into a private PyPI
# index or container image and imported normally; duplicating the logic
# here keeps this script runnable as a single self-contained file, which
# is the common pattern for small SageMaker Processing scripts.

BINARY_YESNO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

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

BINARY_PASSTHROUGH_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "has_support_services",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df = df.drop_duplicates()
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype(int)
    for col in BINARY_YESNO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12", "13-24", "25-48", "49-72"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    df["has_support_services"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def _count_services(row):
        count = 0
        for col in service_cols:
            if row[col] in ("Yes", 1, "DSL", "Fiber optic"):
                count += 1
        return count

    df["num_services"] = df.apply(_count_services, axis=1)
    return df


def build_preprocessor() -> ColumnTransformer:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    input_file = os.path.join(args.input_path, "telco_churn.csv")
    print(f"Reading raw data from {input_file}")
    raw_df = pd.read_csv(input_file)
    print(f"Raw shape: {raw_df.shape}")

    cleaned = clean(raw_df)
    featured = add_derived_features(cleaned)
    print(f"Cleaned + featured shape: {featured.shape}")

    feature_cols = BINARY_PASSTHROUGH_COLS + CATEGORICAL_COLS + NUMERIC_COLS
    X = featured[feature_cols]
    y = featured["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - args.train_ratio), stratify=y, random_state=RANDOM_STATE
    )
    val_size = args.val_ratio / (1 - args.train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_size),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    # SageMaker's built-in XGBoost container expects CSV with the label
    # in the FIRST column, no header row, no index column.
    def _to_label_first_csv(X_enc, y, path):
        X_dense = X_enc.toarray() if hasattr(X_enc, "toarray") else np.asarray(X_enc)
        out = pd.DataFrame(X_dense)
        out.insert(0, "label", y.reset_index(drop=True))
        out.to_csv(path, header=False, index=False)
        print(f"Wrote {len(out)} rows -> {path}")

    os.makedirs(os.path.join(args.output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "validation"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "test"), exist_ok=True)

    _to_label_first_csv(
        X_train_enc, y_train, os.path.join(args.output_path, "train", "train.csv")
    )
    _to_label_first_csv(
        X_val_enc, y_val, os.path.join(args.output_path, "validation", "validation.csv")
    )
    _to_label_first_csv(
        X_test_enc, y_test, os.path.join(args.output_path, "test", "test.csv")
    )

    # The encoded test.csv above no longer has human-readable gender/
    # SeniorCitizen columns in a fixed, named position -- it's a raw
    # numeric matrix shaped by the ColumnTransformer's internal column
    # ordering. Rather than have the downstream fairness audit step rely on
    # guessing which integer column position corresponds to which sensitive
    # attribute, we export those two raw columns separately here, in the
    # exact same row order as test.csv, so fairness_audit_job.py can join
    # on row position without any fragile assumptions about encoder internals.
    sensitive_test = featured.loc[
        X_test.index, ["gender", "SeniorCitizen"]
    ].reset_index(drop=True)
    sensitive_test.to_csv(
        os.path.join(args.output_path, "test", "test_sensitive_features.csv"),
        index=False,
    )
    print(
        f"Wrote {len(sensitive_test)} rows -> "
        f"{os.path.join(args.output_path, 'test', 'test_sensitive_features.csv')}"
    )

    print("Processing complete.")


if __name__ == "__main__":
    main()
