"""
Data preprocessing for the Telco Customer Churn dataset.

Responsibilities:
- Load raw CSV
- Fix TotalCharges type coercion (string -> float, with blanks for zero-tenure customers)
- Drop non-predictive identifier columns
- Binary-encode Yes/No and gender fields
- Engineer derived features (tenure_group, has_support_services, num_services)
- Return a clean, model-ready DataFrame

This module is intentionally free of train/test splitting logic -- that lives
in train.py so preprocessing can be unit tested deterministically on the full
dataset without leakage concerns.
"""

from __future__ import annotations

import pandas as pd

RAW_PATH_DEFAULT = "data/raw/telco_churn.csv"

# Columns that are Yes/No and map cleanly to 1/0
BINARY_YESNO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# Multi-category columns that will be one-hot encoded downstream in the
# scikit-learn Pipeline (kept as strings here; encoding happens in features.py)
MULTI_CATEGORY_COLS = [
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
]

CONTINUOUS_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path: str = RAW_PATH_DEFAULT) -> pd.DataFrame:
    """Load the raw Telco churn CSV from disk."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply core cleaning steps to the raw dataframe.

    Steps (in order, matches design document Data Engineering section):
      1. Cast TotalCharges to numeric; blanks correspond to customers with
         tenure == 0 (brand-new customers who have not been billed yet), so
         we impute these as 0.0 rather than dropping the rows -- dropping
         would discard legitimate zero-tenure customers, which are exactly
         the population most useful for early-churn signal.
      2. Drop customerID (unique identifier, non-predictive, would leak
         row-identity into a tree-based model if accidentally left in).
      3. Drop exact-duplicate rows (defensive; the canonical dataset has 0,
         but pipeline should not assume that holds for future ingests).
      4. Encode gender and Yes/No binary columns to 0/1 integers.
    """
    df = df.copy()

    # Step 1: TotalCharges type coercion
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing_total_charges = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Step 2: drop identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Step 3: drop duplicates defensively
    df = df.drop_duplicates()

    # Step 4: binary encodings
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype(int)
    for col in BINARY_YESNO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)

    df.attrs["n_missing_total_charges_imputed"] = int(n_missing_total_charges)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer derived features described in the design document's
    Feature Engineering section.
    """
    df = df.copy()

    # tenure_group: bucket tenure (months) into 4 lifecycle stages
    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12", "13-24", "25-48", "49-72"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # has_support_services: 1 if customer has OnlineSecurity OR TechSupport
    df["has_support_services"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    # num_services: count of add-on services subscribed (0-8)
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
            val = row[col]
            if val in ("Yes", 1, "DSL", "Fiber optic"):
                count += 1
        return count

    df["num_services"] = df.apply(_count_services, axis=1)
    return df


def run_pipeline(raw_path: str = RAW_PATH_DEFAULT) -> pd.DataFrame:
    """Full preprocessing entrypoint: load -> clean -> derive features."""
    df = load_raw(raw_path)
    df = clean(df)
    df = add_derived_features(df)
    return df


if __name__ == "__main__":
    out = run_pipeline()
    out.to_csv("data/processed/telco_churn_clean.csv", index=False)
    print(f"Processed {len(out)} rows -> data/processed/telco_churn_clean.csv")
    print(
        f"Imputed {out.attrs.get('n_missing_total_charges_imputed', 0)} missing TotalCharges values"
    )
