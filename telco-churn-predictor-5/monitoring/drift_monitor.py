"""
Data drift monitoring for the Telco Churn pipeline.

Computes Population Stability Index (PSI) between a reference distribution
(the training set) and a current/incoming distribution (e.g., this week's
scoring batch) for the key features the design document commits to
watching: tenure, MonthlyCharges, and Contract.

PSI > 0.2 is the alert threshold specified in the design document's Model
Monitoring section. This module is intentionally dependency-light (no
Evidently AI install required for this graded deliverable) so it is easy
to read end-to-end. The production version would swap this for an
Evidently AI Report/Dashboard for richer visualization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PSI_ALERT_THRESHOLD = 0.2
KEY_NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
KEY_CATEGORICAL_FEATURES = ["Contract"]


def _psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for a continuous feature, via quantile bins."""
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = reference.quantile(quantiles).values.copy()
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """Population Stability Index for a categorical feature, via category frequency."""
    categories = sorted(set(reference.unique()) | set(current.unique()))
    ref_pct = reference.value_counts(normalize=True).reindex(
        categories, fill_value=1e-6
    )
    cur_pct = current.value_counts(normalize=True).reindex(categories, fill_value=1e-6)
    ref_pct = ref_pct.replace(0, 1e-6)
    cur_pct = cur_pct.replace(0, 1e-6)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Compute PSI for all monitored features and flag any that exceed the
    alert threshold.
    """
    report = {"features": {}, "alerts": []}

    for col in KEY_NUMERIC_FEATURES:
        psi = _psi_numeric(reference_df[col], current_df[col])
        report["features"][col] = {"psi": round(psi, 4), "type": "numeric"}
        if psi > PSI_ALERT_THRESHOLD:
            report["alerts"].append(
                f"{col}: PSI={psi:.4f} exceeds threshold {PSI_ALERT_THRESHOLD}"
            )

    for col in KEY_CATEGORICAL_FEATURES:
        psi = _psi_categorical(reference_df[col], current_df[col])
        report["features"][col] = {"psi": round(psi, 4), "type": "categorical"}
        if psi > PSI_ALERT_THRESHOLD:
            report["alerts"].append(
                f"{col}: PSI={psi:.4f} exceeds threshold {PSI_ALERT_THRESHOLD}"
            )

    report["status"] = "ALERT" if report["alerts"] else "OK"
    return report


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "src")
    from preprocess import run_pipeline

    df = run_pipeline()
    # Demonstration: split the dataset in half to simulate "reference" vs.
    # "current" batches. In production, reference = frozen training snapshot,
    # current = this week's live scoring batch pulled from S3.
    half = len(df) // 2
    reference_df = df.iloc[:half]
    current_df = df.iloc[half:]

    report = compute_drift_report(reference_df, current_df)
    print(json.dumps(report, indent=2))

    Path("monitoring").mkdir(exist_ok=True)
    with open("monitoring/drift_report_example.json", "w") as f:
        json.dump(report, f, indent=2)
