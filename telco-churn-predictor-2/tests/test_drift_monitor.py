"""
Unit tests for monitoring/drift_monitor.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "monitoring"))

from drift_monitor import compute_drift_report, _psi_numeric, _psi_categorical


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(50, 10, 1000))
    cur = ref.copy()
    psi = _psi_numeric(ref, cur)
    assert psi < 0.01


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(50, 10, 1000))
    cur = pd.Series(rng.normal(90, 10, 1000))  # large mean shift
    psi = _psi_numeric(ref, cur)
    assert psi > 0.2


def test_categorical_psi_near_zero_for_same_distribution():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cur = pd.Series(["A"] * 500 + ["B"] * 500)
    psi = _psi_categorical(ref, cur)
    assert psi < 0.01


def test_drift_report_flags_alert_on_shifted_data():
    rng = np.random.default_rng(42)
    reference_df = pd.DataFrame(
        {
            "tenure": rng.normal(30, 15, 500),
            "MonthlyCharges": rng.normal(60, 20, 500),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], 500),
        }
    )
    current_df = pd.DataFrame(
        {
            "tenure": rng.normal(80, 15, 500),  # shifted hard
            "MonthlyCharges": rng.normal(60, 20, 500),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], 500),
        }
    )
    report = compute_drift_report(reference_df, current_df)
    assert report["status"] == "ALERT"
    assert any("tenure" in alert for alert in report["alerts"])
