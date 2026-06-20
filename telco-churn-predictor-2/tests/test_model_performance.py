"""
Model performance regression gate.

Asserts the trained model's test-set AUC-ROC clears the 0.80 threshold
committed to in the design document's Goals section. This is the CI/CD
evaluation gate -- a failing assertion here should block promotion to
production in the GitHub Actions workflow.
"""

import json
from pathlib import Path

METRICS_PATH = Path("models/metrics.json")
MIN_AUC_ROC = 0.80
MAX_REGRESSION_FROM_PRIOR = 0.02  # max allowed AUC-ROC drop vs. last production model

PRIOR_MODEL_AUC_PATH = Path("models/prior_production_auc.json")


def test_model_meets_minimum_auc_threshold():
    assert METRICS_PATH.exists(), "metrics.json not found -- run train.py first"
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    test_auc = metrics["test_set_metrics"]["auc_roc"]
    assert (
        test_auc >= MIN_AUC_ROC
    ), f"Test AUC-ROC {test_auc:.4f} is below the required minimum {MIN_AUC_ROC}"


def test_model_does_not_regress_versus_prior_production_model():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    current_auc = metrics["test_set_metrics"]["auc_roc"]

    if not PRIOR_MODEL_AUC_PATH.exists():
        # First deployment -- nothing to compare against yet.
        return

    with open(PRIOR_MODEL_AUC_PATH) as f:
        prior_auc = json.load(f)["auc_roc"]

    degradation = prior_auc - current_auc
    assert degradation <= MAX_REGRESSION_FROM_PRIOR, (
        f"New model AUC-ROC ({current_auc:.4f}) regresses by {degradation:.4f} "
        f"versus prior production model ({prior_auc:.4f}), exceeding the "
        f"{MAX_REGRESSION_FROM_PRIOR} allowed threshold"
    )


def test_recall_on_churn_class_is_reasonable():
    """
    Churn prediction is recall-sensitive: missing an actual churner (false
    negative) costs more than a false positive (an unnecessary retention
    offer). We gate on recall staying above 0.65 so a future retrain can't
    silently trade away recall for precision without a visible test failure.
    """
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    recall = metrics["test_set_metrics"]["recall"]
    assert recall >= 0.65, f"Churn-class recall {recall:.4f} fell below the 0.65 floor"


if __name__ == "__main__":
    test_model_meets_minimum_auc_threshold()
    test_model_does_not_regress_versus_prior_production_model()
    test_recall_on_churn_class_is_reasonable()
    print("All model performance gates passed.")
