"""
Model training and evaluation for the Telco Churn classifier.

Trains three candidate models (Logistic Regression baseline, Random Forest
primary, XGBoost alternative) using identical stratified train/val/test
splits and identical preprocessing, so results are directly comparable.
SMOTE oversampling is applied to the training fold only, after splitting,
to avoid leaking synthetic neighbors across the validation/test boundary.

Run as a script, this writes:
  - models/best_model.joblib       (fitted Pipeline: preprocessing + classifier)
  - models/metrics.json            (metrics for all three candidates)
  - models/feature_importance.json (top features from the selected model)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import run_pipeline
from features import build_preprocessing_pipeline, get_feature_target_split

RANDOM_STATE = 42
MODELS_DIR = Path("models")


def make_splits(X, y):
    """
    Stratified 70/15/15 train/validation/test split.

    Stratification on y preserves the ~73/27 churn ratio across all three
    splits, which matters here because a non-stratified split could, by
    chance, produce a test set with a meaningfully different base rate and
    make AUC-ROC comparisons across runs noisy.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y) -> dict:
    """Compute the metric set specified in the design document."""
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        "auc_roc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def train_logistic_regression(X_train, y_train, preprocessor):
    pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train, preprocessor):
    """
    Grid search over the hyperparameter ranges specified in the design
    document. SMOTE is applied to the training data only, after the
    preprocessor has been fit and used to transform X_train -- this keeps
    SMOTE's synthetic-neighbor generation working in already-encoded numeric
    space, which is required since SMOTE cannot operate on raw strings.
    """
    X_train_enc = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(base_rf, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    grid.fit(X_train_res, y_train_res)

    full_pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", grid.best_estimator_),
        ]
    )
    # preprocessor is already fit; mark pipeline steps as fit by refitting
    # the classifier on resampled data while reusing the fitted preprocessor
    full_pipe.named_steps["preprocess"] = preprocessor
    return full_pipe, grid.best_params_


def train_xgboost(X_train, y_train, preprocessor):
    X_train_enc = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_res, y_train_res)

    full_pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    full_pipe.named_steps["preprocess"] = preprocessor
    return full_pipe


def get_feature_importance(pipe, top_n=10) -> list:
    """
    Extract top-N feature importances from a fitted pipeline.

    Tree-based models (Random Forest, XGBoost) expose feature_importances_.
    Logistic Regression instead exposes coef_; we rank by absolute
    coefficient magnitude since sign indicates direction (churn-increasing
    vs. churn-decreasing) rather than importance.
    """
    try:
        clf = pipe.named_steps["clf"]
        preprocessor = pipe.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            order = np.argsort(importances)[::-1][:top_n]
            return [
                {"feature": str(feature_names[i]), "importance": float(importances[i])}
                for i in order
            ]
        elif hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
            order = np.argsort(np.abs(coefs))[::-1][:top_n]
            return [
                {
                    "feature": str(feature_names[i]),
                    "coefficient": float(coefs[i]),
                    "direction": (
                        "increases_churn_risk"
                        if coefs[i] > 0
                        else "decreases_churn_risk"
                    ),
                }
                for i in order
            ]
        else:
            return [{"error": "model exposes neither feature_importances_ nor coef_"}]
    except Exception as e:
        return [{"error": str(e)}]


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    print("Loading and preprocessing data...")
    df = run_pipeline()
    X, y = get_feature_target_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(
        f"Train churn rate: {y_train.mean():.4f}  Val: {y_val.mean():.4f}  Test: {y_test.mean():.4f}"
    )

    results = {}
    fitted_models = {}

    # --- Logistic Regression baseline ---
    t0 = time.time()
    lr_pipe = train_logistic_regression(
        X_train, y_train, build_preprocessing_pipeline()
    )
    results["logistic_regression"] = evaluate(lr_pipe, X_val, y_val)
    results["logistic_regression"]["train_seconds"] = round(time.time() - t0, 2)
    fitted_models["logistic_regression"] = lr_pipe
    print(f"LogReg  val AUC-ROC: {results['logistic_regression']['auc_roc']:.4f}")

    # --- Random Forest (grid search + SMOTE) ---
    t0 = time.time()
    rf_pipe, rf_best_params = train_random_forest(
        X_train, y_train, build_preprocessing_pipeline()
    )
    results["random_forest"] = evaluate(rf_pipe, X_val, y_val)
    results["random_forest"]["train_seconds"] = round(time.time() - t0, 2)
    results["random_forest"]["best_params"] = rf_best_params
    fitted_models["random_forest"] = rf_pipe
    print(
        f"RF      val AUC-ROC: {results['random_forest']['auc_roc']:.4f}  params={rf_best_params}"
    )

    # --- XGBoost (SMOTE + scale_pos_weight) ---
    t0 = time.time()
    xgb_pipe = train_xgboost(X_train, y_train, build_preprocessing_pipeline())
    results["xgboost"] = evaluate(xgb_pipe, X_val, y_val)
    results["xgboost"]["train_seconds"] = round(time.time() - t0, 2)
    fitted_models["xgboost"] = xgb_pipe
    print(f"XGBoost val AUC-ROC: {results['xgboost']['auc_roc']:.4f}")

    # --- Select best model by validation AUC-ROC ---
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    best_model = fitted_models[best_name]
    print(f"\nSelected model: {best_name}")

    # --- Final, held-out test set evaluation (only on the selected model) ---
    test_metrics = evaluate(best_model, X_test, y_test)
    results["selected_model"] = best_name
    results["test_set_metrics"] = test_metrics
    print(
        f"TEST SET  AUC-ROC: {test_metrics['auc_roc']:.4f}  F1: {test_metrics['f1']:.4f}  "
        f"Precision: {test_metrics['precision']:.4f}  Recall: {test_metrics['recall']:.4f}"
    )

    # --- Feature importance (if tree-based) ---
    feature_importance = get_feature_importance(best_model)

    # --- Persist artifacts ---
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(MODELS_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    print(f"\nArtifacts written to {MODELS_DIR}/")
    return results


if __name__ == "__main__":
    main()
