# Telco Customer Churn Prediction

Binary classification system predicting customer churn risk for a telecom
provider, built for AAI-540. Companion repository to the ML System Design
Document.

## Results Summary

| Model | Validation AUC-ROC | Test AUC-ROC | Test F1 | Test Recall |
|---|---|---|---|---|
| Logistic Regression (selected) | 0.8426 | **0.8549** | 0.6298 | 0.8172 |
| Random Forest (GridSearchCV + SMOTE) | 0.8290 | -- | -- | -- |
| XGBoost (SMOTE + scale_pos_weight) | 0.8224 | -- | -- | -- |

Logistic Regression was selected as the production model based on highest
validation AUC-ROC. Full metrics: `models/metrics.json`.

## Repository Structure

```
telco-churn-predictor/
├── data/
│   ├── raw/telco_churn.csv          # Raw Kaggle/IBM source data
│   └── processed/                    # Cleaned, feature-engineered output
├── src/
│   ├── preprocess.py                 # Cleaning + derived feature engineering
│   ├── features.py                   # sklearn ColumnTransformer pipeline
│   ├── train.py                      # Training, tuning, model selection
│   ├── fairness_audit.py             # Fairlearn bias audit
│   └── serve.py                      # FastAPI batch inference service
├── monitoring/
│   └── drift_monitor.py              # PSI-based data drift detection
├── sagemaker/                         # SageMaker-native pipeline (alternative
│                                       # to the EC2/FastAPI deployment below) —
│                                       # see sagemaker/README.md
├── tests/
│   ├── test_preprocess.py
│   ├── test_model_performance.py     # CI/CD evaluation gate
│   └── test_drift_monitor.py
├── models/                           # Trained artifacts (gitignored in practice)
├── notebooks/
│   ├── telco_churn_pipeline.ipynb    # Full pipeline, executed end-to-end (EDA → preprocessing → features → training → fairness audit)
│   ├── telco_churn_pipeline.py       # Same notebook in jupytext percent format (for clean diffs/editing)
│   └── figures/                      # Saved EDA charts
└── .github/workflows/ml-pipeline.yml # CI/CD pipeline
```

## Running the Pipeline

The full pipeline can be run either as the modular scripts in `src/`, or as
a single executed notebook at `notebooks/telco_churn_pipeline.ipynb` —
both paths call the same underlying functions, so results are identical.

```bash
pip install -r requirements.txt

# 1. Preprocess
PYTHONPATH=src python3 src/preprocess.py

# 2. Train, tune, and select the best model
PYTHONPATH=src python3 src/train.py

# 3. Run the fairness audit
PYTHONPATH=src python3 src/fairness_audit.py

# 4. Run tests
PYTHONPATH=src:monitoring pytest tests/ -v

# 5. Serve locally
PYTHONPATH=src uvicorn serve:app --app-dir src --reload

# Or, run everything interactively in one notebook:
jupyter notebook notebooks/telco_churn_pipeline.ipynb
```

## SageMaker Alternative

The design document describes a self-managed EC2 + FastAPI deployment. A
parallel implementation using AWS SageMaker's built-in XGBoost container,
SageMaker Pipelines, and Batch Transform is in `sagemaker/` — see
`sagemaker/README.md` for what's included, what's been validated, and how
to run it against a real AWS account.

## Key Findings

- **Contract type dominates**: month-to-month churn rate is 42.6% vs. 2.8%
  for two-year contracts.
- **Tenure matters**: customers in their first 12 months churn at 47.4%,
  dropping to 9.5% after 49+ months.
- **Payment method signal**: electronic check users churn at 45.1% vs.
  15.2% for automatic credit card payments.
- **Fairness finding**: gender shows negligible disparity (demographic
  parity difference 0.0016), but SeniorCitizen shows a substantial gap
  (0.314) -- the model flags senior citizens as high-risk nearly twice as
  often as non-seniors. See `models/fairness_audit.json` and the design
  document's Security Checklist for discussion.
