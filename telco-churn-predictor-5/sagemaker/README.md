# SageMaker Pipeline — Telco Churn

**Want a single notebook that demonstrates the whole project end-to-end?**
Use [`notebooks/00_end_to_end_telco_churn.ipynb`](notebooks/00_end_to_end_telco_churn.ipynb).
It's a direct, linear, run-top-to-bottom notebook — data prep → upload to
S3 → XGBoost training → batch transform (with input/output filtering) →
fairness audit → model registration → model card → optional real-time
endpoint → cleanup — using the SageMaker SDK directly (no Pipelines DAG
abstraction). This is the one to use for a live demo or course submission
screenshot sequence.

The `pipeline/build_pipeline.py` + `notebooks/01_register_pipeline.ipynb` /
`02_execute_pipeline.ipynb` approach described below is a different,
more "managed/automated" architecture (a reusable SageMaker Pipeline DAG
with a scheduled retraining trigger) — useful if you want the conditional
AUC-ROC gate enforced automatically on every future run, rather than
something you click through once for a demo. Both reach the same project
goals; pick whichever fits what you're presenting.

**For a step-by-step setup and live-demo guide, see [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md).**
This README documents what's in this directory and what's been validated;
the runbook is the operational "how do I actually run and present this"
companion.

This directory contains a SageMaker-native version of the churn pipeline,
using the **built-in XGBoost algorithm container** rather than a custom
training script. It is a parallel implementation to `src/` and
`.github/workflows/ml-pipeline.yml` — both reach the same goals stated in
the design document, via different infrastructure.

All code here was validated locally (syntax, imports, and full SageMaker
Pipeline graph construction/property-reference resolution) but **was not
executed against AWS**, since that requires real credentials and incurs
real cost. Validation steps taken and their results are documented below
so you know exactly what has and has not been confirmed.

## What's included

```
sagemaker/
├── notebooks/
│   ├── 01_register_pipeline.ipynb   # Builds + upserts the pipeline definition
│   └── 02_execute_pipeline.ipynb    # Starts an execution, polls it, pulls results from S3
├── processing/
│   ├── preprocess_job.py       # Cleans raw data, engineers features, splits
│   │                           # train/val/test, writes label-first CSVs in
│   │                           # the format the built-in XGBoost container
│   │                           # expects, plus test_sensitive_features.csv
│   │                           # for the fairness audit step
│   ├── evaluate_job.py         # Loads the trained model + test set, computes
│   │                           # AUC-ROC/F1/precision/recall, writes evaluation.json
│   └── fairness_audit_job.py   # Loads the trained model + test set, computes
│                                # Fairlearn demographic parity / equalized odds
│                                # across gender and SeniorCitizen
├── training/
│   └── launch_training_job.py   # Standalone script: launches a single
│                                 # XGBoost Training job (no Pipeline needed)
└── pipeline/
    └── build_pipeline.py  # Full SageMaker Pipeline: Process -> Train ->
                            # Evaluate + Fairness Audit (parallel) ->
                            # [if AUC>=0.80] -> Register + Batch Transform
```

**Easiest way to run this end-to-end: the two notebooks in `notebooks/`.**
Both default to `DRY_RUN = True`, which exercises the entire logical flow
— including actually constructing the full Pipeline object — without
making any AWS calls that need real credentials. Both were executed in
this mode with zero errors (see validation table below). Set
`DRY_RUN = False` and fill in your bucket/role to run for real.

## What was validated locally (and how)

| Check | Method | Result |
|---|---|---|
| `preprocess_job.py` runs correctly | Ran locally against simulated `/opt/ml/processing/{input,output}` paths | Produced 4,914 / 1,053 / 1,054 row train/val/test CSVs, matching `src/train.py`'s splits exactly, plus a row-aligned `test_sensitive_features.csv` |
| XGBoost container image resolves | `sagemaker.image_uris.retrieve(framework="xgboost", region="us-east-1", version="1.7-1")` | Resolved to `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1`, a real AWS-managed image |
| `launch_training_job.py` constructs valid SDK objects | Instantiated `TrainingInput`/`Estimator` without calling `.fit()` | Valid SageMaker API objects, correct `scale_pos_weight` (2.7793) matching the local pipeline's class ratio |
| `evaluate_job.py` + `fairness_audit_job.py` run correctly against a real artifact | Trained a real XGBoost `Booster` locally on the actual processed train split, packaged it as `model.tar.gz` exactly as SageMaker would, ran both scripts against it outside any AWS container | Both scripts ran successfully end-to-end; produced real, sane metrics (0.8498 test AUC-ROC) and a fairness finding consistent with the local pipeline's Logistic Regression results |
| `build_pipeline.py` constructs a valid Pipeline | Built the full `Pipeline` object end-to-end with a real `PipelineSession`, including resolving every step's property references | Found and fixed two real bugs (see below) before reaching a clean build with 5 top-level steps |
| Pipeline definition generation | Called `pipeline.definition()` | Failed only on `NoCredentialsError` (expected — this step uploads scripts to S3) — no structural errors |
| `01_register_pipeline.ipynb` / `02_execute_pipeline.ipynb` run end-to-end | Executed both notebooks in full with `jupyter nbconvert --execute` in `DRY_RUN=True` mode | Both completed with **zero errors** across all 40 combined cells; the registration notebook's real (non-simulated) cell re-confirmed the 5-step Pipeline construction and printed every pipeline parameter's actual default value |
| `00_end_to_end_telco_churn.ipynb`'s Data Preparation section | Extracted and ran every pure-local (non-AWS) cell standalone against the real dataset | All cells succeeded with zero errors; produced the exact same numbers as the rest of this project (22 duplicate rows, 7,021 cleaned rows, 0.333/0.171 has_support_services churn rates). Found and fixed one real bug in the process: a duplicate-row check that included `customerID` always returns 0 (since the ID is unique per row) — fixed to check on feature columns only, which correctly surfaces the real 22 duplicates |

**Bugs found and fixed during this validation** (worth knowing about if you
modify this code):
1. `ModelMetrics`' S3 URI was built with Python string `+` concatenation on
   a pipeline property reference, which SageMaker Pipelines doesn't support
   (these are lazy graph nodes, not strings) — fixed with `sagemaker.workflow.functions.Join`.
2. The Batch Transform step initially referenced the training step's S3
   model artifact path as a `model_name`, which `Transformer` cannot use
   directly — fixed by adding an explicit `CreateModelStep` that produces a
   real deployable `Model` resource for the transformer to reference.

## What was NOT validated (requires your AWS account)

- Whether `preprocess_job.py` actually runs successfully **inside** the
  SageMaker-managed SKLearnProcessor container (it was only tested as a
  plain local Python script)
- Whether a SageMaker-managed Training job's resulting `model.tar.gz` has
  the exact same internal layout as the one built locally for this
  validation — `evaluate_job.py` and `fairness_audit_job.py` were confirmed
  to work against a *locally trained* XGBoost booster packaged the same way,
  but the actual container image's XGBoost version was not confirmed to match
  (see the file-format warning noted above)
- IAM permissions on your execution role
- Actual cost/runtime

## How to run this for real

1. **Upload the raw CSV to S3:**
   ```bash
   aws s3 cp data/raw/telco_churn.csv s3://YOUR_BUCKET/telco-churn/raw/telco_churn.csv
   ```

2. **Create or identify a SageMaker execution role** with `AmazonSageMakerFullAccess`
   and S3 read/write on your bucket. Use its ARN below.

3. **Quick test — single training job, no pipeline:**
   ```bash
   pip install sagemaker boto3
   python sagemaker/training/launch_training_job.py \
       --bucket YOUR_BUCKET \
       --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
   ```
   This expects `train.csv`/`validation.csv` to already exist under
   `s3://YOUR_BUCKET/telco-churn/processed/{train,validation}/` — run the
   processing script first (either locally, uploading its output, or as a
   real Processing job) to produce them.

4. **Full pipeline (Process -> Train -> Evaluate -> Conditional Register + Transform):**
   ```bash
   python sagemaker/pipeline/build_pipeline.py \
       --bucket YOUR_BUCKET \
       --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
   ```
   Add `--upsert-only` to define/update the pipeline in SageMaker Studio's
   Pipelines UI without immediately starting an execution.

5. **Watch it run** in SageMaker Studio under Pipelines, or:
   ```python
   from sagemaker.workflow.pipeline import Pipeline
   p = Pipeline(name="TelcoChurnPipeline")
   p.describe()
   ```

## How this maps to the design document

| Design doc concept | GitHub Actions implementation | SageMaker implementation |
|---|---|---|
| Data validation gate | `pytest` schema check | Could add a Processing step running Great Expectations before `preprocess_job.py` |
| Model training | `src/train.py` (local GridSearchCV) | `TrainingStep` with the built-in XGBoost container |
| Evaluation gate (AUC >= 0.80) | `tests/test_model_performance.py` | `ConditionStep` reading `evaluation.json` via `JsonGet` |
| Model registration | N/A (artifact uploaded to S3 directly) | `RegisterModel` -> SageMaker Model Registry |
| Batch deployment | FastAPI service on EC2 | `TransformStep` (SageMaker Batch Transform) |
| Fairness audit | `src/fairness_audit.py` (Fairlearn) | `FairnessAuditChurnModel` ProcessingStep, running `fairness_audit_job.py` |

The fairness audit now runs as its own pipeline step, `FairnessAuditChurnModel`,
in parallel with the evaluation gate (both depend on the training step, not
on each other). It is wired as informational rather than blocking, matching
the GitHub Actions design: it always runs and its output is always uploaded
as a tracked artifact, regardless of whether the AUC-ROC condition passes.

**A real cross-check worth noting:** porting this step required solving a
genuine data-plumbing problem. `preprocess_job.py` one-hot-encodes and scales
all features before writing `test.csv`, so the encoded test set no longer has
`gender`/`SeniorCitizen` in a fixed, named column position. Rather than have
`fairness_audit_job.py` guess at column positions inside the encoded matrix,
`preprocess_job.py` was updated to also write `test_sensitive_features.csv` --
the two raw sensitive columns, row-aligned with `test.csv` -- so the fairness
step can join on row order without depending on the ColumnTransformer's
internal output ordering.

This was tested end-to-end locally (see below) against a real XGBoost
booster trained on the actual processed data, and the result is reassuring:
**gender showed a demographic parity difference of 0.0045, and SeniorCitizen
showed 0.2508** -- both in the same direction and rough magnitude as the
local Logistic Regression pipeline's findings (0.0016 and 0.314,
respectively), despite being a completely different algorithm. That
consistency across two different models is good evidence the SeniorCitizen
disparity is a real property of the data and the business problem, not an
artifact of one particular model's training run.

## Local end-to-end validation of the fairness audit

Because `evaluate_job.py` and `fairness_audit_job.py` both depend on loading
a real XGBoost `Booster` artifact -- something that can't be faked with mock
objects -- they were validated by actually training a real XGBoost model
locally (50 rounds, on the real processed train split) and packaging it as
`model.tar.gz` exactly as a SageMaker Training job would, then running both
scripts against it outside of any AWS container:

```
Test AUC-ROC: 0.8498  F1: 0.5767  Precision: 0.6714  Recall: 0.5054

=== gender ===
Demographic parity difference: 0.0045
Equalized odds difference:     0.0042

=== SeniorCitizen ===
Demographic parity difference: 0.2508
Equalized odds difference:     0.2753
```

**One real risk surfaced by this test, worth knowing about before running
for real:** `booster.load_model()` raised a `UserWarning` about an
unrecognized file format, because the model file had no extension --
it succeeded only by falling back to a format auto-guess. This is a known
sensitivity in XGBoost's model serialization across versions: the *training*
side (inside SageMaker's managed container, running whatever XGBoost version
that container image bundles) and the *evaluation/fairness* side (running in
the same container image, so it should match -- but this was only confirmed
locally with whatever XGBoost version is in this validation environment) need
to agree on serialization format. If a real run throws a harder error here
rather than a warning, the fix is to either save/load with an explicit `.json`
or `.ubj` extension, or pin both training and evaluation to read the format
the container's XGBoost version writes by default.
