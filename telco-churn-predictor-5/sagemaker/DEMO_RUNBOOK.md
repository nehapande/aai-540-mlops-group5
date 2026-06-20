# Demonstrating the Telco Churn Pipeline on SageMaker

A practical, ordered runbook for setting up AWS, running the pipeline for
real, and presenting it as a live demo. Each command below was checked
against the actual AWS CLI / SageMaker SDK to confirm flag names and API
shapes are correct -- but none of it has been run against a live AWS
account, since this assistant has no AWS credentials. Treat Part 1-2 as
"should work as written" and Parts 3+ as "run this yourself and watch for
the specific failure points called out."

---

## Part 1 — One-time AWS setup (~15 minutes)

### 1.1 Create an S3 bucket

```bash
aws s3 mb s3://YOUR-NAME-telco-churn-demo --region us-east-1
```

Bucket names are globally unique across all AWS accounts, so pick
something with your name or a random suffix.

### 1.2 Create the SageMaker execution role

Save this as `trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Create the role:

```bash
aws iam create-role \
  --role-name TelcoChurnSageMakerRole \
  --assume-role-policy-document file://trust-policy.json
```

Attach the managed SageMaker policy (broad, fine for a course project —
scope it down for anything real):

```bash
aws iam attach-role-policy \
  --role-name TelcoChurnSageMakerRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

Attach S3 access scoped to just your bucket. Save as `s3-policy.json`
(replace `YOUR-NAME-telco-churn-demo` with your actual bucket name):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::YOUR-NAME-telco-churn-demo",
        "arn:aws:s3:::YOUR-NAME-telco-churn-demo/*"
      ]
    }
  ]
}
```

```bash
aws iam put-role-policy \
  --role-name TelcoChurnSageMakerRole \
  --policy-name TelcoChurnS3Access \
  --policy-document file://s3-policy.json
```

Get the role ARN you'll need for every script going forward:

```bash
aws iam get-role --role-name TelcoChurnSageMakerRole --query 'Role.Arn' --output text
```

### 1.3 Upload the raw dataset

```bash
aws s3 cp data/raw/telco_churn.csv \
  s3://YOUR-NAME-telco-churn-demo/telco-churn/raw/telco_churn.csv
```

---

## Part 2 — Where to actually run the code

**Recommended: SageMaker Studio.** It gives you a managed Jupyter
environment with the SageMaker SDK preinstalled, the Pipelines DAG
visualization (the single most useful thing for a demo), and the Model
Registry UI, all in one browser tab.

1. AWS Console → SageMaker → Studio → create a Studio domain if you don't
   have one (Quick setup is fine for this) → Launch Studio
2. Open a Notebook (JupyterLab) inside Studio
3. Upload or `git clone` this repo into the Studio environment
4. Open a terminal inside Studio and `pip install -r requirements.txt`
   (Studio's default kernel already has `sagemaker` and `boto3`, but not
   `fairlearn`/`imbalanced-learn`/`xgboost` at the versions this repo pins)

Alternative: a plain **SageMaker Notebook Instance** works too, with the
same setup steps — slightly older UI, no integrated Pipelines DAG viewer in
the notebook itself (you'd view the DAG in the SageMaker console's Pipelines
page instead, which is still fine for a demo).

---

## Part 3 — Cheapest possible first test: one Training job, no Pipeline

Before standing up the full DAG, confirm the basics work with the smallest
possible real AWS interaction. This costs a few cents and takes ~5 minutes.

```bash
# 1. Run preprocessing locally, upload its output to S3
PYTHONPATH=src python3 src/preprocess.py
python3 sagemaker/processing/preprocess_job.py \
  --input-path data/raw --output-path /tmp/sm_local_output
aws s3 sync /tmp/sm_local_output/train s3://YOUR-NAME-telco-churn-demo/telco-churn/processed/train/
aws s3 sync /tmp/sm_local_output/validation s3://YOUR-NAME-telco-churn-demo/telco-churn/processed/validation/
aws s3 sync /tmp/sm_local_output/test s3://YOUR-NAME-telco-churn-demo/telco-churn/processed/test/

# 2. Launch the real Training job
python3 sagemaker/training/launch_training_job.py \
  --bucket YOUR-NAME-telco-churn-demo \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/TelcoChurnSageMakerRole
```

**What to watch for while this runs:**
- The script blocks and streams CloudWatch logs to your terminal (`logs="All"`) — you'll see XGBoost's actual training rounds and AUC printed live
- If it fails immediately with an access-denied error, it's almost always the IAM role missing a permission — check the role has `AmazonSageMakerFullAccess` attached
- If it fails on reading the S3 input, double check the `--bucket`/`--prefix` match exactly where you uploaded the processed CSVs in step 1

When it finishes, confirm the artifact landed in S3:

```bash
aws s3 ls s3://YOUR-NAME-telco-churn-demo/telco-churn/models/ --recursive
```

This alone is a legitimate, demonstrable result: a real model trained on
real AWS infrastructure, with a real S3 artifact you can point to.

---

## Part 4 — Standing up the full Pipeline

**Recommended for a demo: use the two notebooks in `sagemaker/notebooks/`.**
They walk through registration and execution step-by-step with printed
output at each stage, which reads much better live than a terminal running
a single script. Open `01_register_pipeline.ipynb`, set `BUCKET`/`ROLE_ARN`,
flip `DRY_RUN` to `False`, and run all cells. Then do the same in
`02_execute_pipeline.ipynb`. Both notebooks default to `DRY_RUN = True`,
which is worth running once first, untouched, to see the full expected
flow and output shape before spending any real AWS time.

Equivalent, if you'd rather drive it from a terminal/script instead:

```bash
python3 sagemaker/pipeline/build_pipeline.py \
  --bucket YOUR-NAME-telco-churn-demo \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/TelcoChurnSageMakerRole \
  --upsert-only
```

`--upsert-only` registers the pipeline definition in SageMaker without
spending compute on an execution yet — good for confirming the DAG itself
is valid before paying for a run.

**Where to look:** AWS Console → SageMaker → Pipelines → `TelcoChurnPipeline`.
You'll see the 5-step DAG rendered visually: Preprocess → Train → (branches
into) Evaluate and Fairness Audit in parallel → Condition Check → (if true)
Register + Create Model + Batch Transform. This graph view is the single
best thing to have on screen during a live demo — it makes the architecture
self-explanatory without you narrating every step.

When ready to actually execute it (script equivalent of `02_execute_pipeline.ipynb`):

```bash
python3 sagemaker/pipeline/build_pipeline.py \
  --bucket YOUR-NAME-telco-churn-demo \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/TelcoChurnSageMakerRole
```

(same command, without `--upsert-only`, starts an execution).

**Expected runtime:** roughly 10-20 minutes total across all steps for a
dataset this size, mostly spent on container spin-up time per step rather
than actual compute (each Processing/Training job takes a few minutes just
to provision, regardless of how small the data is).

---

## Part 5 — What to actually demonstrate (suggested live-demo sequence)

If this is for a course presentation, here's an order that tells a story
rather than just listing AWS services:

1. **Show the design document's architecture section** — frame what you're
   about to demo as the realization of a documented design decision, not
   just "look what AWS can do."

2. **Open the Pipelines DAG in the SageMaker console** (Console → SageMaker
   → Pipelines → TelcoChurnPipeline → Graph tab). Walk through each node:
   - Preprocess: "this is the same cleaning logic as our local `preprocess.py`, ported to run as a managed job"
   - Train: "built-in XGBoost container, no custom Docker image needed"
   - point at the parallel branch: "evaluation and fairness audit run side by side — fairness is informational, not blocking, which is itself a documented design decision"
   - Condition step: "this is the 0.80 AUC-ROC gate from our design doc, now enforced automatically rather than just asserted in a test file"

3. **Click into a completed execution** and show the actual runtime per step
   — this is a good moment to talk about the cost/latency tradeoffs of
   managed infrastructure vs. the local pipeline's near-instant iteration loop.

4. **Open the Evaluation step's output** (S3 → `.../evaluation/evaluation.json`)
   and show the real AUC-ROC number SageMaker computed — tie it back to the
   0.8549 number from the local pipeline as a sanity-check comparison
   ("different infrastructure, consistent result").

5. **Open the Fairness Audit step's output** and show the SeniorCitizen
   disparity finding — this is the most substantive, discussion-worthy part
   of the whole project, and demonstrating that it survived the port to
   managed infrastructure (and was even cross-validated against a different
   algorithm during development) is a strong note to land on.

6. **Show the Model Registry entry** (Console → SageMaker → Model Registry
   → `telco-churn-xgboost-models`) — point out the model's approval status
   and the attached `model_metrics`, which is what a real MLOps reviewer
   would check before approving production deployment.

7. **Show the Batch Transform output in S3** — the actual per-customer churn
   predictions, tying back to the design document's "batch, not real-time"
   deployment decision.

8. **Close by comparing this to the GitHub Actions implementation** — same
   logical pipeline, two different pieces of infrastructure, and a brief,
   honest note on the tradeoffs (managed-but-slower-iteration vs.
   self-hosted-but-more-control) rather than presenting one as simply better.

---

## Part 6 — Known failure points to have an answer ready for

- **Model file format warning on load**: documented in `sagemaker/README.md`
  — if `evaluate_job.py` or `fairness_audit_job.py` throws a harder error
  than a warning when loading the trained booster, it's an XGBoost
  serialization version mismatch between the training and evaluation steps.
  Fix: pin both to read/write the same explicit format (`.json` or `.ubj`).
- **IAM AccessDenied**: almost always the execution role missing a specific
  permission. `AmazonSageMakerFullAccess` plus the scoped S3 policy above
  covers everything this pipeline needs; if you tightened permissions
  further, check CloudTrail for the specific denied action.
- **ConditionStep evaluates to false unexpectedly**: check the
  `evaluation.json` the Evaluate step actually wrote in S3 — if the AUC-ROC
  is below 0.80 on a SageMaker-trained run even though the local pipeline
  cleared it at 0.8549, the most likely cause is the XGBoost hyperparameters
  in `launch_training_job.py`/`build_pipeline.py` not matching the local
  `train.py` XGBoost candidate exactly, or the train/test split landing
  differently due to a SageMaker-side data type quirk in the CSV write/read
  round-trip.
- **Cost runaway**: nothing here should cost more than a few dollars for a
  handful of runs, but set a budget alert (AWS Console → Billing → Budgets)
  before a live demo so a forgotten `ml.p3` instance type typo doesn't
  surprise you. Everything in this repo's scripts defaults to `ml.m5.xlarge`
  or smaller.

---

## Part 7 — Cleanup after the demo

```bash
# Delete the pipeline definition (does not delete past execution history/artifacts)
aws sagemaker delete-pipeline --pipeline-name TelcoChurnPipeline

# Empty and delete the S3 bucket
aws s3 rm s3://YOUR-NAME-telco-churn-demo --recursive
aws s3 rb s3://YOUR-NAME-telco-churn-demo

# Delete the IAM role
aws iam detach-role-policy --role-name TelcoChurnSageMakerRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam delete-role-policy --role-name TelcoChurnSageMakerRole --policy-name TelcoChurnS3Access
aws iam delete-role --role-name TelcoChurnSageMakerRole
```

Also check the SageMaker console for any registered models in the Model
Registry and any Studio domains left running — Studio domains in particular
can accrue idle storage charges if left around.
