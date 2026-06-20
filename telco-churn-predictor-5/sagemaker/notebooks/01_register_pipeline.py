# %% [markdown]
# # Register the Telco Churn SageMaker Pipeline
#
# This notebook builds the `TelcoChurnPipeline` definition (Process ->
# Train -> Evaluate + Fairness Audit -> Condition -> Register + Transform)
# and registers (`upsert`s) it with SageMaker. It does **not** start an
# execution — that's the companion notebook, `02_execute_pipeline.ipynb`.
#
# Run this notebook from SageMaker Studio, a SageMaker Notebook Instance,
# or any machine with AWS credentials configured and the `sagemaker` SDK
# installed.
#
# **DRY_RUN mode**: set `DRY_RUN = True` (default) to validate the entire
# notebook's logic — including constructing the full Pipeline object and
# resolving every step's property references — without making any AWS
# calls that require real credentials. Set `DRY_RUN = False` once you've
# filled in your real bucket/role below and are ready to register the
# pipeline for real.

# %%
DRY_RUN = True  # set to False to actually call AWS

# %% [markdown]
# ## 1. Setup

# %%
import sys
import json

sys.path.insert(0, "../pipeline")

import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

from build_pipeline import build_pipeline, PIPELINE_NAME, MODEL_PACKAGE_GROUP_NAME

print(f"sagemaker SDK version: {sagemaker.__version__}")
print(f"Pipeline name: {PIPELINE_NAME}")
print(f"Model package group: {MODEL_PACKAGE_GROUP_NAME}")
print(f"DRY_RUN = {DRY_RUN}")

# %% [markdown]
# ## 2. Configure parameters
#
# Replace these with your actual bucket and IAM role ARN before setting
# `DRY_RUN = False`. In dry-run mode, placeholder values are fine — no
# network call is made against them.

# %%
BUCKET = "YOUR-NAME-telco-churn-demo"
ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/TelcoChurnSageMakerRole"
REGION = "us-east-1"

print(f"Bucket:   {BUCKET}")
print(f"Role ARN: {ROLE_ARN}")
print(f"Region:   {REGION}")

if not DRY_RUN:
    assert "YOUR" not in BUCKET, "Replace the placeholder BUCKET before running for real."
    assert "YOUR" not in ROLE_ARN, "Replace the placeholder ROLE_ARN before running for real."

# %% [markdown]
# ## 3. Verify AWS credentials (skipped in dry-run mode)
#
# A quick, cheap call (`sts:GetCallerIdentity`) to confirm credentials are
# present and valid before attempting anything more expensive.

# %%
if DRY_RUN:
    print("[DRY RUN] Skipping AWS credential check.")
else:
    sts = boto3.client("sts", region_name=REGION)
    identity = sts.get_caller_identity()
    print(f"Authenticated as: {identity['Arn']}")
    print(f"Account: {identity['Account']}")

# %% [markdown]
# ## 4. Build the pipeline definition
#
# `build_pipeline()` constructs the full step graph locally — this does
# **not** require AWS credentials on its own (confirmed: no network calls
# are made just by instantiating the SDK's step/parameter objects). It
# only becomes a real AWS interaction once we call `.upsert()` below.

# %%
boto_session = boto3.Session(region_name=REGION)
pipeline_session = PipelineSession(boto_session=boto_session, default_bucket=BUCKET)

pipeline = build_pipeline(
    bucket=BUCKET,
    role_arn=ROLE_ARN,
    region=REGION,
    pipeline_session=pipeline_session,
)

print(f"Pipeline '{pipeline.name}' constructed with {len(pipeline.steps)} top-level steps:")
for step in pipeline.steps:
    print(f"  - {step.name} ({type(step).__name__})")

# %% [markdown]
# ## 5. Inspect the pipeline parameters
#
# These are the values an execution can override without redefining the
# pipeline — e.g. running with a different `MinAucRocThreshold` for an
# experiment, without touching this notebook.

# %%
for param in pipeline.parameters:
    print(f"  {param.name}: default = {param.default_value!r}")

# %% [markdown]
# ## 6. Register (upsert) the pipeline
#
# This is the first cell that makes a real, mutating AWS API call
# (`sagemaker:CreatePipeline` / `sagemaker:UpdatePipeline` under the hood).
# In dry-run mode this is skipped and replaced with a description of what
# would happen.

# %%
if DRY_RUN:
    print("[DRY RUN] Would call pipeline.upsert(role_arn=ROLE_ARN).")
    print("[DRY RUN] This would create or update the pipeline definition named")
    print(f"[DRY RUN]   '{PIPELINE_NAME}' in your SageMaker account in region {REGION}.")
else:
    response = pipeline.upsert(role_arn=ROLE_ARN)
    print("Pipeline upserted successfully.")
    print(json.dumps(response, indent=2, default=str))

# %% [markdown]
# ## 7. Confirm registration
#
# Describe the pipeline back from SageMaker to confirm it's registered
# and matches what we just defined.

# %%
if DRY_RUN:
    print("[DRY RUN] Would call pipeline.describe() and print the PipelineArn.")
else:
    description = pipeline.describe()
    print(f"Pipeline ARN: {description['PipelineArn']}")
    print(f"Pipeline status: {description['PipelineStatus']}")
    print(f"Last modified: {description['LastModifiedTime']}")

# %% [markdown]
# ## 8. Where to look in the console
#
# AWS Console → SageMaker → Pipelines → `TelcoChurnPipeline` → **Graph**
# tab. You should see the 5-step DAG: Preprocess → Train → (parallel)
# Evaluate + Fairness Audit → Condition Check → (if AUC-ROC ≥ 0.80)
# Register + Create Model + Batch Transform.
#
# No execution has been started yet — the pipeline is only *defined*.
# Continue to `02_execute_pipeline.ipynb` to actually run it.
