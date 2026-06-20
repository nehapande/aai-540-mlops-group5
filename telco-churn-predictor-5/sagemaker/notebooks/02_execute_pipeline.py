# %% [markdown]
# # Execute the Telco Churn SageMaker Pipeline
#
# This notebook starts a real execution of the `TelcoChurnPipeline`
# (already registered via `01_register_pipeline.ipynb`), polls it to
# completion, and pulls the evaluation and fairness audit results back
# from S3 for inspection.
#
# **DRY_RUN mode**: as in the registration notebook, set `DRY_RUN = True`
# (default) to validate the entire flow without making AWS calls. The
# dry-run path prints realistic *example* output drawn from this project's
# actual local pipeline results, clearly labeled as simulated, so you can
# see what a real run's output looks like before spending any AWS time or
# money. Set `DRY_RUN = False` once you're ready to run for real.

# %%
DRY_RUN = True  # set to False to actually call AWS

# %% [markdown]
# ## 1. Setup

# %%
import json
import time

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline

print(f"sagemaker SDK version: {sagemaker.__version__}")
print(f"DRY_RUN = {DRY_RUN}")

# %% [markdown]
# ## 2. Configure parameters
#
# Must match what you used in `01_register_pipeline.ipynb`.

# %%
BUCKET = "YOUR-NAME-telco-churn-demo"
ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/TelcoChurnSageMakerRole"
REGION = "us-east-1"
PIPELINE_NAME = "TelcoChurnPipeline"

# Optional: override pipeline parameters for this specific execution
# without redefining the pipeline. Leave as None to use the defaults
# baked into the pipeline definition (see build_pipeline.py).
EXECUTION_PARAMETERS = {
    "MinAucRocThreshold": 0.80,
}

print(f"Bucket:        {BUCKET}")
print(f"Role ARN:      {ROLE_ARN}")
print(f"Region:        {REGION}")
print(f"Pipeline name: {PIPELINE_NAME}")
print(f"Execution parameter overrides: {EXECUTION_PARAMETERS}")

if not DRY_RUN:
    assert "YOUR" not in BUCKET, "Replace the placeholder BUCKET before running for real."
    assert "YOUR" not in ROLE_ARN, "Replace the placeholder ROLE_ARN before running for real."

# %% [markdown]
# ## 3. Connect to the registered pipeline
#
# This assumes `01_register_pipeline.ipynb` has already been run with
# `DRY_RUN = False` against this same pipeline name.

# %%
if DRY_RUN:
    print(f"[DRY RUN] Would connect to pipeline '{PIPELINE_NAME}' in region {REGION}.")
    pipeline = None
else:
    boto_session = boto3.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)
    pipeline = Pipeline(name=PIPELINE_NAME, sagemaker_session=sm_session)
    print(f"Connected to pipeline: {pipeline.name}")

# %% [markdown]
# ## 4. Start an execution

# %%
if DRY_RUN:
    print("[DRY RUN] Would call pipeline.start(parameters=EXECUTION_PARAMETERS).")
    print("[DRY RUN] Simulating an execution ARN for the rest of this notebook:")
    execution_arn = (
        "arn:aws:sagemaker:us-east-1:106122975094:pipeline/TelcoChurnPipeline/"
        "execution/dryrun0000000"
    )
    print(f"[DRY RUN]   {execution_arn}")
else:
    execution = pipeline.start(parameters=EXECUTION_PARAMETERS)
    execution_arn = execution.arn
    print(f"Started execution: {execution_arn}")

# %% [markdown]
# ## 5. Poll for completion
#
# A real execution takes roughly 10-20 minutes end-to-end across all
# steps (mostly container spin-up time per step, not actual compute, given
# this dataset's size). This cell polls every 30 seconds and prints status
# changes as they happen.

# %%
if DRY_RUN:
    print("[DRY RUN] Would poll execution.describe()['PipelineExecutionStatus']")
    print("[DRY RUN] every 30 seconds until it reaches 'Succeeded' or 'Failed'.")
    print("[DRY RUN] Simulating a successful completion for the rest of this notebook.")
    final_status = "Succeeded"
    print(f"[DRY RUN] Final status: {final_status}")
else:
    POLL_INTERVAL_SECONDS = 30
    last_status = None
    while True:
        status = execution.describe()["PipelineExecutionStatus"]
        if status != last_status:
            print(f"Status: {status}")
            last_status = status
        if status in ("Succeeded", "Failed", "Stopped"):
            final_status = status
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    print(f"Execution finished with status: {final_status}")

# %% [markdown]
# ## 6. List step-by-step results
#
# Shows the status of each of the 5 top-level steps, including which
# branch of the Condition step was taken.

# %%
if DRY_RUN:
    # Simulated step list, shaped exactly like the real
    # execution.list_steps() response, with statuses representing a
    # successful run that cleared the AUC-ROC gate.
    simulated_steps = [
        {"StepName": "PreprocessTelcoChurnData", "StepStatus": "Succeeded"},
        {"StepName": "TrainXGBoostChurnModel", "StepStatus": "Succeeded"},
        {"StepName": "EvaluateChurnModel", "StepStatus": "Succeeded"},
        {"StepName": "FairnessAuditChurnModel", "StepStatus": "Succeeded"},
        {"StepName": "CheckAucRocThreshold", "StepStatus": "Succeeded"},
        {"StepName": "RegisterChurnModel", "StepStatus": "Succeeded"},
        {"StepName": "CreateChurnModelForTransform", "StepStatus": "Succeeded"},
        {"StepName": "BatchScoreTestSet", "StepStatus": "Succeeded"},
    ]
    print("[DRY RUN] Simulated step statuses:")
    for s in simulated_steps:
        print(f"  {s['StepName']:35s} {s['StepStatus']}")
else:
    steps = execution.list_steps()
    for s in steps:
        print(f"  {s['StepName']:35s} {s['StepStatus']}")

# %% [markdown]
# ## 7. Pull the evaluation results from S3
#
# The `EvaluateChurnModel` step writes `evaluation.json` to S3. We fetch it
# directly to see the real AUC-ROC the SageMaker-trained model achieved on
# the test set, and compare it against the local pipeline's 0.8549 result
# as a sanity check.

# %%
if DRY_RUN:
    # Example numbers from this project's actual local validation run
    # (sagemaker/README.md), clearly NOT from a real SageMaker execution.
    print("[DRY RUN] Simulated evaluation.json contents (from local validation, NOT a real S3 fetch):")
    simulated_evaluation = {
        "binary_classification_metrics": {
            "auc": {"value": 0.8498, "standard_deviation": "NaN"},
            "f1": {"value": 0.5767, "standard_deviation": "NaN"},
            "precision": {"value": 0.6714, "standard_deviation": "NaN"},
            "recall": {"value": 0.5054, "standard_deviation": "NaN"},
        }
    }
    print(json.dumps(simulated_evaluation, indent=2))
else:
    s3 = boto3.client("s3", region_name=REGION)
    # Find the actual evaluation step's output S3 URI from the execution
    steps = execution.list_steps()
    eval_step = next(s for s in steps if s["StepName"] == "EvaluateChurnModel")
    eval_s3_uri = eval_step["Metadata"]["ProcessingJob"]["Arn"]  # job ARN; describe it for output path
    sm_client = boto3.client("sagemaker", region_name=REGION)
    job_name = eval_s3_uri.split("/")[-1]
    job_desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
    output_s3_uri = job_desc["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    bucket_name, key_prefix = output_s3_uri.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket_name, Key=f"{key_prefix}/evaluation.json")
    evaluation_report = json.loads(obj["Body"].read())
    print(json.dumps(evaluation_report, indent=2))

# %% [markdown]
# ## 8. Pull the fairness audit results from S3
#
# Same pattern, for the `FairnessAuditChurnModel` step's output.

# %%
if DRY_RUN:
    print("[DRY RUN] Simulated fairness_audit.json contents (from local validation, NOT a real S3 fetch):")
    simulated_fairness = {
        "gender": {
            "demographic_parity_difference": 0.0045,
            "equalized_odds_difference": 0.0042,
        },
        "SeniorCitizen": {
            "demographic_parity_difference": 0.2508,
            "equalized_odds_difference": 0.2753,
        },
    }
    print(json.dumps(simulated_fairness, indent=2))
    print()
    print("[DRY RUN] Note: SeniorCitizen shows a substantial disparity (~0.25) versus")
    print("[DRY RUN] gender (~0.005) -- this is consistent with the local pipeline's")
    print("[DRY RUN] findings and warrants the same human review discussed in the")
    print("[DRY RUN] design document's Security Checklist.")
else:
    steps = execution.list_steps()
    fairness_step = next(s for s in steps if s["StepName"] == "FairnessAuditChurnModel")
    job_arn = fairness_step["Metadata"]["ProcessingJob"]["Arn"]
    job_name = job_arn.split("/")[-1]
    job_desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
    output_s3_uri = job_desc["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    bucket_name, key_prefix = output_s3_uri.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket_name, Key=f"{key_prefix}/fairness_audit.json")
    fairness_report = json.loads(obj["Body"].read())
    print(json.dumps(fairness_report, indent=2))

# %% [markdown]
# ## 9. Check whether the condition gate passed
#
# If `CheckAucRocThreshold` took the `if_steps` branch, `RegisterChurnModel`
# and `BatchScoreTestSet` will show `Succeeded`; if it took the (empty)
# `else_steps` branch, they won't appear in the step list at all.

# %%
if DRY_RUN:
    print("[DRY RUN] Simulated: condition passed -- RegisterChurnModel and")
    print("[DRY RUN] BatchScoreTestSet both ran.")
else:
    steps = execution.list_steps()
    step_names = {s["StepName"] for s in steps}
    if "RegisterChurnModel" in step_names:
        print("Condition PASSED -- model was registered and batch-scored.")
    else:
        print("Condition FAILED -- model did not clear the AUC-ROC threshold.")
        print("Check the EvaluateChurnModel step's output above for the actual AUC-ROC.")

# %% [markdown]
# ## 10. List registered model packages
#
# Confirms the model landed in the SageMaker Model Registry, ready for
# manual approval.

# %%
if DRY_RUN:
    print("[DRY RUN] Would call sagemaker_client.list_model_packages(")
    print("[DRY RUN]     ModelPackageGroupName='telco-churn-xgboost-models')")
else:
    sm_client = boto3.client("sagemaker", region_name=REGION)
    response = sm_client.list_model_packages(
        ModelPackageGroupName="telco-churn-xgboost-models",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=5,
    )
    for pkg in response["ModelPackageSummaryList"]:
        print(f"  {pkg['ModelPackageArn']}")
        print(f"    Status: {pkg['ModelApprovalStatus']}  Created: {pkg['CreationTime']}")

# %% [markdown]
# ## 11. Summary
#
# - Pipeline execution: see status in Section 5
# - Test-set AUC-ROC: see Section 7 — compare against the local pipeline's 0.8549
# - Fairness findings: see Section 8 — compare against the local pipeline's
#   gender (0.0016) and SeniorCitizen (0.314) demographic parity differences
# - Model Registry: see Section 10 for the registered model package, pending
#   manual approval before any real production use
#
# **Next step in the SageMaker console:** AWS Console → SageMaker → Model
# Registry → `telco-churn-xgboost-models` → review the model's metrics and
# manually approve or reject it.
