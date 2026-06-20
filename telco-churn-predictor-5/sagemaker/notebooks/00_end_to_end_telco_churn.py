# %% [markdown]
# # Telco Customer Churn Prediction
# ## Train, Deploy, Batch Transform, Fairness Audit + Model Registry on Amazon SageMaker
#
# End-to-end demonstration: data prep → XGBoost training → batch transform
# (with input/output filtering) → fairness audit → model registration →
# model card → real-time endpoint deployment → cleanup.
#
# Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
# (IBM sample data, 7,043 customers × 21 columns). Binary target: `Churn`
# (Yes/No).
#
# **Run this notebook inside SageMaker Studio or a SageMaker Notebook
# Instance** — `sagemaker.get_execution_role()` only resolves correctly in
# those environments. Cells through Section 2 (Data Preparation) are pure
# pandas/numpy and need no AWS access; everything from Section 3 onward
# makes real, billable AWS API calls.

# %% [markdown]
# ---
# ## 0. Setup

# %%
import os
import re
import json
import boto3
import sagemaker
import numpy as np
import pandas as pd
from time import gmtime, strftime, sleep
import datetime

print(f"sagemaker SDK version: {sagemaker.__version__}")

# %% [markdown]
# The next cell only works inside a SageMaker-managed environment (Studio,
# Notebook Instance, or a Training/Processing job container) — it resolves
# the execution role attached to that environment.

# %%
role = sagemaker.get_execution_role()
sess = sagemaker.Session()
region = sess.boto_region_name
bucket = sess.default_bucket()
prefix = "DEMO-telco-churn-xgboost"

# Boto3 clients
s3 = boto3.client("s3")
sm_client = boto3.client("sagemaker", region_name=region)
sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

print("Region :", region)
print("Bucket :", bucket)
print("Role   :", role)

# %% [markdown]
# ---
# ## 1. Data Preparation

# %% [markdown]
# ### 1.1 Load Raw Data
#
# Downloaded from IBM's public GitHub mirror of the Kaggle dataset (no
# Kaggle authentication required).

# %%
filename = "telco_churn.csv"
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data = pd.read_csv(url)
data.to_csv(filename, index=False)

print(f"Dataset shape: {data.shape}")
data.sample(8)

# %% [markdown]
# #### Key observations:
# - 7,043 customers, 21 columns
# - `customerID`: unique identifier — excluded from training, kept for
#   joining batch predictions back to source rows
# - `Churn`: binary target (Yes/No), ~26.5% positive class — moderately
#   imbalanced
# - `TotalCharges` is read as a string and contains 11 blank values
#   (customers with `tenure == 0`, i.e. brand-new accounts not yet billed)
# - 22 duplicate rows exist when comparing all columns *except*
#   `customerID` (each row has a unique ID, so a plain full-row duplicate
#   check would never find any) — these are removed below

# %%
non_id_cols = data.columns.difference(["customerID"])
print(
    "Duplicate rows (excluding customerID):", data.duplicated(subset=non_id_cols).sum()
)
print(
    "Blank TotalCharges rows:",
    (data["TotalCharges"].astype(str).str.strip() == "").sum(),
)
print()
print(data["Churn"].value_counts())
print(data["Churn"].value_counts(normalize=True).round(4))

# %% [markdown]
# ### 1.2 Clean Data

# %%
# Fix TotalCharges type; blanks correspond to tenure == 0 customers, impute as 0.0
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)

# Drop duplicate rows, comparing all columns except customerID (see note above)
non_id_cols = data.columns.difference(["customerID"])
data = data.drop_duplicates(subset=non_id_cols).reset_index(drop=True)

# Encode the target and gender as 0/1
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
data["gender"] = data["gender"].map({"Male": 1, "Female": 0})

# Encode remaining binary Yes/No columns
for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    data[col] = data[col].map({"Yes": 1, "No": 0})

print(f"Shape after cleaning: {data.shape}")
data.isnull().sum().sum(), "nulls remaining (should be 0)"

# %% [markdown]
# ### 1.3 Feature Engineering
#
# Three derived features, validated against the actual churn rate:
# - `tenure_group`: lifecycle bucket (0–12, 13–24, 25–48, 49–72 months)
# - `has_support_services`: 1 if the customer has OnlineSecurity OR TechSupport
# - `num_services`: count of subscribed add-on services (0–8)

# %%
data["tenure_group"] = pd.cut(
    data["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"],
)

data["has_support_services"] = (
    (data["OnlineSecurity"] == "Yes") | (data["TechSupport"] == "Yes")
).astype(int)

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
data["num_services"] = data[service_cols].apply(
    lambda row: sum(1 for v in row if v in ("Yes", 1, "DSL", "Fiber optic")), axis=1
)

print("Churn rate by tenure_group:")
print(data.groupby("tenure_group")["Churn"].mean())
print()
print("Churn rate by has_support_services:")
print(data.groupby("has_support_services")["Churn"].mean())

# %% [markdown]
# ### 1.4 One-Hot Encode Remaining Categorical Columns
#
# XGBoost's built-in algorithm requires a purely numeric CSV input, so all
# remaining multi-category string columns are one-hot encoded.

# %%
categorical_cols = [
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
    "tenure_group",
]
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# customerID, Churn, and all numeric/encoded features remain;
# bool columns from get_dummies need to be cast to int for XGBoost
bool_cols = data_encoded.select_dtypes(include="bool").columns
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)

print(f"Shape after one-hot encoding: {data_encoded.shape}")
data_encoded.head(3)

# %% [markdown]
# ### 1.5 Train / Validation / Batch Split
#
# - **80%** training — `Churn` label included, `customerID` dropped
# - **10%** validation — same as training, used for early stopping signal
# - **10%** batch — `Churn` dropped (it's what we're predicting),
#   `customerID` **kept** for two of the three batch-transform demos below

# %%
np.random.seed(42)
rand_split = np.random.rand(len(data_encoded))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
batch_list = rand_split >= 0.9

data_train = data_encoded[train_list].drop(["customerID"], axis=1)
data_val = data_encoded[val_list].drop(["customerID"], axis=1)
data_batch = data_encoded[batch_list].drop(["Churn"], axis=1)  # keeps customerID
data_batch_noID = data_batch.drop(["customerID"], axis=1)  # no customerID, no Churn

# Retained separately for the fairness audit in Section 6 — ground truth +
# sensitive attributes for every row in the batch split, keyed by customerID
batch_truth = data_encoded[batch_list][
    ["customerID", "gender", "SeniorCitizen", "Churn"]
]

print(f"Train      : {data_train.shape}")
print(f"Validation : {data_val.shape}")
print(f"Batch      : {data_batch.shape}")
print(f"Batch noID : {data_batch_noID.shape}")

# %% [markdown]
# `Churn` must be the **first column** in the training/validation CSVs —
# this is what SageMaker's built-in XGBoost container expects.

# %%
churn_col = data_train.pop("Churn")
data_train.insert(0, "Churn", churn_col)
churn_col_val = data_val.pop("Churn")
data_val.insert(0, "Churn", churn_col_val)

print("Train columns (first 5):", list(data_train.columns[:5]))

# %% [markdown]
# ### 1.6 Write Splits Locally

# %%
train_file = "train_data.csv"
validation_file = "validation_data.csv"
batch_file = "batch_data.csv"
batch_file_noID = "batch_data_noID.csv"

data_train.to_csv(train_file, index=False, header=False)
data_val.to_csv(validation_file, index=False, header=False)
data_batch.to_csv(batch_file, index=False, header=False)
data_batch_noID.to_csv(batch_file_noID, index=False, header=False)

print("Files written:")
for f in [train_file, validation_file, batch_file, batch_file_noID]:
    print(f"  {f}: {os.path.getsize(f):,} bytes")

# %% [markdown]
# ---
# ## 2. Upload Splits to S3
#
# **First AWS-dependent step.** Everything above this point is pure local
# pandas/numpy and required no AWS access.

# %%
sess.upload_data(train_file, bucket=bucket, key_prefix=f"{prefix}/train")
sess.upload_data(validation_file, bucket=bucket, key_prefix=f"{prefix}/validation")
sess.upload_data(batch_file, bucket=bucket, key_prefix=f"{prefix}/batch")
sess.upload_data(batch_file_noID, bucket=bucket, key_prefix=f"{prefix}/batch")

print("All splits uploaded to S3.")
print(f"s3://{bucket}/{prefix}/")

# %% [markdown]
# ---
# ## 3. XGBoost Training Job
#
# Binary classification with `binary:logistic`. `scale_pos_weight` is set
# to the actual training-set class ratio (~2.78) to address the ~73/27
# class imbalance, rather than relying on resampling.

# %%
churn_count = data_train["Churn"].value_counts()
scale_pos_weight = churn_count[0] / churn_count[1]
print(f"Training set class counts: {dict(churn_count)}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

# %%
# %%time

job_name = "telco-churn-xgb-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
output_location = "s3://{}/{}/output/{}".format(bucket, prefix, job_name)
image = sagemaker.image_uris.retrieve(
    framework="xgboost", region=region, version="1.7-1"
)

sm_estimator = sagemaker.estimator.Estimator(
    image,
    role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size=50,
    input_mode="File",
    output_path=output_location,
    sagemaker_session=sess,
)

sm_estimator.set_hyperparameters(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    num_round=200,
)

train_data = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train".format(bucket, prefix),
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
)
validation_data = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validation".format(bucket, prefix),
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
)
data_channels = {"train": train_data, "validation": validation_data}

sm_estimator.fit(inputs=data_channels, job_name=job_name, logs=True)

# %% [markdown]
# ---
# ## 4. Batch Transform
#
# SageMaker Batch Transform supports three I/O processing attributes:
# `input_filter`, `join_source`, and `output_filter`. Demonstrated below in
# increasing order of usefulness for a real retention workflow, where you
# need the predicted probability attached back to a customer identifier.

# %% [markdown]
# ### 4.1 Baseline — No I/O Filtering
#
# Using `batch_data_noID.csv` (no `customerID`). Output is raw probability
# scores only, in row order — not directly useful without re-joining
# against the input by hand.

# %%
# %%time

sm_transformer = sm_estimator.transformer(1, "ml.m5.xlarge")

input_location = "s3://{}/{}/batch/{}".format(bucket, prefix, batch_file_noID)
sm_transformer.transform(input_location, content_type="text/csv", split_type="Line")
sm_transformer.wait()


# %%
def get_csv_output_from_s3(s3uri, file_name_suffix):
    file_name = "{}.out".format(file_name_suffix)
    match = re.match("s3://([^/]+)/(.*)", "{}/{}".format(s3uri, file_name))
    out_bucket, out_prefix = match.group(1), match.group(2)
    s3.download_file(out_bucket, out_prefix, file_name)
    return pd.read_csv(file_name, sep=",", header=None)


baseline_output = get_csv_output_from_s3(sm_transformer.output_path, batch_file_noID)
print("Baseline batch output (probabilities only, no customerID):")
baseline_output.head(8)

# %% [markdown]
# ### 4.2 Join Input + Predictions (`input_filter` + `join_source`)
#
# - `input_filter="$[1:]"` — strip the first column (`customerID`) before
#   sending the row to the model
# - `join_source="Input"` — reattach the prediction to the *original*
#   input row (including the `customerID` the model never saw)

# %%
# %%time

sm_transformer.assemble_with = "Line"
sm_transformer.accept = "text/csv"

input_location = "s3://{}/{}/batch/{}".format(bucket, prefix, batch_file)
sm_transformer.transform(
    input_location,
    split_type="Line",
    content_type="text/csv",
    input_filter="$[1:]",
    join_source="Input",
)
sm_transformer.wait()

# %%
joined_output = get_csv_output_from_s3(sm_transformer.output_path, batch_file)
print("Joined output (customerID + all features + probability):")
joined_output.head(8)

# %% [markdown]
# ### 4.3 Output Filter — customerID + Probability Only
#
# `output_filter="$[0,-1]"` keeps only column 0 (`customerID`, from the
# joined input) and the last column (the prediction) — exactly what a
# retention dashboard needs, with no manual re-joining required.

# %%
# %%time

sm_transformer.transform(
    input_location,
    split_type="Line",
    content_type="text/csv",
    input_filter="$[1:]",
    join_source="Input",
    output_filter="$[0,-1]",
)
sm_transformer.wait()

# %%
filtered_output = get_csv_output_from_s3(sm_transformer.output_path, batch_file)
filtered_output.columns = ["customerID", "churn_probability"]
print("Filtered output (customerID + churn probability only):")
filtered_output.head(8)

# %% [markdown]
# ---
# ## 5. Fairness Audit
#
# Using the joined batch output above, compute Fairlearn's demographic
# parity difference and equalized odds difference across two sensitive
# attributes — `gender` and `SeniorCitizen` — by joining the predictions
# back to the ground-truth labels and sensitive attributes retained in
# `batch_truth` (Section 1.5).

# %%
# !pip install -q fairlearn

# %%
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import recall_score, precision_score, roc_auc_score

# filtered_output has customerID + probability; merge with ground truth
eval_df = filtered_output.merge(batch_truth, on="customerID", how="inner")
eval_df["churn_pred"] = (eval_df["churn_probability"] >= 0.5).astype(int)

print(f"Merged evaluation set: {eval_df.shape[0]} rows")
print(
    f"Batch AUC-ROC: {roc_auc_score(eval_df['Churn'], eval_df['churn_probability']):.4f}"
)

# %%
fairness_results = {}
for sensitive_col in ["gender", "SeniorCitizen"]:
    sensitive_features = eval_df[sensitive_col]

    dp_diff = demographic_parity_difference(
        eval_df["Churn"], eval_df["churn_pred"], sensitive_features=sensitive_features
    )
    eo_diff = equalized_odds_difference(
        eval_df["Churn"], eval_df["churn_pred"], sensitive_features=sensitive_features
    )
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "recall": recall_score,
            "precision": precision_score,
        },
        y_true=eval_df["Churn"],
        y_pred=eval_df["churn_pred"],
        sensitive_features=sensitive_features,
    )

    fairness_results[sensitive_col] = {
        "demographic_parity_difference": float(dp_diff),
        "equalized_odds_difference": float(eo_diff),
    }
    print(f"\n=== {sensitive_col} ===")
    print(f"Demographic parity difference: {dp_diff:.4f}")
    print(f"Equalized odds difference:     {eo_diff:.4f}")
    print(mf.by_group)

# %% [markdown]
# **Expected pattern** (confirmed in local + prior SageMaker validation of
# this project): `gender` shows a small difference (well under 0.05);
# `SeniorCitizen` shows a substantial one (roughly 0.25–0.31) — the model
# flags senior citizens as high-risk at close to double the rate of
# non-seniors. This is a real, data-driven finding (seniors do churn more
# often and more predictably in this dataset), but it raises a fairness
# question worth deliberate business review before this score drives
# automated retention-offer eligibility. See the project's design document
# Security Checklist for the full discussion.

# %% [markdown]
# ---
# ## 6. Register Model in SageMaker
#
# Create a deployable SageMaker `Model` object directly from the training
# job's artifacts.

# %%
model_name = job_name
print("Model name:", model_name)

info = sm_client.describe_training_job(TrainingJobName=model_name)
model_data = info["ModelArtifacts"]["S3ModelArtifacts"]
print("Model artifact:", model_data)

primary_container = {"Image": image, "ModelDataUrl": model_data}

create_model_response = sm_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=primary_container,
)
print("Model ARN:", create_model_response["ModelArn"])

# %% [markdown]
# #### Inspect Training Job Details

# %%
print(json.dumps(info, indent=2, default=str))

# %% [markdown]
# ---
# ## 7. Model Package Group + Model Package
#
# A **Model Package Group** is a versioned container for all model
# packages of the same problem (churn prediction). A **Model Package**
# documents one specific trained model version within that group.

# %%
model_package_group_name = "xgboost-telco-churn-prediction"

create_group_response = sm_client.create_model_package_group(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageGroupDescription=(
        "XGBoost binary classifier predicting telecom customer churn. "
        "Trained on the IBM/Kaggle Telco Customer Churn dataset. "
        "Output: churn probability (0-1)."
    ),
    Tags=[
        {"Key": "Project", "Value": "telco-churn-prediction"},
        {"Key": "Algorithm", "Value": "XGBoost"},
        {"Key": "Owner", "Value": "Your Name"},
        {"Key": "Dataset", "Value": "IBM-Telco-Customer-Churn"},
    ],
)
print("Model Package Group ARN:", create_group_response["ModelPackageGroupArn"])

# %% [markdown]
# ### describe_model_package_group — screenshot this output for submission

# %%
describe_group_response = sm_client.describe_model_package_group(
    ModelPackageGroupName=model_package_group_name
)
print(json.dumps(describe_group_response, indent=2, default=str))

# %%
model_package_response = sm_client.create_model_package(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageDescription=(
        "XGBoost v1 binary classifier — IBM Telco Customer Churn dataset. "
        "Predicts churn probability. ~45 numeric/one-hot features. "
        "Trained with binary:logistic objective, scale_pos_weight for class "
        "imbalance, 200 rounds."
    ),
    InferenceSpecification={
        "Containers": [
            {
                "Image": image,
                "ModelDataUrl": model_data,
                "Framework": "XGBOOST",
                "FrameworkVersion": "1.7-1",
                "NearestModelName": "xgboost",
            }
        ],
        "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
    },
    ModelApprovalStatus="PendingManualApproval",
    CustomerMetadataProperties={
        "TrainingJobName": job_name,
        "TrainingDataS3": f"s3://{bucket}/{prefix}/train/",
        "ProblemType": "BinaryClassification",
        "TargetColumn": "Churn (1=Yes, 0=No)",
        "EvaluationMetric": "AUC, F1",
        "XGBoostVersion": "1.7-1",
        "FairnessAuditPerformed": "Yes - gender, SeniorCitizen (see Section 5)",
    },
    Tags=[
        {"Key": "Project", "Value": "telco-churn-prediction"},
        {"Key": "Algorithm", "Value": "XGBoost"},
    ],
)

model_package_arn = model_package_response["ModelPackageArn"]
print("Model Package ARN:", model_package_arn)

# %% [markdown]
# ### describe_model_package — screenshot this output for submission

# %%
describe_package_response = sm_client.describe_model_package(
    ModelPackageName=model_package_arn
)
print(json.dumps(describe_package_response, indent=2, default=str))

# %% [markdown]
# ---
# ## 8. Write the Model Card
#
# The Model Card captures qualitative metadata: algorithm, training
# details, evaluation metrics, intended use, and the fairness findings from
# Section 5 — exactly the kind of governance artifact the design
# document's Security Checklist calls for.

# %% [markdown]
# ### 8.1 Build Model Card Content

# %%
model_card_content = {
    "model_overview": {
        "model_description": (
            "XGBoost binary classification model predicting telecom customer churn "
            "risk based on demographic, account, and service-subscription attributes."
        ),
        "model_owner": "Your Name",
        "model_artifact": [model_data],
        "algorithm_type": "XGBoost",
        "problem_type": "BinaryClassification",
        "ml_framework": "XGBoost 1.7-1 (SageMaker built-in)",
    },
    "intended_uses": {
        "purpose_of_model": (
            "Score active customers' churn risk to prioritize proactive retention "
            "outreach with a limited retention budget."
        ),
        "intended_uses": (
            "Batch scoring to rank customers by churn risk for the retention team. "
            "Not intended to make automated retention-offer decisions on its own."
        ),
        "factors_affecting_model_efficiency": (
            "Trained on a single historical snapshot with no time-series component; "
            "performance may degrade as contract mix, pricing, or service offerings "
            "change over time."
        ),
        "risk_rating": "Medium",
        "explanations_for_risk_rating": (
            "Not a safety-critical or medical context, but the fairness audit in "
            "this notebook found a substantial disparity in predicted risk between "
            "senior and non-senior customers, which could translate into disparate "
            "retention outreach if used without review."
        ),
    },
    "business_details": {
        "business_problem": (
            "Reduce customer attrition by directing retention outreach toward "
            "customers most likely to churn, rather than spreading it evenly."
        ),
        "business_stakeholders": "Retention/marketing team, customer success leadership.",
        "line_of_business": "Telecommunications — Subscriber Retention",
    },
    "training_details": {
        "objective_function": "Minimize binary cross-entropy (binary:logistic).",
        "training_observations": (
            f"~{len(data_train)} records (~80% of {len(data_encoded)} total, after "
            "deduplication). ~45 numeric/one-hot features. Target: Churn (1=Yes, 0=No)."
        ),
        "training_job_details": {
            "training_arn": info["TrainingJobArn"],
            "training_datasets": [
                f"s3://{bucket}/{prefix}/train/",
                f"s3://{bucket}/{prefix}/validation/",
            ],
            "training_environment": {"container_image": [image]},
            "hyper_parameters": [
                {"name": "objective", "value": "binary:logistic"},
                {"name": "eval_metric", "value": "auc"},
                {"name": "scale_pos_weight", "value": f"{scale_pos_weight:.4f}"},
                {"name": "max_depth", "value": "6"},
                {"name": "eta", "value": "0.1"},
                {"name": "subsample", "value": "0.8"},
                {"name": "colsample_bytree", "value": "0.8"},
                {"name": "num_round", "value": "200"},
            ],
        },
    },
    "evaluation_details": [
        {
            "name": "Batch Transform Set Performance",
            "evaluation_observation": (
                f"Evaluated on the {len(eval_df)}-row batch holdout via Section 4's "
                "batch transform output, joined back to ground truth in Section 5."
            ),
            "metric_groups": [
                {
                    "name": "Classification Metrics",
                    "metric_data": [
                        {
                            "name": "AUC",
                            "type": "number",
                            "value": float(
                                roc_auc_score(
                                    eval_df["Churn"], eval_df["churn_probability"]
                                )
                            ),
                        },
                    ],
                }
            ],
        },
        {
            "name": "Fairness Audit (Fairlearn)",
            "evaluation_observation": (
                "Demographic parity difference computed across gender and "
                "SeniorCitizen on the same batch holdout. See Section 5 for full results."
            ),
            "metric_groups": [
                {
                    "name": "Fairness Metrics",
                    "metric_data": [
                        {
                            "name": "gender_demographic_parity_difference",
                            "type": "number",
                            "value": fairness_results["gender"][
                                "demographic_parity_difference"
                            ],
                        },
                        {
                            "name": "SeniorCitizen_demographic_parity_difference",
                            "type": "number",
                            "value": fairness_results["SeniorCitizen"][
                                "demographic_parity_difference"
                            ],
                        },
                    ],
                }
            ],
        },
    ],
    "additional_information": {
        "ethical_considerations": (
            "Churn risk scores must not be the sole basis for differential treatment "
            "of customers in a protected class. The SeniorCitizen disparity found in "
            "Section 5 should be reviewed by the business before this score drives "
            "any automated eligibility decision."
        ),
        "caveats_and_recommendations": (
            "Re-evaluate fairness metrics after any retraining. Consider a "
            "fairness-constrained retraining (Fairlearn ExponentiatedGradient or "
            "ThresholdOptimizer) if the SeniorCitizen disparity is judged "
            "unacceptable by stakeholders. Monitor for data drift quarterly."
        ),
        "custom_details": {
            "experiment_log": json.dumps(
                [
                    {
                        "version": "v1",
                        "date": str(datetime.date.today()),
                        "changes": "Baseline XGBoost — max_depth=6, eta=0.1, scale_pos_weight for imbalance",
                        "notes": "Initial model. ~45 features, 80/10/10 split.",
                    }
                ]
            )
        },
    },
}

print("Model card sections:", list(model_card_content.keys()))

# %% [markdown]
# ### 8.2 Create Model Card

# %%
model_card_name = "xgboost-telco-churn-prediction-card"

model_card_response = sm_client.create_model_card(
    ModelCardName=model_card_name,
    Content=json.dumps(model_card_content),
    ModelCardStatus="Draft",
    Tags=[
        {"Key": "Project", "Value": "telco-churn-prediction"},
        {"Key": "Algorithm", "Value": "XGBoost"},
        {"Key": "Owner", "Value": "Your Name"},
    ],
)
print("Model Card ARN:", model_card_response["ModelCardArn"])

# %% [markdown]
# ### describe_model_card — screenshot this output for submission

# %%
describe_card_response = sm_client.describe_model_card(ModelCardName=model_card_name)
print(json.dumps(describe_card_response, indent=2, default=str))

# %% [markdown]
# ### 8.3 Display Parsed Model Card

# %%
card_content = json.loads(describe_card_response["Content"])
for section, details in card_content.items():
    print(f"\n=== {section} ===")
    print(json.dumps(details, indent=2, default=str))

# %% [markdown]
# ---
# ## 9. Real-Time Endpoint Deployment (Optional)
#
# Not required for the batch-oriented design this project documents, but
# included to demonstrate the alternative deployment path. Skip this
# section if you only need batch scoring — remember to run the cleanup
# cell in Section 10 regardless of whether you deploy an endpoint.

# %% [markdown]
# ### 9.1 Create Endpoint Configuration

# %%
endpoint_config_name = "telco-churn-endpoint-config-" + strftime(
    "%Y-%m-%d-%H-%M-%S", gmtime()
)
instance_type = "ml.m5.large"

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
            "ModelName": model_name,
            "VariantName": "AllTraffic",
        }
    ],
)
print("Endpoint config ARN:", create_endpoint_config_response["EndpointConfigArn"])

# %% [markdown]
# ### 9.2 Deploy Endpoint

# %%
endpoint_name = "telco-churn-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
print("Endpoint ARN:", create_endpoint_response["EndpointArn"])

# %% [markdown]
# ### 9.3 Wait for Endpoint to be InService

# %%
while True:
    res = sm_client.describe_endpoint(EndpointName=endpoint_name)
    state = res["EndpointStatus"]
    print(f"Endpoint status: {state}")
    if state in ("InService", "Failed"):
        break
    sleep(30)

# %% [markdown]
# ### 9.4 Invoke Endpoint (Single Record)
#
# Use the first row of the no-ID batch data as a test payload.

# %%
test_payload = data_batch_noID.iloc[0:1].to_csv(header=False, index=False).strip()
print("Payload:", test_payload[:200], "...")

response = sm_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=test_payload,
)
result = response["Body"].read().decode("utf-8")
print("Predicted churn probability:", result)

# %% [markdown]
# ### 9.5 Full Response Object

# %%
print(json.dumps({k: str(v) for k, v in response.items() if k != "Body"}, indent=2))

# %% [markdown]
# ---
# ## 10. Cleanup
#
# > ⚠️ Run this cell to delete all billable resources created in this
# > notebook. Model Registry entries (Model Package Group, Model Package,
# > Model Card) are not deleted here, since they're meant to persist as
# > governance records — delete them manually if this was only a demo run.

# %%
# Delete real-time endpoint (only if Section 9 was run)
try:
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print("Deleted endpoint:", endpoint_name)
except NameError:
    print("No endpoint was created — skipping.")

# Delete endpoint config
try:
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    print("Deleted endpoint config:", endpoint_config_name)
except NameError:
    print("No endpoint config was created — skipping.")

# Delete the SageMaker Model
sm_client.delete_model(ModelName=model_name)
print("Deleted model:", model_name)

print("\nCleanup complete. Model Registry entries and S3 data were left intact.")
