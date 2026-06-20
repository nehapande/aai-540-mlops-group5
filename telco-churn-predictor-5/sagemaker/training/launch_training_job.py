"""
Launches a SageMaker Training job using the built-in XGBoost algorithm
container against the train/validation CSVs produced by
sagemaker/processing/preprocess_job.py.

This mirrors the hyperparameter ranges used in the local train.py XGBoost
candidate, but SageMaker's built-in container takes a single fixed set of
hyperparameters per job rather than a grid -- use
sagemaker/pipeline/build_pipeline.py's HyperparameterTuner step (or run
this script multiple times with different --max-depth/--eta args) to
reproduce the GridSearchCV sweep from the local pipeline.

Run this from a SageMaker Notebook Instance, SageMaker Studio, or any
machine with the `sagemaker` SDK installed and AWS credentials configured
with permission to create Training jobs.

Usage:
    python launch_training_job.py \\
        --bucket my-telco-churn-bucket \\
        --role-arn arn:aws:iam::106122975094:role/SageMakerExecutionRole
"""

from __future__ import annotations

import argparse
from datetime import datetime

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

# Positive-class weight, matching the scale_pos_weight used in the local
# XGBoost candidate (src/train.py): ratio of the negative to positive class
# in the training data. Computed from the known 73.5/26.5 split.
SCALE_POS_WEIGHT = 0.7354 / 0.2646  # ~2.78


def get_job_name(prefix: str = "telco-churn-xgboost") -> str:
    """
    Matches the naming convention SageMaker itself uses for auto-generated
    job/model names, e.g. sagemaker-xgboost-2023-08-22-05-28-37-903.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{prefix}-{timestamp}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket holding processed train/validation CSVs",
    )
    parser.add_argument(
        "--prefix", default="telco-churn", help="S3 key prefix under the bucket"
    )
    parser.add_argument(
        "--role-arn", required=True, help="IAM role ARN with SageMaker + S3 permissions"
    )
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--num-round", type=int, default=200)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    xgboost_image_uri = image_uris.retrieve(
        framework="xgboost",
        region=args.region,
        version="1.7-1",  # current stable built-in XGBoost container as of this writing
    )
    print(f"Using XGBoost container image: {xgboost_image_uri}")

    job_name = get_job_name()
    print(f"Training job name: {job_name}")

    train_input = TrainingInput(
        s3_data=f"s3://{args.bucket}/{args.prefix}/processed/train/train.csv",
        content_type="text/csv",
    )
    validation_input = TrainingInput(
        s3_data=f"s3://{args.bucket}/{args.prefix}/processed/validation/validation.csv",
        content_type="text/csv",
    )

    estimator = Estimator(
        image_uri=xgboost_image_uri,
        role=args.role_arn,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=f"s3://{args.bucket}/{args.prefix}/models/",
        sagemaker_session=sm_session,
        base_job_name="telco-churn-xgboost",
    )

    # Hyperparameters mirror the design document's XGBoost configuration:
    # scale_pos_weight set to the actual class ratio, eval_metric=auc,
    # matching src/train.py's train_xgboost() function.
    estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=SCALE_POS_WEIGHT,
        max_depth=args.max_depth,
        eta=args.eta,
        num_round=args.num_round,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    estimator.fit(
        inputs={"train": train_input, "validation": validation_input},
        job_name=job_name,
        wait=True,
        logs="All",
    )

    print(f"Training complete. Model artifact: {estimator.model_data}")
    return estimator


if __name__ == "__main__":
    main()
