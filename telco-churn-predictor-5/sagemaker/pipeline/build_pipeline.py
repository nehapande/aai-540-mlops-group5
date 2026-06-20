"""
SageMaker Pipeline definition for the Telco Churn project.

Wires together the same stages as the local GitHub Actions workflow
(.github/workflows/ml-pipeline.yml), as managed SageMaker Pipeline steps:

    1. ProcessingStep   -- runs sagemaker/processing/preprocess_job.py
    2. TrainingStep     -- trains the built-in XGBoost algorithm
    3. ProcessingStep   -- evaluates the trained model against the test set
    4. ConditionStep    -- the AUC-ROC >= 0.80 gate from the design document;
                           only proceeds to registration if the condition passes
    5. RegisterModel    -- registers the approved model in the SageMaker
                           Model Registry (model package group), analogous
                           to the "promote to production" step in CI/CD
    6. TransformStep     -- runs a Batch Transform job against the held-out
                           test set, matching the design document's batch
                           inference deployment pattern

This script DEFINES the pipeline; it does not execute it on its own.
Run it from a SageMaker Notebook Instance / Studio / any machine with AWS
credentials and the `sagemaker` SDK installed:

    python build_pipeline.py --bucket my-bucket --role-arn arn:aws:iam::...

That prints the pipeline ARN and starts an execution. Subsequent runs can
call pipeline.start() directly without redefining the pipeline.
"""

from __future__ import annotations

import argparse

import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TransformStep,
    CreateModelStep,
)
from sagemaker.transformer import Transformer

PIPELINE_NAME = "TelcoChurnPipeline"
MODEL_PACKAGE_GROUP_NAME = "telco-churn-xgboost-models"
SCALE_POS_WEIGHT = 0.7354 / 0.2646  # matches src/train.py's class-ratio calculation


def build_pipeline(
    bucket: str,
    role_arn: str,
    region: str = "us-east-1",
    pipeline_session: PipelineSession | None = None,
) -> Pipeline:
    sm_session = pipeline_session or PipelineSession(
        boto_session=sagemaker.Session().boto_session
    )

    # ---- Pipeline parameters (overridable per-execution without redefining the pipeline) ----
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{bucket}/telco-churn/raw/telco_churn.csv",
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    min_auc_threshold = ParameterFloat(name="MinAucRocThreshold", default_value=0.80)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    # ---- Step 1: Processing (preprocess_job.py) ----
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        role=role_arn,
        sagemaker_session=sm_session,
        base_job_name="telco-churn-preprocess",
    )

    processing_step = ProcessingStep(
        name="PreprocessTelcoChurnData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data_uri, destination="/opt/ml/processing/input"
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train", source="/opt/ml/processing/output/train"
            ),
            ProcessingOutput(
                output_name="validation", source="/opt/ml/processing/output/validation"
            ),
            ProcessingOutput(
                output_name="test", source="/opt/ml/processing/output/test"
            ),
        ],
        code="../processing/preprocess_job.py",
    )

    # ---- Step 2: Training (built-in XGBoost container) ----
    xgboost_image_uri = image_uris.retrieve(
        framework="xgboost", region=region, version="1.7-1"
    )

    xgb_estimator = Estimator(
        image_uri=xgboost_image_uri,
        role=role_arn,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"s3://{bucket}/telco-churn/models/",
        sagemaker_session=sm_session,
        base_job_name="telco-churn-xgboost",
    )
    xgb_estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=SCALE_POS_WEIGHT,
        max_depth=6,
        eta=0.1,
        num_round=200,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    training_step = TrainingStep(
        name="TrainXGBoostChurnModel",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ---- Step 3: Evaluation (custom script, computes AUC-ROC on the test set) ----
    # Reuses the same built-in XGBoost container as the training step: it
    # ships with pandas, scikit-learn, and xgboost pre-installed, which is
    # everything evaluate_job.py needs, and avoids maintaining a second
    # custom container image just for evaluation.
    evaluation_processor = ScriptProcessor(
        image_uri=xgboost_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        role=role_arn,
        sagemaker_session=sm_session,
        base_job_name="telco-churn-eval",
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    evaluation_step = ProcessingStep(
        name="EvaluateChurnModel",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
        code="../processing/evaluate_job.py",
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    evaluation_step.properties.ProcessingOutputConfig.Outputs[
                        "evaluation"
                    ].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    # ---- Step 3b: Fairness audit (Fairlearn, gender + SeniorCitizen) ----
    # Mirrors src/fairness_audit.py. Runs unconditionally, in parallel with
    # the AUC-ROC gate below -- like the GitHub Actions pipeline, this is
    # treated as informational rather than blocking, since the local
    # pipeline's audit surfaced a real, unresolved disparity finding on
    # SeniorCitizen that warrants human review rather than an automated
    # pass/fail. Its output is uploaded as a tracked artifact either way,
    # so a reviewer sees it on every run regardless of the AUC-ROC outcome.
    fairness_processor = ScriptProcessor(
        image_uri=xgboost_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        role=role_arn,
        sagemaker_session=sm_session,
        base_job_name="telco-churn-fairness",
    )

    fairness_step = ProcessingStep(
        name="FairnessAuditChurnModel",
        processor=fairness_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="fairness", source="/opt/ml/processing/fairness"
            ),
        ],
        code="../processing/fairness_audit_job.py",
    )

    # ---- Step 4: Register the model (only reached if the condition step below passes) ----
    register_step = RegisterModel(
        name="RegisterChurnModel",
        estimator=xgb_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large", "ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # ---- Step 5: Batch Transform against the test set (matches the design
    #              document's batch-inference deployment decision) ----
    # A Transformer needs a deployable SageMaker Model resource (not the
    # Model Registry entry from RegisterModel above, which is a separate
    # governance artifact) -- so we explicitly create one from the same
    # training artifact via CreateModelStep.
    inference_model = Model(
        image_uri=xgboost_image_uri,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        sagemaker_session=sm_session,
    )
    create_model_step = CreateModelStep(
        name="CreateChurnModelForTransform",
        model=inference_model,
    )

    transformer = Transformer(
        model_name=create_model_step.properties.ModelName,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=f"s3://{bucket}/telco-churn/batch-output/",
        sagemaker_session=sm_session,
    )
    transform_step = TransformStep(
        name="BatchScoreTestSet",
        transformer=transformer,
        inputs=sagemaker.inputs.TransformInput(
            data=processing_step.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    )

    # ---- Condition: AUC-ROC >= 0.80, matching the design document's gate ----
    auc_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.auc.value",
        ),
        right=min_auc_threshold,
    )

    condition_step = ConditionStep(
        name="CheckAucRocThreshold",
        conditions=[auc_condition],
        if_steps=[register_step, create_model_step, transform_step],
        else_steps=[],  # model is silently dropped if it fails the gate; a
        # production version would add an SNS notification step here
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            input_data_uri,
            processing_instance_type,
            training_instance_type,
            min_auc_threshold,
            model_approval_status,
        ],
        steps=[
            processing_step,
            training_step,
            evaluation_step,
            fairness_step,
            condition_step,
        ],
        sagemaker_session=sm_session,
    )
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--upsert-only",
        action="store_true",
        help="Define/update the pipeline without starting an execution",
    )
    args = parser.parse_args()

    pipeline = build_pipeline(
        bucket=args.bucket, role_arn=args.role_arn, region=args.region
    )
    pipeline.upsert(role_arn=args.role_arn)
    print(f"Pipeline '{PIPELINE_NAME}' created/updated.")
    print(f"Pipeline ARN: {pipeline.describe()['PipelineArn']}")

    if not args.upsert_only:
        execution = pipeline.start()
        print(f"Started execution: {execution.arn}")


if __name__ == "__main__":
    main()
