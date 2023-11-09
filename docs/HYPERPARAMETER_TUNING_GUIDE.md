# Hyperparameter Tuning Guide

Hyperparameter tuning is the process of finding the best set of hyperparameters for a machine learning model. This guide will walk you through the steps needed to perform hyperparameter tuning on Google Cloud Vertex AI platform using.

*Note: Please be aware that this notebook is intended for informational purposes only. It provides a concise overview of the steps required to implement hyperparameter tuning. However, you won't be able to start the process directly from this notebook.*

## 1. Prepare model training code
Hyperparameter tuning on Vertex AI passes hyperparameter values to your training application as command-line arguments. Therefore, training code needs to be updated:
1. Define a command-line argument in your main training module for each tuned hyperparameter.
2. Use the value passed in those arguments to set the corresponding hyperparameter in your application's code.
3. Import hypertune library, which is used to define the metric you want to optimize. At the end of the script specify which metric value needs to be reported.

Assuming you have copied and are modifying existing training module code in `model/tuning/tune.py`:
```python
 # Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from pathlib import Path
import json
import joblib
import os
import logging

import hypertune
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from xgboost import XGBRegressor

logging.basicConfig(level=logging.DEBUG)

# used for monitoring during prediction time
TRAINING_DATASET_INFO = "training_dataset.json"
# numeric/categorical features in Chicago trips dataset to be preprocessed
NUM_COLS = ["dayofweek", "hourofday", "trip_distance", "trip_miles", "trip_seconds"]
ORD_COLS = ["company"]
OHE_COLS = ["payment_type"]


def split_xy(df: pd.DataFrame, label: str) -> (pd.DataFrame, pd.Series):
    """Split dataframe into X and y."""
    return df.drop(columns=[label]), df[label]


def indices_in_list(elements: list, base_list: list) -> list:
    """Get indices of specific elements in a base list"""
    return [idx for idx, elem in enumerate(base_list) if elem in elements]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--valid-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument(
        "--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help=""
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of estimators (default: 200)",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=10,
        help="Early stopping rounds (default: 10)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="reg:squarederror",
        help="Objective function (default: reg:squarederror)",
    )
    parser.add_argument(
        "--booster", type=str, default="gbtree", help="Booster type (default: gbtree)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.3, help="Learning rate (default: 0.3)"
    )
    parser.add_argument(
        "--min_split_loss",
        type=float,
        default=0,
        help="Minimum split loss (default: 0)",
    )
    parser.add_argument(
        "--max_depth", type=int, default=6, help="Maximum depth (default: 6)"
    )
    parser.add_argument(
        "--label", type=str, required=True, help="Name of the label column (required)"
    )

    args = parser.parse_args()

    if args.model.startswith("gs://"):
        args.model = "/gcs/" + args.model[5:]

    logging.info("Read csv files into dataframes")
    df_train = pd.read_csv(args.train_data)
    df_valid = pd.read_csv(args.valid_data)
    df_test = pd.read_csv(args.test_data)

    logging.info("Split dataframes")
    label = args.label
    X_train, y_train = split_xy(df_train, label)
    X_valid, y_valid = split_xy(df_valid, label)
    X_test, y_test = split_xy(df_test, label)

    logging.info("Get the number of unique categories for ordinal encoded columns")
    ordinal_columns = X_train[ORD_COLS]
    n_unique_cat = ordinal_columns.nunique()

    logging.info("Get indices of columns in base data")
    col_list = X_train.columns.tolist()
    num_indices = indices_in_list(NUM_COLS, col_list)
    cat_indices_onehot = indices_in_list(OHE_COLS, col_list)
    cat_indices_ordinal = indices_in_list(ORD_COLS, col_list)

    ordinal_transformers = [
        (
            f"ordinal encoding for {ord_col}",
            OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=n_unique_cat[ord_col]
            ),
            [ord_index],
        )
        for ord_col in ORD_COLS
        for ord_index in cat_indices_ordinal
    ]
    all_transformers = [
        ("numeric_scaling", StandardScaler(), num_indices),
        (
            "one_hot_encoding",
            OneHotEncoder(handle_unknown="ignore"),
            cat_indices_onehot,
        ),
    ] + ordinal_transformers

    logging.info("Build sklearn preprocessing steps")
    preprocesser = ColumnTransformer(transformers=all_transformers)
    logging.info("Build sklearn pipeline with XGBoost model")
    xgb_model = XGBRegressor(
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        objective=args.objective,
        booster=args.booster,
        learning_rate=args.learning_rate,
        min_split_loss=args.min_split_loss,
        max_depth=args.max_depth,
    )

    pipeline = Pipeline(
        steps=[("feature_engineering", preprocesser), ("train_model", xgb_model)]
    )

    logging.info("Transform validation data")
    valid_preprocesser = preprocesser.fit(X_train)
    X_valid_transformed = valid_preprocesser.transform(X_valid)

    logging.info("Fit model")
    pipeline.fit(
        X_train, y_train, train_model__eval_set=[(X_valid_transformed, y_valid)]
    )

    logging.info("Predict test data")
    y_pred = pipeline.predict(X_test)
    y_pred = y_pred.clip(0)

    metrics = {
        "problemType": "regression",
        "rootMeanSquaredError": np.sqrt(skmetrics.mean_squared_error(y_test, y_pred)),
        "meanAbsoluteError": skmetrics.mean_absolute_error(y_test, y_pred),
        "meanAbsolutePercentageError": skmetrics.mean_absolute_percentage_error(
            y_test, y_pred
        ),
        "rSquared": skmetrics.r2_score(y_test, y_pred),
        "rootMeanSquaredLogError": np.sqrt(
            skmetrics.mean_squared_log_error(y_test, y_pred)
        ),
    }

    logging.info(f"Save model to: {args.model}")
    Path(args.model).mkdir(parents=True)
    joblib.dump(pipeline, f"{args.model}/model.joblib")

    # Persist URIs of training file(s) for model monitoring in batch predictions
    # See https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig.TrainingDataset  # noqa: E501
    # for the expected schema.
    path = f"{args.model}/{TRAINING_DATASET_INFO}"
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [args.train_data]},
        "dataFormat": "csv",
        "targetField": label,
    }
    logging.info(f"Training dataset info: {training_dataset_for_monitoring}")

    with open(path, "w") as fp:
        logging.info(f"Save training dataset info for model monitoring: {path}")
        json.dump(training_dataset_for_monitoring, fp)

    # DEFINE METRIC
    hp_metric = metrics["rootMeanSquaredError"]

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="rootMeanSquaredError",
        metric_value=hp_metric,
    )


if __name__ == "__main__":
    main()
```

## 2. Prepare Tuning Container
To execute hyperparameter tuning, your code needs to be containerized and include the `cloudml-hypertune` Python package. This package is used to pass metrics to Vertex AI. You can achieve this by updating either the `Dockerfile` or `pyproject.toml`. Also make sure your tuning container contains training module.

To build the container and upload it to arfifact registry you can use `make build` command or use any other alternative methods.

Example of updated Dockerfile:
```Dockerfile
FROM python:3.9.16-slim AS builder

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

ARG POETRY_VERSION=1.5.1

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN pip install poetry==${POETRY_VERSION}
RUN poetry config virtualenvs.create false && poetry install

FROM builder AS training

COPY training/train.py training/train.py

FROM builder as tuning
RUN pip install cloudml-hypertune
COPY tuning/tune.py tuning/tune.py

FROM builder AS serving

RUN poetry install --with serving
COPY serving/main.py serving/main.py

CMD exec uvicorn serving.main:app --host "0.0.0.0" --port "$AIP_HTTP_PORT"

```

## 3. Create Hyperparameter tuning pipeline
Now we need to create Hyperparameter Tuning Pipeline. To simplify development we will reuse `HyperparameterTuningJobRunOp` component from `google_cloud_pipeline_components` library. To extract the results, we are introducing custom components: `GetTrialsOp` and `GetBestTrialOps`.

`HyperparameterTuningJobRunOp` does not inherently support input data as an argument. To address this, an additional component needs to be created. Given that our training module accepts data via command line arguments, this can be accomplished through the use of `worker_pool_specs`, which contains the hyperparameter tuning container specification.

To define metrics and parameters to optimize use `hyperparameter_tuning_job.serialize_metrics` and `hyperparameter_tuning_job.serialize_parameters` retrospectively. Please refer to example pipeline code (*HYPERPARAMETER TUNING ARGS SECTION*).

Example hyperparameter tuning pipeline: `pipelines/tuning/pipeline.py`
```python
import os
import pathlib

from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import (
    HyperparameterTuningJobRunOp, 
)
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google_cloud_pipeline_components.v1 import hyperparameter_tuning_job
from kfp import dsl
from kfp.dsl import Dataset, Input, PIPELINE_JOB_CREATE_TIME_UTC_PLACEHOLDER
from pipelines import generate_query
from bigquery_components import extract_bq_to_dataset

CONTAINER_IMAGE_REGISTRY = os.environ["CONTAINER_IMAGE_REGISTRY"]
VERTEX_PIPELINE_ROOT = os.environ["VERTEX_PIPELINE_ROOT"]
RESOURCE_SUFFIX = os.environ.get("RESOURCE_SUFFIX", "default")
TUNING_IMAGE = f"{CONTAINER_IMAGE_REGISTRY}/tuning:{RESOURCE_SUFFIX}"


@dsl.component(base_image="python:3.9")
def worker_pool_specs(
    train_data: Input[Dataset],
    valid_data: Input[Dataset],
    test_data: Input[Dataset],
    tuning_container_image: str,
    hparams: dict,
) -> list:
    """
    Generate a specification for worker pools to perform hyperparameter tuning..

    Args:
        train_data (Input[Dataset]): Input dataset for training.
        valid_data (Input[Dataset]): Input dataset for validation.
        test_data (Input[Dataset]): Input dataset for testing.
        tuning_container_image (str): Container image to be used for the tuning job.
        hparams (dict): A dictionary containing hyperparameter key-value pairs.

    Returns:
        list: A list of worker pool specifications, each specifying the machine type,
        Docker image, and command arguments for a worker pool.

    Example:
    worker_pool_specs(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        tuning_container_image="gcr.io/my-project/tuning-image:latest",
        hparams={"learning_rate": 0.001, "batch_size": 64}
    )

    The function returns a list containing a single worker pool specification.

    Note:
    The worker pool specification is a data structure used to define the resources and
    configurations needed for running distributed training or hyperparameter tuning jobs
    in a managed AI platform.

    For each hyperparameter combination to be tuned, you may need to create a separate
    worker pool specification with appropriate command arguments which depends on your
    tuning training script.
    """

    CMDARGS = [
        "tuning/tune.py",
        "--train-data",
        train_data.path,
        "--valid-data",
        valid_data.path,
        "--test-data",
        test_data.path,
    ]

    for key, value in hparams.items():
        CMDARGS.extend(["--" + str(key), str(value)])

    # The spec of the worker pools including machine type and Docker image
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": tuning_container_image,
                "command": ["python"],
                "args": CMDARGS,
            },
        }
    ]
    return worker_pool_specs


@dsl.component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-pipeline-components",
        "protobuf",
    ],
    base_image="python:3.9",
)
def GetTrialsOp(gcp_resources: str) -> list:
    """Retrieves the best trial from the trials.

    Args:
        gcp_resources (str): Proto tracking the hyperparameter tuning job.

    Returns:
        List of strings representing the intermediate JSON representation of the
        trials from the hyperparameter tuning job.
    """
    from google.cloud import aiplatform
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources
    from google.protobuf.json_format import Parse
    from google.cloud.aiplatform_v1.types import study

    api_endpoint_suffix = "-aiplatform.googleapis.com"
    gcp_resources_proto = Parse(gcp_resources, GcpResources())
    gcp_resources_split = gcp_resources_proto.resources[0].resource_uri.partition(
        "projects"
    )
    resource_name = gcp_resources_split[1] + gcp_resources_split[2]
    prefix_str = gcp_resources_split[0]
    prefix_str = prefix_str[: prefix_str.find(api_endpoint_suffix)]
    api_endpoint = prefix_str[(prefix_str.rfind("//") + 2) :] + api_endpoint_suffix

    client_options = {"api_endpoint": api_endpoint}
    job_client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    response = job_client.get_hyperparameter_tuning_job(name=resource_name)

    return [study.Trial.to_json(trial) for trial in response.trials]


@dsl.component(packages_to_install=["google-cloud-aiplatform"], base_image="python:3.9")
def GetBestTrialOp(trials: list, study_spec_metrics: list) -> str:
    """Retrieves the best trial from the trials.

    Args:
        trials (list): Required. List representing the intermediate
          JSON representation of the trials from the hyperparameter tuning job.
        study_spec_metrics (list): Required. List serialized from dictionary
          representing the metrics to optimize.
          The dictionary key is the metric_id, which is reported by your training
          job, and the dictionary value is the optimization goal of the metric
          ('minimize' or 'maximize'). example:
          metrics = hyperparameter_tuning_job.serialize_metrics(
              {'loss': 'minimize', 'accuracy': 'maximize'})

    Returns:
        String representing the intermediate JSON representation of the best
        trial from the list of trials.

    Raises:
        RuntimeError: If there are multiple metrics.
    """
    from google.cloud.aiplatform_v1.types import study

    if len(study_spec_metrics) > 1:
        raise RuntimeError(
            "Unable to determine best parameters for multi-objective"
            " hyperparameter tuning."
        )
    trials_list = [study.Trial.from_json(trial) for trial in trials]
    best_trial = None
    goal = study_spec_metrics[0]["goal"]
    best_fn = None
    if goal == study.StudySpec.MetricSpec.GoalType.MAXIMIZE:
        best_fn = max
    elif goal == study.StudySpec.MetricSpec.GoalType.MINIMIZE:
        best_fn = min
    best_trial = best_fn(
        trials_list, key=lambda trial: trial.final_measurement.metrics[0].value
    )

    return study.Trial.to_json(best_trial)


@dsl.pipeline(name="hyperparameter-tuning-pipeline")
def pipeline(
    project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    project_location: str = os.environ.get("VERTEX_LOCATION"),
    ingestion_project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    dataset_id: str = "preprocessing",
    dataset_location: str = os.environ.get("VERTEX_LOCATION"),
    ingestion_dataset_id: str = "chicago_taxi_trips",
    timestamp: str = "2022-12-01 00:00:00",
    resource_suffix: str = os.environ.get("RESOURCE_SUFFIX"),
    test_dataset_uri: str = "",
):
    """
    XGB hyperparametur tuning pipeline which.
    1. Splits and extracts a dataset from BQ to GCS
    2. Execute hyperparameter tuning to optimize the XGBoost model.
    3. Retrieve and record the best hyperparameters from the tuning job.

    Args:
        project_id (str): project id of the Google Cloud project
        project_location (str): location of the Google Cloud project
        ingestion_project_id (str): project id containing the source bigquery data
            for ingestion. This can be the same as `project_id` if the source data is
            in the same project where the ML pipeline is executed.
        dataset_id (str): id of BQ dataset used to store all staging data & predictions
        dataset_location (str): location of dataset
        ingestion_dataset_id (str): dataset id of ingestion data
        timestamp (str): Optional. Empty or a specific timestamp in ISO 8601 format
            (YYYY-MM-DDThh:mm:ss.sss±hh:mm or YYYY-MM-DDThh:mm:ss).
            If any time part is missing, it will be regarded as zero.
        resource_suffix (str): Optional. Additional suffix to append GCS resources
            that get overwritten.
        test_dataset_uri (str): Optional. GCS URI of held-out test dataset.
    """

    # Create variables to ensure the same arguments are passed
    # into different components of the pipeline
    label_column_name = "total_fare"
    time_column = "trip_start_timestamp"
    ingestion_table = "taxi_trips"
    table_suffix = f"_xgb_training_{resource_suffix}"  # suffix to table names
    ingested_table = "ingested_data" + table_suffix
    preprocessed_table = "preprocessed_data" + table_suffix
    train_table = "train_data" + table_suffix
    valid_table = "valid_data" + table_suffix
    test_table = "test_data" + table_suffix
    primary_metric = "rootMeanSquaredError"
    # --------------------------------------
    # HYPERPARAMETER TUNING ARGS
    # --------------------------------------
    hparams = dict(
        n_estimators=200,
        early_stopping_rounds=10,
        objective="reg:squarederror",
        booster="gbtree",
        # learning_rate=0.3, Hyperparameter to optimise
        min_split_loss=0,
        max_depth=6,
        label=label_column_name,
    )
    # List serialized from the dictionary representing metrics to optimize.
    # The dictionary key is the metric_id, which is reported by your training job,
    # and the dictionary value is the optimization goal of the metric.
    study_spec_metrics = hyperparameter_tuning_job.serialize_metrics(
        {"rootMeanSquaredError": "minimize"}
    )

    # List serialized from the parameter dictionary. The dictionary
    # represents parameters to optimize. The dictionary key is the parameter_id,
    # which is passed into your training job as a command line key word argument,and the
    # dictionary value is the parameter specification of the metric.
    study_spec_parameters = hyperparameter_tuning_job.serialize_parameters(
        {
            "learning_rate": hpt.DoubleParameterSpec(min=0.001, max=1, scale="log"),
        }
    )

    max_trial_count = 3
    parallel_trial_count = 3
    base_output_directory = (
        f"{VERTEX_PIPELINE_ROOT}/hpt-{PIPELINE_JOB_CREATE_TIME_UTC_PLACEHOLDER}"
    )
    display_name = f"hpt-{PIPELINE_JOB_CREATE_TIME_UTC_PLACEHOLDER}"
    study_spec_algorithm = "ALGORITHM_UNSPECIFIED"
    study_spec_measurement_selection_type = "BEST_MEASUREMENT"
    # --------------------------------------

    # generate sql queries which are used in ingestion and preprocessing
    # operations

    queries_folder = pathlib.Path(__file__).parent.parent / "training" / "queries"

    preprocessing_query = generate_query(
        queries_folder / "preprocessing.sql",
        source_dataset=f"{ingestion_project_id}.{ingestion_dataset_id}",
        source_table=ingestion_table,
        preprocessing_dataset=f"{ingestion_project_id}.{dataset_id}",
        ingested_table=ingested_table,
        dataset_region=project_location,
        filter_column=time_column,
        target_column=label_column_name,
        filter_start_value=timestamp,
        train_table=train_table,
        validation_table=valid_table,
        test_table=test_table,
    )

    preprocessing = (
        BigqueryQueryJobOp(
            project=project_id,
            location=dataset_location,
            query=preprocessing_query,
        )
        .set_caching_options(False)
        .set_display_name("Ingest & preprocess data")
    )

    # data extraction to gcs

    train_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=train_table,
            dataset_location=dataset_location,
        )
        .after(preprocessing)
        .set_display_name("Extract train data")
        .set_caching_options(False)
    ).outputs["dataset"]
    valid_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=valid_table,
            dataset_location=dataset_location,
        )
        .after(preprocessing)
        .set_display_name("Extract validation data")
        .set_caching_options(False)
    ).outputs["dataset"]
    test_dataset = (
        extract_bq_to_dataset(
            bq_client_project_id=project_id,
            source_project_id=project_id,
            dataset_id=dataset_id,
            table_name=test_table,
            dataset_location=dataset_location,
            destination_gcs_uri=test_dataset_uri,
        )
        .after(preprocessing)
        .set_display_name("Extract test data")
        .set_caching_options(False)
    ).outputs["dataset"]

    worker_pool = worker_pool_specs(
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        hparams=hparams,
        tuning_container_image=TUNING_IMAGE,
    ).set_display_name("Worker Pool Specs")
    tuning = HyperparameterTuningJobRunOp(
        display_name=display_name,
        project=project_id,
        location=project_location,
        worker_pool_specs=worker_pool.output,
        study_spec_metrics=study_spec_metrics,
        study_spec_parameters=study_spec_parameters,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        base_output_directory=base_output_directory,
        study_spec_algorithm=study_spec_algorithm,
        study_spec_measurement_selection_type=study_spec_measurement_selection_type,
    )

    trials = GetTrialsOp(gcp_resources=tuning.outputs["gcp_resources"])

    best_trial = GetBestTrialOp(
        trials=trials.output, study_spec_metrics=study_spec_metrics
    )
```
## 4. Trigger Hyperparameter tuning pipeline
If your pipeline is defined in `pipelines/tuning/pipeline.py`, you should be able to trigger the hyperparameter tuning pipeline by executing make run pipeline/tuning. Otherwise, use any other alternative methods.
