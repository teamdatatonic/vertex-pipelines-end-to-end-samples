# Copyright 2022 Google LLC
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
