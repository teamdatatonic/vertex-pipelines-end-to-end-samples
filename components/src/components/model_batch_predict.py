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

from kfp.dsl import Input, Model, component, OutputPath
from typing import List


@component(
    base_image="python:3.12.8",
    packages_to_install=[
        "google-cloud-aiplatform==1.135.0",
        "google-cloud-pipeline-components==2.22.0",
    ],
)
def model_batch_predict(
    model: Input[Model],
    gcp_resources: OutputPath(str),
    job_display_name: str,
    location: str,
    project: str,
    source_uri: str,
    destination_uri: str,
    source_format: str,
    destination_format: str,
    machine_type: str = "n1-standard-2",
    starting_replica_count: int = 1,
    max_replica_count: int = 1,
    instance_config: dict = None,
    enable_monitoring: bool = True,
    monitoring_training_gcs_uri: str = "",
    monitored_features: dict = {},
    default_drift_threshold: float = 0.3,
    notification_emails: List[str] = [],
    enable_cloud_logging: bool = True,
):
    """
    Trigger a batch prediction job and, once it succeeds, run Model Monitoring v2
    (feature drift) against its output.

    Args:
        model (Input[Model]): Input model to use for calculating predictions.
        job_display_name: Name of the batch prediction job.
        location (str): location of the Google Cloud project. Defaults to None.
        project (str): project id of the Google Cloud project. Defaults to None.
        source_uri (str): bq:// URI or a list of gcs:// URIs to read input instances.
        destination_uri (str): bq:// or gs:// URI to store output predictions.
        source_format (str): E.g. "bigquery", "jsonl", "csv". See:
            https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1beta1.types.BatchPredictionJob.InputConfig
        destination_format (str): E.g. "bigquery", "jsonl", "csv". See:
            https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1beta1.types.BatchPredictionJob.OutputConfig
        machine_type (str): Machine type.
        starting_replica_count (int): Starting replica count.
        max_replica_count (int): Max replicat count.
        instance_config (dict): Configuration defining how to transform batch prediction
            input instances to the instances that the Model accepts. See:
            https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/projects.locations.batchPredictionJobs#instanceconfig
        enable_monitoring (bool): Whether to run Model Monitoring v2 (feature drift)
            against the batch job's output once it succeeds. Requires
            monitoring_training_gcs_uri and monitored_features to be set; otherwise
            monitoring is skipped with a warning.
        monitoring_training_gcs_uri (str): GCS URI (CSV) of the training data used as
            the drift-detection baseline, e.g. the value produced by the
            lookup_model component's training_dataset_gcs_uri output.
        monitored_features (dict): Maps each feature name to its Model Monitoring v2
            schema data type. Supported values: "float", "integer", "boolean",
            "string", "categorical". E.g. {"trip_miles": "float", "company":
            "categorical"}. See:
            https://cloud.google.com/vertex-ai/docs/model-monitoring/model-monitoring-overview
        default_drift_threshold (float): Alert threshold applied to every monitored
            feature (categorical and numeric) that doesn't have a more specific
            threshold configured.
        notification_emails (List[str]): Email addresses to notify when a drift
            alert fires (optional).
        enable_cloud_logging (bool): Whether Model Monitoring alerts are also written
            to Cloud Logging, independent of `notification_emails`.
    Returns:
        OutputPath: gcp_resources for Vertex AI UI integration.
    """

    import logging
    import re
    import time
    import uuid

    from functools import partial
    from google.protobuf.json_format import ParseDict, MessageToJson
    from google.cloud.aiplatform_v1beta1.services.job_service import JobServiceClient
    from google.cloud.aiplatform_v1beta1.types import (
        BatchPredictionJob,
        GetBatchPredictionJobRequest,
    )
    from google.cloud.aiplatform_v1beta1.types.job_state import JobState
    from google_cloud_pipeline_components.container.v1.gcp_launcher.utils import (
        error_util,
    )
    from google_cloud_pipeline_components.container.utils import execution_context
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    def _slugify(name: str, max_length: int = 63) -> str:
        """Make `name` a valid Model Monitoring resource ID.

        Must match ``^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$``.
        """
        slug = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
        slug = re.sub(r"-{2,}", "-", slug)
        if not slug or not slug[0].isalpha():
            slug = f"m-{slug}"
        return slug[:max_length].rstrip("-")

    def send_cancel_request(client: JobServiceClient, batch_job_uri: str):
        logging.info("Sending BatchPredictionJob cancel request")
        client.cancel_batch_prediction_job(name=batch_job_uri)

    def is_job_successful(job_state: JobState) -> bool:
        _JOB_SUCCESSFUL_STATES = [
            JobState.JOB_STATE_SUCCEEDED,
        ]
        _JOB_FAILED_STATES = [
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_EXPIRED,
        ]

        if job_state in _JOB_SUCCESSFUL_STATES:
            logging.info(
                f"GetBatchPredictionJobRequest response state={job_state}. "
                "Job completed"
            )
            return True
        elif job_state in _JOB_FAILED_STATES:
            raise RuntimeError(
                "Job {} failed with error state: {}.".format(response.name, job_state)
            )
        else:
            logging.info(f"Job {response.name} is in a non-final state {job_state}.")
        return False

    _POLLING_INTERVAL_IN_SECONDS = 20
    _CONNECTION_ERROR_RETRY_LIMIT = 5

    api_endpoint = f"{location}-aiplatform.googleapis.com"

    input_config = {"instancesFormat": source_format}
    output_config = {"predictionsFormat": destination_format}
    if source_format == "bigquery" and destination_format == "bigquery":
        input_config["bigquerySource"] = {"inputUri": source_uri}
        output_config["bigqueryDestination"] = {"outputUri": destination_uri}
    else:
        input_config["gcsSource"] = {"uris": [source_uri]}
        output_config["gcsDestination"] = {"outputUriPrefix": destination_uri}

    message = {
        "displayName": job_display_name,
        "model": model.metadata["resourceName"],
        "inputConfig": input_config,
        "outputConfig": output_config,
        "dedicatedResources": {
            "machineSpec": {"machineType": machine_type},
            "startingReplicaCount": starting_replica_count,
            "maxReplicaCount": max_replica_count,
        },
    }

    if instance_config:
        message["instanceConfig"] = instance_config

    request = ParseDict(message, BatchPredictionJob()._pb)

    logging.info(f"Submitting batch prediction job: {job_display_name}")
    logging.info(request)
    client = JobServiceClient(client_options={"api_endpoint": api_endpoint})
    response = client.create_batch_prediction_job(
        parent=f"projects/{project}/locations/{location}",
        batch_prediction_job=request,
    )
    logging.info(f"Submitted batch prediction job: {response.name}")

    # output GCP resource for Vertex AI UI integration
    batch_job_resources = GcpResources()
    dr = batch_job_resources.resources.add()
    dr.resource_type = "BatchPredictionJob"
    dr.resource_uri = response.name
    with open(gcp_resources, "w") as f:
        f.write(MessageToJson(batch_job_resources))

    with execution_context.ExecutionContext(
        on_cancel=partial(
            send_cancel_request,
            api_endpoint,
            response.name,
        )
    ):
        retry_count = 0
        while True:
            try:
                job_status_request = GetBatchPredictionJobRequest(
                    {"name": response.name}
                )
                job_state = client.get_batch_prediction_job(
                    request=job_status_request
                ).state
                retry_count = 0
            except ConnectionError as err:
                retry_count += 1
                if retry_count <= _CONNECTION_ERROR_RETRY_LIMIT:
                    logging.warning(
                        f"ConnectionError ({err}) encountered when polling job: "
                        f"{response.name}. Retrying."
                    )
                else:
                    error_util.exit_with_internal_error(
                        f"Request failed after {_CONNECTION_ERROR_RETRY_LIMIT} retries."
                    )
            if is_job_successful(job_state):
                break
            logging.info(
                f"Waiting for {_POLLING_INTERVAL_IN_SECONDS} seconds for next poll."
            )
            time.sleep(_POLLING_INTERVAL_IN_SECONDS)

        if not enable_monitoring:
            logging.info("enable_monitoring is False, skipping Model Monitoring v2")
        elif not monitoring_training_gcs_uri or not monitored_features:
            logging.warning(
                "enable_monitoring is True but monitoring_training_gcs_uri or "
                "monitored_features was not provided; skipping Model Monitoring v2 "
                "setup for this batch job."
            )
        else:
            from vertexai.resources.preview import ml_monitoring
            from vertexai.resources.preview.ml_monitoring.spec import (
                notification as notif_spec,
                objective,
                schema as schema_spec,
            )
            from google.cloud.aiplatform_v1beta1.services.model_monitoring_service import (  # noqa: E501
                ModelMonitoringServiceClient,
            )
            from google.cloud.aiplatform_v1beta1.types import (
                model_monitoring_job as gca_model_monitoring_job,
                model_monitoring_service as gca_model_monitoring_service,
                model_monitoring_spec as gca_model_monitoring_spec,
            )

            resource_name = model.metadata["resourceName"]
            if "@" in resource_name:
                monitor_model_name, monitor_model_version_id = resource_name.rsplit(
                    "@", 1
                )
            else:
                monitor_model_name, monitor_model_version_id = resource_name, "1"

            model_monitoring_schema = schema_spec.ModelMonitoringSchema(
                feature_fields=[
                    schema_spec.FieldSchema(name=name, data_type=data_type)
                    for name, data_type in monitored_features.items()
                ]
            )
            baseline_dataset = objective.MonitoringInput(
                gcs_uri=monitoring_training_gcs_uri, data_format="csv"
            )
            target_dataset = objective.MonitoringInput(
                batch_prediction_job=response.name
            )
            feature_drift_spec = objective.DataDriftSpec(
                categorical_metric_type="l_infinity",
                numeric_metric_type="jensen_shannon_divergence",
                default_categorical_alert_threshold=default_drift_threshold,
                default_numeric_alert_threshold=default_drift_threshold,
                feature_alert_thresholds={
                    name: default_drift_threshold for name in monitored_features
                },
            )
            tabular_objective_spec = objective.TabularObjective(
                feature_drift_spec=feature_drift_spec
            )
            notification_spec = None
            if notification_emails or enable_cloud_logging:
                notification_spec = notif_spec.NotificationSpec(
                    user_emails=notification_emails,
                    enable_cloud_logging=enable_cloud_logging,
                )

            monitor_display_name = f"{job_display_name}-monitoring"
            existing_monitors = ml_monitoring.ModelMonitor.list(
                project=project,
                location=location,
                filter=f'display_name="{monitor_display_name}"',
            )
            if existing_monitors:
                logging.info(
                    f"Reusing existing ModelMonitor: "
                    f"{existing_monitors[0].resource_name}"
                )
                monitor = existing_monitors[0]
            else:
                logging.info(f"Creating new ModelMonitor: {monitor_display_name}")
                monitor = ml_monitoring.ModelMonitor.create(
                    project=project,
                    location=location,
                    display_name=monitor_display_name,
                    model_name=monitor_model_name,
                    model_version_id=monitor_model_version_id,
                    training_dataset=baseline_dataset,
                    model_monitoring_schema=model_monitoring_schema,
                )

            # `ModelMonitor.run()` / `ModelMonitoringJob.create()` (used by the
            # vertexai.resources.preview.ml_monitoring SDK) unconditionally
            # call `model_monitoring_job._block_until_complete()` after
            # submitting the job - this happens regardless of `sync`; `sync`
            # only controls whether that blocking wait happens on the
            # calling thread (sync=True) or on a background thread from
            # google.cloud.aiplatform's shared ThreadPoolExecutor
            # (sync=False). Either way, this component's container previously
            # ended up waiting for the *entire* monitoring computation to
            # finish (10-30+ minutes) before it could exit, because Python's
            # interpreter shutdown always joins that pool's non-daemon
            # threads (see `concurrent.futures.thread`, which registers via
            # `threading._register_atexit` - this runs *before* normal
            # `atexit.register()` callbacks, so registering our own atexit
            # hook to force-exit earlier does not help either).
            #
            # Drift detection is an observability side effect of the batch
            # job, not something the pipeline's critical path should block
            # on. To get genuine fire-and-forget submission, bypass the
            # preview SDK's ModelMonitoringJob.create()/run() entirely and
            # call the underlying GAPIC RPC directly - like the rest of this
            # component already does for the BatchPredictionJob itself. This
            # is a plain, synchronous unary call: it returns as soon as the
            # job resource is accepted, with no local polling loop at all.
            monitoring_job_id = _slugify(
                f"{job_display_name}-mon-{uuid.uuid4().hex[:8]}"
            )
            monitoring_job_request = gca_model_monitoring_job.ModelMonitoringJob(
                display_name=f"{job_display_name}-monitoring-run",
                model_monitoring_spec=gca_model_monitoring_spec.ModelMonitoringSpec(
                    objective_spec=gca_model_monitoring_spec.ModelMonitoringObjectiveSpec(  # noqa: E501
                        tabular_objective=tabular_objective_spec._as_proto(),
                        baseline_dataset=baseline_dataset._as_proto(),
                        target_dataset=target_dataset._as_proto(),
                    ),
                    notification_spec=(
                        notification_spec._as_proto() if notification_spec else None
                    ),
                ),
            )
            monitoring_client = ModelMonitoringServiceClient(
                client_options={"api_endpoint": api_endpoint}
            )
            created_monitoring_job = monitoring_client.create_model_monitoring_job(
                request=gca_model_monitoring_service.CreateModelMonitoringJobRequest(
                    parent=monitor.resource_name,
                    model_monitoring_job=monitoring_job_request,
                    model_monitoring_job_id=monitoring_job_id,
                )
            )
            logging.info(
                "Model Monitoring v2 job submitted (fire-and-forget): "
                f"{created_monitoring_job.name}"
            )
