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

from typing import List

from kfp.dsl import component


@component(
    base_image="python:3.12.8",
    packages_to_install=["google-cloud-aiplatform==1.135.0"],
)
def run_model_monitoring_job(
    project: str,
    location: str,
    model_monitor_name: str,
    training_dataset_gcs_uri: str,
    target_bq_table_uri: str,
    job_display_name: str,
    monitored_features: dict,
    default_drift_threshold: float = 0.3,
    notification_emails: List[str] = None,
    enable_cloud_logging: bool = True,
) -> None:
    """
    Submit a Model Monitoring v2 job (fire-and-forget) against logged endpoint
    traffic.

    Uses the GAPIC ``create_model_monitoring_job`` RPC rather than
    ``ModelMonitor.run()``, which blocks until the analysis finishes. Drift
    detection is an observability side effect and should not hold the pipeline
    before undeploy.

    Args:
        project: GCP project ID.
        location: GCP region (e.g. europe-west2).
        model_monitor_name: Resource name of an existing ModelMonitor.
        training_dataset_gcs_uri: GCS CSV URI used as the drift baseline.
        target_bq_table_uri: BigQuery table URI of endpoint request-response
            logs (``bq://project.dataset.table``).
        job_display_name: Display name prefix for the monitoring job.
        monitored_features: Maps each feature name to its schema data type.
        default_drift_threshold: Alert threshold applied to every feature.
        notification_emails: Email addresses to notify when an alert fires.
        enable_cloud_logging: Also write alerts to Cloud Logging.
    """

    import logging
    import re
    import uuid

    from google.cloud.aiplatform_v1beta1.services.model_monitoring_service import (
        ModelMonitoringServiceClient,
    )
    from google.cloud.aiplatform_v1beta1.types import (
        model_monitoring_job as gca_model_monitoring_job,
        model_monitoring_service as gca_model_monitoring_service,
        model_monitoring_spec as gca_model_monitoring_spec,
    )
    from vertexai.resources.preview.ml_monitoring.spec import (
        notification as notif_spec,
        objective,
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if notification_emails is None:
        notification_emails = []

    def _slugify(name: str, max_length: int = 63) -> str:
        slug = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
        slug = re.sub(r"-{2,}", "-", slug)
        if not slug or not slug[0].isalpha():
            slug = f"m-{slug}"
        return slug[:max_length].rstrip("-")

    if (
        not training_dataset_gcs_uri
        or not target_bq_table_uri
        or not monitored_features
    ):
        logger.warning(
            "Skipping Model Monitoring v2: training_dataset_gcs_uri, "
            "target_bq_table_uri, or monitored_features was empty "
            "(training=%r, target=%r).",
            training_dataset_gcs_uri,
            target_bq_table_uri,
        )
        return

    baseline_dataset = objective.MonitoringInput(
        gcs_uri=training_dataset_gcs_uri, data_format="csv"
    )
    target_dataset = objective.MonitoringInput(table_uri=target_bq_table_uri)
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

    monitoring_job_id = _slugify(f"{job_display_name}-mon-{uuid.uuid4().hex[:8]}")
    monitoring_job_request = gca_model_monitoring_job.ModelMonitoringJob(
        display_name=f"{job_display_name}-run",
        model_monitoring_spec=gca_model_monitoring_spec.ModelMonitoringSpec(
            objective_spec=gca_model_monitoring_spec.ModelMonitoringObjectiveSpec(
                tabular_objective=tabular_objective_spec._as_proto(),
                baseline_dataset=baseline_dataset._as_proto(),
                target_dataset=target_dataset._as_proto(),
            ),
            notification_spec=(
                notification_spec._as_proto() if notification_spec else None
            ),
        ),
    )
    api_endpoint = f"{location}-aiplatform.googleapis.com"
    monitoring_client = ModelMonitoringServiceClient(
        client_options={"api_endpoint": api_endpoint}
    )
    created_monitoring_job = monitoring_client.create_model_monitoring_job(
        request=gca_model_monitoring_service.CreateModelMonitoringJobRequest(
            parent=model_monitor_name,
            model_monitoring_job=monitoring_job_request,
            model_monitoring_job_id=monitoring_job_id,
        )
    )
    logger.info(
        "Model Monitoring v2 job submitted (fire-and-forget): %s",
        created_monitoring_job.name,
    )
