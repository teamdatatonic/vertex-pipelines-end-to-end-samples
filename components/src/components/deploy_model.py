from typing import NamedTuple

from kfp.dsl import Artifact, Dataset, Input, component
from google_cloud_pipeline_components.types.artifact_types import VertexModel


@component(
    base_image="python:3.12.8",
    packages_to_install=[
        "google-cloud-aiplatform==1.135.0",
        "google-cloud-pipeline-components==2.22.0",
        "google-cloud-bigquery>=3.40.0",
    ],
)
def deploy_model(
    vertex_model: Input[VertexModel],
    training_data: Input[Dataset],
    project: str,
    location: str,
    endpoint_name: str,
    service_account: str,
    enable_monitoring: bool = True,
    monitor_interval_hours_cron_job: str = "0 0 * * *",
    monitor_window_hours: int = 24,
    default_drift_threshold: float = 0.3,
    notification_emails: list = [],
    machine_type: str = "n2-highcpu-2",
    traffic_percentage: int = 100,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    request_response_logging_sampling_rate: float = 0.3,
    monitored_feature_names: list = [],
    monitored_prediction_field_names: list = [],
) -> NamedTuple("Outputs", [("endpoint_id", str)]):
    """
    Deploy a model to a Vertex AI endpoint using a rolling strategy,
    and optionally enable Model Monitoring v2 (feature drift and scheduled runs).

    Finds an existing endpoint by display name or creates a new one, then
    deploys the model. If there are already models on the endpoint, the new
    model receives ``traffic_percentage`` and the previous model gets the
    remainder. Models older than the two most recent are undeployed.

    Args:
        vertex_model: VertexModel artifact from the upload_model component.
        training_data: Training dataset used as baseline for skew detection.
        project: GCP project ID.
        location: GCP region (e.g. europe-west2).
        endpoint_name: Display name for the endpoint.
        service_account: Service account for the deployed model.
        enable_monitoring: Whether to create a model monitoring job.
        monitor_interval_hours_cron_job: Cron expression for the monitoring schedule (e.g. "0 0 * * *").
        monitor_window_hours: Hours of endpoint traffic to analyse per monitoring run.
        default_drift_threshold: Drift detection threshold applied to all features.
        notification_emails: Email addresses to notify when alerts fire.
        machine_type: Machine type for serving.
        traffic_percentage: Percentage of traffic for the new deployment.
        min_replica_count: Minimum replica count.
        max_replica_count: Maximum replica count.
        request_response_logging_sampling_rate: Fraction of predictions to log (0.0–1.0).
        monitored_feature_names: Feature names to monitor for drift (must exist in request and baseline).
        monitored_prediction_field_names: Top-level float keys on each prediction object in the response.
    """

    import logging
    import google.cloud.aiplatform as aip
    from google.cloud import bigquery
    from google.api_core import exceptions
    from vertexai.resources.preview import ml_monitoring
    from vertexai.resources.preview.ml_monitoring.spec import (
        notification as notif_spec,
        objective,
        schema as schema_spec,
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aip.init(project=project, location=location)

    model_resource_name = vertex_model.metadata["resourceName"]
    logger.info("Deploying model: %s", model_resource_name)

    model = aip.Model(model_resource_name)

    endpoints = aip.Endpoint.list(
        filter=f'display_name="{endpoint_name}"',
        project=project,
        location=location,
    )

    if len(endpoints) > 1:
        raise RuntimeError(
            f"Multiple endpoints found with name '{endpoint_name}'. "
            "Please resolve manually."
        )

    endpoint = endpoints[0] if endpoints else None

    if endpoint is None:
        bq_client = bigquery.Client(project=project)
        dataset_id = f"{project}.model_monitoring_logs"
        try:
            bq_client.get_dataset(dataset_id)
            logger.info("Dataset %s already exists.", dataset_id)
        except exceptions.NotFound:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = location
            bq_client.create_dataset(dataset, timeout=30)
            logger.info("Created dataset %s", dataset_id)
        logger.info("No endpoint found, creating: %s", endpoint_name)
        endpoint = aip.Endpoint.create(
            display_name=endpoint_name,
            project=project,
            location=location,
            enable_request_response_logging=True,
            request_response_logging_bq_destination_table=f"bq://{project}.model_monitoring_logs.logs_{endpoint_name.replace('-', '_')}",
            request_response_logging_sampling_rate=request_response_logging_sampling_rate,
        )
        logger.info("Created endpoint: %s", endpoint.resource_name)

    deployed_models = endpoint.list_models()

    if not deployed_models:
        logger.info("No models on endpoint, deploying directly")
        model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=traffic_percentage,
            service_account=service_account,
            sync=True,
        )
    else:
        last_deployed = max(
            deployed_models, key=lambda dm: dm.create_time
        )
        logger.info(
            "Rolling deploy: new model gets %d%% traffic, "
            "existing model %s keeps %d%%",
            traffic_percentage,
            last_deployed.id,
            100 - traffic_percentage,
        )
        model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_split={
                "0": traffic_percentage,
                last_deployed.id: 100 - traffic_percentage,
            },
            service_account=service_account,
            sync=True,
        )

    deployed_models = sorted(
        endpoint.list_models(), key=lambda dm: dm.create_time
    )
    for old_model in deployed_models[:-2]:
        logger.info("Undeploying old model: %s", old_model.id)
        endpoint.undeploy(old_model.id)

    endpoint_id = endpoint.resource_name.split("/")[-1]
    logger.info("Deployment complete on endpoint: %s (id=%s)", endpoint.resource_name, endpoint_id)

    if enable_monitoring:
        logger.info("Setting up Model Monitoring v2 for endpoint: %s", endpoint_name)

        gcs_uri = training_data.uri

        resource_name = vertex_model.metadata["resourceName"]
        if "@" in resource_name:
            model_name, model_version_id = resource_name.rsplit("@", 1)
        else:
            model_name = resource_name
            model_version_id = "1"

        feature_fields = [
            schema_spec.FieldSchema(name=col, data_type="float")
            for col in monitored_feature_names
        ]
        prediction_fields = [
            schema_spec.FieldSchema(name=col, data_type="float")
            for col in monitored_prediction_field_names
        ]
        model_monitoring_schema = schema_spec.ModelMonitoringSchema(
            feature_fields=feature_fields,
            prediction_fields=prediction_fields,
        )

        training_input = objective.MonitoringInput(
            gcs_uri=gcs_uri,
            data_format="csv",
        )

        feature_alert_thresholds = {
            col: default_drift_threshold for col in monitored_feature_names
        }

        feature_drift_spec = objective.DataDriftSpec(
            categorical_metric_type="l_infinity",
            numeric_metric_type="jensen_shannon_divergence",
            default_categorical_alert_threshold=default_drift_threshold,
            default_numeric_alert_threshold=default_drift_threshold,
            feature_alert_thresholds=feature_alert_thresholds,
        )
        tabular_objective_spec = objective.TabularObjective(
            feature_drift_spec=feature_drift_spec,
        )

        notification_spec = None
        if notification_emails:
            notification_spec = notif_spec.NotificationSpec(
                user_emails=notification_emails,
                enable_cloud_logging=True,
            )

        display_name_monitoring = f"{endpoint_name}-monitoring"
        try:
            existing_monitors = ml_monitoring.ModelMonitor.list(
                project=project,
                location=location,
                filter=f'display_name="{display_name_monitoring}"',
            )
        except Exception:
            try:
                all_monitors = ml_monitoring.ModelMonitor.list(
                    project=project,
                    location=location,
                )
                existing_monitors = [
                    m for m in (all_monitors or [])
                    if getattr(m._gca_resource, "display_name", None) == display_name_monitoring
                ]
            except Exception as e:
                logger.warning("Could not list existing model monitors: %s", e)
                existing_monitors = []
        for mon in existing_monitors or []:
            try:
                for s in mon.list_schedules():
                    mon.delete_schedule(s.name)
                    logger.info("Deleted previous monitoring schedule: %s", s.name)
            except Exception as e:
                logger.warning("Could not delete schedule for monitor %s: %s", mon.resource_name, e)

        model_monitor = ml_monitoring.ModelMonitor.create(
            model_name=model_name,
            model_version_id=model_version_id,
            training_dataset=training_input,
            display_name=f"{endpoint_name}-monitoring",
            model_monitoring_schema=model_monitoring_schema,
            tabular_objective_spec=tabular_objective_spec,
            notification_spec=notification_spec,
            project=project,
            location=location,
        )
        logger.info("Model Monitor (v2) created: %s", model_monitor._gca_resource.name)

        target_input = objective.MonitoringInput(
            endpoints=[endpoint.resource_name],
            window=f"{monitor_window_hours}h",
        )
        baseline_input = objective.MonitoringInput(
            gcs_uri=gcs_uri,
            data_format="csv",
        )
        schedule = model_monitor.create_schedule(
            cron=monitor_interval_hours_cron_job,
            target_dataset=target_input,
            baseline_dataset=baseline_input,
            display_name=f"{endpoint_name}-monitoring-schedule",
            tabular_objective_spec=tabular_objective_spec,
            notification_spec=notification_spec,
        )
        logger.info("Model monitoring schedule created: %s", schedule.name)
    else:
        logger.info("Model monitoring disabled, skipping")

    return (endpoint_id,)
