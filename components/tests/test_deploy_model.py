# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from types import SimpleNamespace
from unittest import mock

import pytest
from google.api_core.exceptions import NotFound
from kfp.dsl import Artifact

import components

deploy_model = components.deploy_model.python_func

PROJECT = "test-project"
LOCATION = "europe-west2"
ENDPOINT_NAME = "my-endpoint"
ENDPOINT_ID = "456"
SERVICE_ACCOUNT = "vertex-pipelines@test-project.iam.gserviceaccount.com"
MODEL_RESOURCE = "projects/test-project/locations/europe-west2/models/123@1"
TRAINING_DATA_URI = "gs://bucket/train.csv"
FEATURE_NAMES = ["mileage", "photos_damage_count"]
PREDICTION_FIELDS = ["prediction_value"]


def _make_vertex_model(tmp_path, resource_name: str = MODEL_RESOURCE) -> Artifact:
    artifact = Artifact(uri=str(tmp_path / "vertex_model"))
    artifact.metadata["resourceName"] = resource_name
    return artifact


def _make_deployed_model(model_id: str, create_time: str):
    model = mock.MagicMock()
    model.id = model_id
    model.create_time = create_time
    return model


@pytest.fixture
def sdk_mocks():
    """Mock Vertex / BigQuery SDKs imported inside deploy_model."""
    mock_aip = mock.MagicMock()
    mock_bq = mock.MagicMock()
    mock_bq_client = mock.MagicMock()
    mock_bq.Client.return_value = mock_bq_client

    mock_ml_monitoring = mock.MagicMock()
    mock_monitor = mock.MagicMock()
    mock_monitor._gca_resource.name = "projects/test/locations/loc/modelMonitors/1"
    mock_ml_monitoring.ModelMonitor.create.return_value = mock_monitor
    mock_ml_monitoring.ModelMonitor.list.return_value = []

    mock_notif = mock.MagicMock()
    mock_objective = mock.MagicMock()
    mock_schema = mock.MagicMock()
    mock_preview = mock.MagicMock(ml_monitoring=mock_ml_monitoring)
    mock_spec = mock.MagicMock(
        notification=mock_notif,
        objective=mock_objective,
        schema=mock_schema,
    )

    modules = {
        "google.cloud.aiplatform": mock_aip,
        "google.cloud.bigquery": mock_bq,
        "vertexai": mock.MagicMock(),
        "vertexai.resources": mock.MagicMock(),
        "vertexai.resources.preview": mock_preview,
        "vertexai.resources.preview.ml_monitoring": mock_ml_monitoring,
        "vertexai.resources.preview.ml_monitoring.spec": mock_spec,
        "vertexai.resources.preview.ml_monitoring.spec.notification": mock_notif,
        "vertexai.resources.preview.ml_monitoring.spec.objective": mock_objective,
        "vertexai.resources.preview.ml_monitoring.spec.schema": mock_schema,
    }

    with mock.patch.dict(sys.modules, modules):
        yield SimpleNamespace(
            aip=mock_aip,
            bq=mock_bq,
            bq_client=mock_bq_client,
            ml_monitoring=mock_ml_monitoring,
            monitor=mock_monitor,
            notif=mock_notif,
            objective=mock_objective,
            schema=mock_schema,
        )


def _configure_endpoint(
    sdk_mocks,
    *,
    existing_endpoint: bool = True,
    deployed_models=None,
):
    mock_model = mock.MagicMock()
    sdk_mocks.aip.Model.return_value = mock_model

    mock_endpoint = mock.MagicMock()
    mock_endpoint.resource_name = (
        f"projects/{PROJECT}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
    )
    mock_endpoint.list_models.return_value = list(deployed_models or [])

    if existing_endpoint:
        sdk_mocks.aip.Endpoint.list.return_value = [mock_endpoint]
    else:
        sdk_mocks.aip.Endpoint.list.return_value = []
        sdk_mocks.aip.Endpoint.create.return_value = mock_endpoint

    return mock_model, mock_endpoint


def _call_deploy(
    tmp_path,
    *,
    enable_monitoring: bool = False,
    traffic_percentage: int = 100,
    notification_emails=None,
    monitored_feature_names=None,
    monitored_prediction_field_names=None,
    training_data_uri: str = TRAINING_DATA_URI,
    **kwargs,
):
    return deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=enable_monitoring,
        traffic_percentage=traffic_percentage,
        notification_emails=notification_emails or [],
        monitored_feature_names=monitored_feature_names or [],
        monitored_prediction_field_names=monitored_prediction_field_names or [],
        training_data_uri=training_data_uri,
        **kwargs,
    )


def test_creates_endpoint_and_bq_dataset_when_none_exists(tmp_path, sdk_mocks):
    mock_model, _ = _configure_endpoint(sdk_mocks, existing_endpoint=False)
    sdk_mocks.bq_client.get_dataset.side_effect = NotFound("missing")

    result = _call_deploy(tmp_path)

    sdk_mocks.bq.Client.assert_called_once_with(project=PROJECT)
    sdk_mocks.bq_client.create_dataset.assert_called_once()
    sdk_mocks.aip.Endpoint.create.assert_called_once()
    create_kwargs = sdk_mocks.aip.Endpoint.create.call_args[1]
    assert create_kwargs["display_name"] == ENDPOINT_NAME
    assert create_kwargs["enable_request_response_logging"] is True
    assert (
        create_kwargs["request_response_logging_bq_destination_table"]
        == f"bq://{PROJECT}.model_monitoring_logs.logs_{ENDPOINT_NAME.replace('-', '_')}"
    )
    mock_model.deploy.assert_called_once()
    assert result[0] == ENDPOINT_ID


def test_uses_existing_endpoint(tmp_path, sdk_mocks):
    mock_model, _ = _configure_endpoint(sdk_mocks, existing_endpoint=True)

    result = _call_deploy(tmp_path)

    sdk_mocks.aip.Endpoint.create.assert_not_called()
    sdk_mocks.bq.Client.assert_not_called()
    mock_model.deploy.assert_called_once()
    assert result[0] == ENDPOINT_ID


def test_direct_deploy_when_no_models_on_endpoint(tmp_path, sdk_mocks):
    mock_model, mock_endpoint = _configure_endpoint(
        sdk_mocks, existing_endpoint=True, deployed_models=[]
    )

    _call_deploy(tmp_path, traffic_percentage=100)

    deploy_kwargs = mock_model.deploy.call_args[1]
    assert deploy_kwargs["endpoint"] is mock_endpoint
    assert deploy_kwargs["traffic_percentage"] == 100
    assert deploy_kwargs["service_account"] == SERVICE_ACCOUNT
    assert deploy_kwargs["sync"] is True
    assert "traffic_split" not in deploy_kwargs


def test_rolling_deploy_with_existing_model(tmp_path, sdk_mocks):
    existing = _make_deployed_model("existing-model-id", "2026-01-01T00:00:00Z")
    mock_model, _ = _configure_endpoint(
        sdk_mocks, existing_endpoint=True, deployed_models=[existing]
    )

    _call_deploy(tmp_path, traffic_percentage=80)

    deploy_kwargs = mock_model.deploy.call_args[1]
    assert deploy_kwargs["service_account"] == SERVICE_ACCOUNT
    assert deploy_kwargs["traffic_split"] == {
        "0": 80,
        "existing-model-id": 20,
    }
    assert "traffic_percentage" not in deploy_kwargs


def test_undeploys_models_older_than_two_most_recent(tmp_path, sdk_mocks):
    models = [
        _make_deployed_model(f"model-{i}", ts)
        for i, ts in enumerate(["2026-01-01", "2026-02-01", "2026-03-01"])
    ]
    mock_model, mock_endpoint = _configure_endpoint(
        sdk_mocks, existing_endpoint=True, deployed_models=[models[0]]
    )
    # First list (pre-deploy) has one model; second list (post-deploy) has three.
    mock_endpoint.list_models.side_effect = [[models[0]], models]

    _call_deploy(tmp_path)

    mock_endpoint.undeploy.assert_called_once_with("model-0")


def test_raises_on_multiple_endpoints(tmp_path, sdk_mocks):
    sdk_mocks.aip.Endpoint.list.return_value = [mock.MagicMock(), mock.MagicMock()]

    with pytest.raises(RuntimeError, match="Multiple endpoints"):
        _call_deploy(tmp_path)


def test_loads_correct_model_resource(tmp_path, sdk_mocks):
    _configure_endpoint(sdk_mocks, existing_endpoint=True)

    _call_deploy(tmp_path)

    sdk_mocks.aip.Model.assert_called_once_with(MODEL_RESOURCE)
    sdk_mocks.aip.init.assert_called_once_with(project=PROJECT, location=LOCATION)


def test_monitoring_enabled_creates_model_monitor_and_schedule(tmp_path, sdk_mocks):
    _configure_endpoint(sdk_mocks, existing_endpoint=True)

    _call_deploy(
        tmp_path,
        enable_monitoring=True,
        monitor_interval_hours_cron_job="0 */12 * * *",
        monitor_window_hours=12,
        default_drift_threshold=0.3,
        notification_emails=["alerts@example.com"],
        monitored_feature_names=FEATURE_NAMES,
        monitored_prediction_field_names=PREDICTION_FIELDS,
        training_data_uri=TRAINING_DATA_URI,
    )

    sdk_mocks.ml_monitoring.ModelMonitor.create.assert_called_once()
    create_kwargs = sdk_mocks.ml_monitoring.ModelMonitor.create.call_args[1]
    assert (
        create_kwargs["model_name"]
        == "projects/test-project/locations/europe-west2/models/123"
    )
    assert create_kwargs["model_version_id"] == "1"
    assert create_kwargs["display_name"] == f"{ENDPOINT_NAME}-monitoring"
    assert create_kwargs["project"] == PROJECT
    assert create_kwargs["location"] == LOCATION
    assert create_kwargs["notification_spec"] is not None

    sdk_mocks.objective.MonitoringInput.assert_any_call(
        gcs_uri=TRAINING_DATA_URI,
        data_format="csv",
    )
    sdk_mocks.monitor.create_schedule.assert_called_once()
    schedule_kwargs = sdk_mocks.monitor.create_schedule.call_args[1]
    assert schedule_kwargs["cron"] == "0 */12 * * *"
    assert schedule_kwargs["display_name"] == f"{ENDPOINT_NAME}-monitoring-schedule"


def test_monitoring_disabled_skips_model_monitor(tmp_path, sdk_mocks):
    _configure_endpoint(sdk_mocks, existing_endpoint=True)

    _call_deploy(tmp_path, enable_monitoring=False)

    sdk_mocks.ml_monitoring.ModelMonitor.create.assert_not_called()
    sdk_mocks.monitor.create_schedule.assert_not_called()


def test_monitoring_without_emails_omits_notification_spec(tmp_path, sdk_mocks):
    _configure_endpoint(sdk_mocks, existing_endpoint=True)

    _call_deploy(
        tmp_path,
        enable_monitoring=True,
        notification_emails=[],
    )

    create_kwargs = sdk_mocks.ml_monitoring.ModelMonitor.create.call_args[1]
    assert create_kwargs["notification_spec"] is None
    sdk_mocks.notif.NotificationSpec.assert_not_called()


def test_monitoring_model_resource_without_version_defaults_to_one(tmp_path, sdk_mocks):
    _configure_endpoint(sdk_mocks, existing_endpoint=True)
    model_without_version = (
        "projects/test-project/locations/europe-west2/models/123"
    )

    deploy_model(
        vertex_model=_make_vertex_model(tmp_path, model_without_version),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=True,
        training_data_uri=TRAINING_DATA_URI,
    )

    create_kwargs = sdk_mocks.ml_monitoring.ModelMonitor.create.call_args[1]
    assert create_kwargs["model_name"] == model_without_version
    assert create_kwargs["model_version_id"] == "1"
