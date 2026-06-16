import os
import sys
from unittest import mock

import pandas as pd

os.environ.setdefault("CONTAINER_IMAGE_REGISTRY", "test-registry")
import pytest
from kfp.dsl import Artifact, Dataset

_mock_aip = mock.MagicMock()
sys.modules["google.cloud.aiplatform"] = _mock_aip

_mock_ml_monitoring = mock.MagicMock()
_mock_model_monitor_class = mock.MagicMock()
_mock_model_monitor_instance = mock.MagicMock()
_mock_ml_monitoring.ModelMonitor = _mock_model_monitor_class
_mock_model_monitor_class.create.return_value = _mock_model_monitor_instance

_mock_spec_notification = mock.MagicMock()
_mock_spec_objective = mock.MagicMock()
_mock_spec_schema = mock.MagicMock()
_mock_spec = mock.MagicMock()
_mock_spec.notification = _mock_spec_notification
_mock_spec.objective = _mock_spec_objective
_mock_spec.schema = _mock_spec_schema
_mock_preview = mock.MagicMock()
_mock_preview.ml_monitoring = _mock_ml_monitoring
sys.modules["vertexai"] = mock.MagicMock()
sys.modules["vertexai.resources"] = mock.MagicMock()
sys.modules["vertexai.resources.preview"] = _mock_preview
sys.modules["vertexai.resources.preview.ml_monitoring"] = _mock_ml_monitoring
sys.modules["vertexai.resources.preview.ml_monitoring.spec"] = _mock_spec
sys.modules["vertexai.resources.preview.ml_monitoring.spec.notification"] = (
    _mock_spec_notification
)
sys.modules["vertexai.resources.preview.ml_monitoring.spec.objective"] = (
    _mock_spec_objective
)
sys.modules["vertexai.resources.preview.ml_monitoring.spec.schema"] = (
    _mock_spec_schema
)

_mock_storage = mock.MagicMock()
sys.modules["google.cloud.storage"] = _mock_storage

from google.api_core.exceptions import NotFound as _BQNotFound
_mock_bq_client = mock.MagicMock()
_mock_bq_client.get_dataset.side_effect = _BQNotFound("dataset not found")
_mock_bq_client.create_dataset.return_value = None
_mock_bigquery = mock.MagicMock()
_mock_bigquery.Client.return_value = _mock_bq_client
sys.modules["google.cloud.bigquery"] = _mock_bigquery

_mock_gcpc_artifact_types = mock.MagicMock()


class _FakeVertexModel(Artifact):
    schema_title = "google.VertexModel"
    schema_version = "0.0.1"
    __module__ = "google_cloud_pipeline_components.types.artifact_types"
    __qualname__ = "VertexModel"


_mock_gcpc_artifact_types.VertexModel = _FakeVertexModel
sys.modules["google_cloud_pipeline_components"] = mock.MagicMock()
sys.modules["google_cloud_pipeline_components.types"] = mock.MagicMock()
sys.modules["google_cloud_pipeline_components.types.artifact_types"] = (
    _mock_gcpc_artifact_types
)

import components  # noqa: E402

deploy_model = components.deploy_model.python_func

PROJECT = "test-project"
LOCATION = "europe-west2"
ENDPOINT_NAME = "my-endpoint"
SERVICE_ACCOUNT = "vertex-endpoint-sa@test-project.iam.gserviceaccount.com"
MODEL_RESOURCE = "projects/test-project/locations/europe-west2/models/123@1"
TARGET_FIELD = "target"
TRAIN_COLUMNS = ["listing_id", "mileage", "colour", TARGET_FIELD]
FEATURE_COLUMNS = [c for c in TRAIN_COLUMNS if c != TARGET_FIELD]


def _make_vertex_model(tmp_path):
    art = Artifact(uri=str(tmp_path / "vertex_model"))
    art.metadata["resourceName"] = MODEL_RESOURCE
    art.metadata["feature_columns"] = FEATURE_COLUMNS
    return art


def _make_training_data(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(columns=TRAIN_COLUMNS).to_csv(csv_path, index=False)
    return Dataset(uri=str(csv_path))


def _setup_mocks(existing_endpoint=True, deployed_models=None):
    sys.modules["google.cloud.aiplatform"] = _mock_aip
    sys.modules["google.cloud.storage"] = _mock_storage
    _mock_aip.reset_mock()
    _mock_ml_monitoring.reset_mock()
    _mock_model_monitor_class.reset_mock()
    _mock_model_monitor_instance.reset_mock()
    _mock_model_monitor_class.create.return_value = _mock_model_monitor_instance
    _mock_model_monitor_class.list.return_value = []
    _mock_storage.reset_mock()

    mock_model = mock.MagicMock()
    _mock_aip.Model.return_value = mock_model

    mock_endpoint = mock.MagicMock()
    mock_endpoint.resource_name = f"projects/{PROJECT}/locations/{LOCATION}/endpoints/456"
    mock_endpoint.list_models.return_value = deployed_models or []

    if existing_endpoint:
        _mock_aip.Endpoint.list.return_value = [mock_endpoint]
    else:
        _mock_aip.Endpoint.list.return_value = []
        _mock_aip.Endpoint.create.return_value = mock_endpoint

    return mock_model, mock_endpoint


def test_creates_endpoint_when_none_exists(tmp_path):
    mock_model, mock_endpoint = _setup_mocks(existing_endpoint=False)

    result = deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=False,
    )

    _mock_aip.Endpoint.create.assert_called_once()
    mock_model.deploy.assert_called_once()
    assert result[0] == "456"


def test_uses_existing_endpoint(tmp_path):
    mock_model, mock_endpoint = _setup_mocks(existing_endpoint=True)

    result = deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=False,
    )

    _mock_aip.Endpoint.create.assert_not_called()
    mock_model.deploy.assert_called_once()
    assert result[0] == "456"


def test_direct_deploy_when_no_models_on_endpoint(tmp_path):
    mock_model, _ = _setup_mocks(existing_endpoint=True, deployed_models=[])

    deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=False,
    )

    call_kwargs = mock_model.deploy.call_args[1]
    assert call_kwargs["traffic_percentage"] == 100
    assert call_kwargs["service_account"] == SERVICE_ACCOUNT
    assert "traffic_split" not in call_kwargs


def test_rolling_deploy_with_existing_model(tmp_path):
    existing = mock.MagicMock()
    existing.id = "existing-model-id"
    existing.create_time = "2026-01-01T00:00:00Z"
    mock_model, mock_endpoint = _setup_mocks(
        existing_endpoint=True, deployed_models=[existing],
    )

    deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=False,
    )

    call_kwargs = mock_model.deploy.call_args[1]
    assert call_kwargs["service_account"] == SERVICE_ACCOUNT
    assert call_kwargs["traffic_split"] == {
        "0": 100,
        "existing-model-id": 0,
    }


def test_undeploys_old_models(tmp_path):
    models = []
    for i, ts in enumerate(["2026-01-01", "2026-02-01", "2026-03-01"]):
        m = mock.MagicMock()
        m.id = f"model-{i}"
        m.create_time = ts
        models.append(m)

    mock_model, mock_endpoint = _setup_mocks(
        existing_endpoint=True, deployed_models=[models[0]],
    )
    mock_endpoint.list_models.side_effect = [[models[0]], models]

    deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=False,
    )

    mock_endpoint.undeploy.assert_called_once_with("model-0")


def test_raises_on_multiple_endpoints(tmp_path):
    sys.modules["google.cloud.aiplatform"] = _mock_aip
    _mock_aip.reset_mock()
    _mock_aip.Endpoint.list.return_value = [mock.MagicMock(), mock.MagicMock()]

    with pytest.raises(RuntimeError, match="Multiple endpoints"):
        deploy_model(
            vertex_model=_make_vertex_model(tmp_path),
            training_data=_make_training_data(tmp_path),
            project=PROJECT,
            location=LOCATION,
            endpoint_name=ENDPOINT_NAME,
            service_account=SERVICE_ACCOUNT,
        )


def test_loads_correct_model_resource(tmp_path):
    _setup_mocks(existing_endpoint=True)

    deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=False,
    )

    _mock_aip.Model.assert_called_once_with(MODEL_RESOURCE)


def test_monitoring_enabled_creates_model_monitor_v2_and_schedule(tmp_path):
    mock_model, mock_endpoint = _setup_mocks(existing_endpoint=True)

    deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=True,
        monitor_interval_hours_cron_job="0 */12 * * *",
        monitor_window_hours=12,
        default_drift_threshold=0.3,
        notification_emails=["alerts@example.com"],
    )

    _mock_model_monitor_class.create.assert_called_once()
    create_kwargs = _mock_model_monitor_class.create.call_args[1]
    assert create_kwargs["model_name"] == "projects/test-project/locations/europe-west2/models/123"
    assert create_kwargs["model_version_id"] == "1"
    assert create_kwargs["display_name"] == f"{ENDPOINT_NAME}-monitoring"
    assert create_kwargs["project"] == PROJECT
    assert create_kwargs["location"] == LOCATION
    assert create_kwargs["notification_spec"] is not None

    _mock_model_monitor_instance.create_schedule.assert_called_once()
    schedule_kwargs = _mock_model_monitor_instance.create_schedule.call_args[1]
    assert schedule_kwargs["cron"] == "0 */12 * * *"
    assert schedule_kwargs["display_name"] == f"{ENDPOINT_NAME}-monitoring-schedule"


def test_monitoring_disabled_skips_model_monitor(tmp_path):
    mock_model, mock_endpoint = _setup_mocks(existing_endpoint=True)

    deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=False,
    )

    _mock_model_monitor_class.create.assert_not_called()


def test_monitoring_no_emails_passes_no_notification_spec(tmp_path):
    mock_model, mock_endpoint = _setup_mocks(existing_endpoint=True)

    deploy_model(
        vertex_model=_make_vertex_model(tmp_path),
        training_data=_make_training_data(tmp_path),
        project=PROJECT,
        location=LOCATION,
        endpoint_name=ENDPOINT_NAME,
        service_account=SERVICE_ACCOUNT,
        enable_monitoring=True,
        notification_emails=[],
    )

    create_kwargs = _mock_model_monitor_class.create.call_args[1]
    assert create_kwargs["notification_spec"] is None
