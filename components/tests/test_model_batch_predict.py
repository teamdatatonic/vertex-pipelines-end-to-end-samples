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
import sys
from unittest import mock

import pytest
from kfp.dsl import Model
from google.cloud.aiplatform_v1beta1.types.job_state import JobState

_mock_ml_monitoring = mock.MagicMock()
_mock_model_monitor_class = mock.MagicMock()
_mock_ml_monitoring.ModelMonitor = _mock_model_monitor_class

_mock_spec_notification = mock.MagicMock()
_mock_spec_objective = mock.MagicMock()
_mock_spec_schema = mock.MagicMock()
_mock_preview = mock.MagicMock()
_mock_preview.ml_monitoring = _mock_ml_monitoring
sys.modules["vertexai"] = mock.MagicMock()
sys.modules["vertexai.resources"] = mock.MagicMock()
sys.modules["vertexai.resources.preview"] = _mock_preview
sys.modules["vertexai.resources.preview.ml_monitoring"] = _mock_ml_monitoring
sys.modules["vertexai.resources.preview.ml_monitoring.spec"] = mock.MagicMock(
    notification=_mock_spec_notification,
    objective=_mock_spec_objective,
    schema=_mock_spec_schema,
)
sys.modules[
    "vertexai.resources.preview.ml_monitoring.spec.notification"
] = _mock_spec_notification
sys.modules[
    "vertexai.resources.preview.ml_monitoring.spec.objective"
] = _mock_spec_objective
sys.modules["vertexai.resources.preview.ml_monitoring.spec.schema"] = _mock_spec_schema

import components  # noqa: E402

model_batch_predict = components.model_batch_predict.python_func


TRAINING_GCS_URI = "gs://bucket/train.csv"
MONITORED_FEATURES = {"trip_miles": "float", "company": "categorical"}

mock_job1 = mock.Mock()
mock_job1.name = "mock-batch-job"
mock_job1.state = JobState.JOB_STATE_SUCCEEDED


def _reset_monitoring_mocks():
    _mock_model_monitor_class.reset_mock()
    _mock_model_monitor_class.list.return_value = []
    mock_monitor_instance = mock.MagicMock()
    _mock_model_monitor_class.create.return_value = mock_monitor_instance
    return mock_monitor_instance


@mock.patch(
    "google.cloud.aiplatform_v1beta1.services.job_service.JobServiceClient.create_batch_prediction_job",  # noqa : E501
    return_value=mock_job1,
)
@mock.patch(
    "google.cloud.aiplatform_v1beta1.services.job_service.JobServiceClient.get_batch_prediction_job",  # noqa : E501
    return_value=mock_job1,
)
@pytest.mark.parametrize(
    "source_format,destination_format,source_uri",
    [
        ("bigquery", "bigquery", "bq://a.b.c"),
        ("csv", "csv", '["gs://file.csv"]'),
    ],
)
def test_model_batch_predict_submits_and_polls_job(
    create_job, get_job, tmp_path, source_format, destination_format, source_uri
):
    """
    Asserts model_batch_predict successfully creates and polls the batch
    prediction job, with Model Monitoring v2 disabled.
    """
    _reset_monitoring_mocks()
    mock_model = Model(uri=str(tmp_path / "model"), metadata={"resourceName": ""})
    gcp_resources_path = tmp_path / "gcp_resources.json"

    try:
        model_batch_predict(
            model=mock_model,
            job_display_name="",
            location="",
            project="",
            source_uri=source_uri,
            destination_uri=destination_format,
            source_format=source_format,
            destination_format=destination_format,
            enable_monitoring=False,
            gcp_resources=str(gcp_resources_path),
        )

        create_job.assert_called_once()
        get_job.assert_called_once()
        assert gcp_resources_path.exists()
    finally:
        gcp_resources_path.unlink(missing_ok=True)
    _mock_model_monitor_class.create.assert_not_called()


@mock.patch(
    "google.cloud.aiplatform_v1beta1.services.job_service.JobServiceClient.create_batch_prediction_job",  # noqa : E501
    return_value=mock_job1,
)
@mock.patch(
    "google.cloud.aiplatform_v1beta1.services.job_service.JobServiceClient.get_batch_prediction_job",  # noqa : E501
    return_value=mock_job1,
)
def test_model_batch_predict_skips_monitoring_without_training_uri(
    create_job, get_job, tmp_path
):
    """
    If enable_monitoring is True but no baseline dataset is configured, monitoring
    is skipped (with a warning) rather than failing the batch job.
    """
    mock_monitor_instance = _reset_monitoring_mocks()
    mock_model = Model(uri=str(tmp_path / "model"), metadata={"resourceName": "m@1"})
    gcp_resources_path = tmp_path / "gcp_resources.json"

    try:
        model_batch_predict(
            model=mock_model,
            job_display_name="predict-job",
            location="europe-west2",
            project="my-project",
            source_uri="bq://a.b.c",
            destination_uri="bq://a.b.d",
            source_format="bigquery",
            destination_format="bigquery",
            enable_monitoring=True,
            monitoring_training_gcs_uri="",
            monitored_features=MONITORED_FEATURES,
            gcp_resources=str(gcp_resources_path),
        )
    finally:
        gcp_resources_path.unlink(missing_ok=True)

    _mock_model_monitor_class.create.assert_not_called()
    mock_monitor_instance.run.assert_not_called()


@mock.patch(
    "google.cloud.aiplatform_v1beta1.services.job_service.JobServiceClient.create_batch_prediction_job",  # noqa : E501
    return_value=mock_job1,
)
@mock.patch(
    "google.cloud.aiplatform_v1beta1.services.job_service.JobServiceClient.get_batch_prediction_job",  # noqa : E501
    return_value=mock_job1,
)
def test_model_batch_predict_creates_monitor_when_none_exists(
    create_job, get_job, tmp_path
):
    """
    When Model Monitoring v2 is enabled and no ModelMonitor exists yet for this
    job, a new one is created and run against the batch job's output.
    """
    mock_monitor_instance = _reset_monitoring_mocks()
    mock_model = Model(
        uri=str(tmp_path / "model"),
        metadata={"resourceName": "projects/p/locations/l/models/123@1"},
    )
    gcp_resources_path = tmp_path / "gcp_resources.json"

    try:
        model_batch_predict(
            model=mock_model,
            job_display_name="predict-job",
            location="europe-west2",
            project="my-project",
            source_uri="bq://a.b.c",
            destination_uri="bq://a.b.d",
            source_format="bigquery",
            destination_format="bigquery",
            enable_monitoring=True,
            monitoring_training_gcs_uri=TRAINING_GCS_URI,
            monitored_features=MONITORED_FEATURES,
            notification_emails=["a@b.com"],
            gcp_resources=str(gcp_resources_path),
        )
    finally:
        gcp_resources_path.unlink(missing_ok=True)

    _mock_model_monitor_class.list.assert_called_once_with(
        project="my-project",
        location="europe-west2",
        filter='display_name="predict-job-monitoring"',
    )
    _mock_model_monitor_class.create.assert_called_once()
    create_kwargs = _mock_model_monitor_class.create.call_args.kwargs
    assert create_kwargs["model_name"] == "projects/p/locations/l/models/123"
    assert create_kwargs["model_version_id"] == "1"

    mock_monitor_instance.run.assert_called_once()
    run_kwargs = mock_monitor_instance.run.call_args.kwargs
    assert run_kwargs["display_name"] == "predict-job-monitoring-run"


@mock.patch(
    "google.cloud.aiplatform_v1beta1.services.job_service.JobServiceClient.create_batch_prediction_job",  # noqa : E501
    return_value=mock_job1,
)
@mock.patch(
    "google.cloud.aiplatform_v1beta1.services.job_service.JobServiceClient.get_batch_prediction_job",  # noqa : E501
    return_value=mock_job1,
)
def test_model_batch_predict_reuses_existing_monitor(create_job, get_job, tmp_path):
    """
    Re-running the pipeline for the same job_display_name must not create a
    duplicate ModelMonitor resource - the existing one should be reused.
    """
    _reset_monitoring_mocks()
    existing_monitor = mock.MagicMock()
    _mock_model_monitor_class.list.return_value = [existing_monitor]

    mock_model = Model(uri=str(tmp_path / "model"), metadata={"resourceName": "m@1"})
    gcp_resources_path = tmp_path / "gcp_resources.json"

    try:
        model_batch_predict(
            model=mock_model,
            job_display_name="predict-job",
            location="europe-west2",
            project="my-project",
            source_uri="bq://a.b.c",
            destination_uri="bq://a.b.d",
            source_format="bigquery",
            destination_format="bigquery",
            enable_monitoring=True,
            monitoring_training_gcs_uri=TRAINING_GCS_URI,
            monitored_features=MONITORED_FEATURES,
            gcp_resources=str(gcp_resources_path),
        )
    finally:
        gcp_resources_path.unlink(missing_ok=True)

    _mock_model_monitor_class.create.assert_not_called()
    existing_monitor.run.assert_called_once()
