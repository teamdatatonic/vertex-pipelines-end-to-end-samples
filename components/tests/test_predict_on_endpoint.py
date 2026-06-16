import json
from unittest import mock

import pytest
from kfp.dsl import Artifact

import components

predict_on_endpoint = components.predict_on_endpoint.python_func


@mock.patch("google.cloud.aiplatform.Endpoint")
@mock.patch("google.cloud.aiplatform.init")
def test_predict_on_endpoint(mock_init, mock_endpoint_cls, tmp_path):
    mock_endpoint = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.predictions = [42.0]
    mock_endpoint.predict.return_value = mock_response
    mock_endpoint_cls.return_value = mock_endpoint

    predictions_artifact = Artifact(uri=str(tmp_path / "predictions.json"))

    predict_on_endpoint(
        endpoint_id="12345",
        project="my-project",
        predictions=predictions_artifact,
        location="europe-west2",
        num_requests=1,
    )

    mock_init.assert_called_once_with(project="my-project", location="europe-west2")
    mock_endpoint_cls.assert_called_once_with(
        endpoint_name="projects/my-project/locations/europe-west2/endpoints/12345",
    )
    mock_endpoint.predict.assert_called_once()
    instances_sent = mock_endpoint.predict.call_args[1]["instances"]
    assert len(instances_sent) == 1
    assert instances_sent[0]["mileage"] == 50392
    assert instances_sent[0]["colour"] == "grey"

    with open(predictions_artifact.path) as f:
        result = json.load(f)
    assert result == {"predictions": [42.0]}


@mock.patch("google.cloud.aiplatform.Endpoint")
@mock.patch("google.cloud.aiplatform.init")
def test_predict_on_endpoint_custom_location(mock_init, mock_endpoint_cls, tmp_path):
    mock_endpoint = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.predictions = [99.0]
    mock_endpoint.predict.return_value = mock_response
    mock_endpoint_cls.return_value = mock_endpoint

    predictions_artifact = Artifact(uri=str(tmp_path / "predictions.json"))

    predict_on_endpoint(
        endpoint_id="111",
        project="my-project",
        predictions=predictions_artifact,
        location="us-central1",
        num_requests=1,
    )

    mock_init.assert_called_once_with(project="my-project", location="us-central1")
    mock_endpoint_cls.assert_called_once_with(
        endpoint_name="projects/my-project/locations/us-central1/endpoints/111",
    )
