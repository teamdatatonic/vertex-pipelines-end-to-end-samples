from kfp.dsl import Artifact, Output, component


@component(
    base_image="python:3.12.8",
    packages_to_install=["google-cloud-aiplatform==1.135.0"],
)
def predict_on_endpoint(
    endpoint_id: str,
    project: str,
    predictions: Output[Artifact],
    location: str = "europe-west2",
    num_requests: int = 1500,
) -> None:
    """
    Send smoke-test prediction requests to a Vertex AI endpoint and save
    the results. Sends multiple requests so model monitoring has enough
    logged traffic to produce plots.

    Args:
        endpoint_id: Vertex AI endpoint ID.
        project: GCP project ID.
        predictions: Output artifact where prediction results are saved.
        location: GCP region (default: europe-west2).
        num_requests: Number of prediction requests to send.
    """

    import json
    import logging

    import google.cloud.aiplatform as aip
    from google.api_core.exceptions import InternalServerError

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    instances = [
        {
            "dayofweek": 4.0,
            "hourofday": 14.0,
            "trip_distance": 3210.5,
            "trip_miles": 2.1,
            "trip_seconds": 720.0,
            "payment_type": "Credit Card",
            "company": "Flash Cab",
        }
    ]

    aip.init(project=project, location=location)

    endpoint = aip.Endpoint(
        endpoint_name=f"projects/{project}/locations/{location}/endpoints/{endpoint_id}",
    )

    logger.info(
        "Sending %d request(s) (1 instance each) to endpoint %s for smoke test and monitoring traffic",
        num_requests,
        endpoint_id,
    )
    all_predictions = []
    max_retries = 3
    for i in range(num_requests):
        for attempt in range(max_retries):
            try:
                response = endpoint.predict(instances=instances)
                all_predictions.extend(response.predictions)
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info("Request %d/%d succeeded", i + 1, num_requests)
                break
            except InternalServerError as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "Prediction returned 500 (request %d, attempt %s/%s), retrying: %s",
                    i + 1, attempt + 1, max_retries, e,
                )

    result = {"predictions": all_predictions}
    with open(predictions.path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Saved %d prediction(s) from %d requests to %s", len(all_predictions), num_requests, predictions.path)
