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
            "mileage": 50392,
            "previous_owners_count": 1,
            "guide_valuation_clean": 16200.0,
            "guide_valuation_retail": 18600.0,
            "acceleration_value": 9.2,
            "doors_count": 5,
            "emissions_co2_g_per_km": 125,
            "engine_cc": 1499,
            "insurance_group": 15,
            "price_basic_value": 18500.0,
            "seats": 5,
            "top_speed_value": 125,
            "torque_nm": 200,
            "transmission_speeds": 6,
            "avg_number_of_bids_trend": 1.0,
            "avg_number_of_bids_trend_level_1": 1.0,
            "avg_max_dealer_offer_vs_cap_trend": 0.98,
            "avg_max_dealer_offer_vs_cap_trend_level_1": 0.98,
            "vehicles_on_sale_trend_level_1": 100,
            "tyres_condition": "none",
            "service_history": "full_main",
            "frontend_condition": "none",
            "interior_condition": "none",
            "nearside_condition": "none",
            "offside_condition": "none",
            "rearend_condition": "none",
            "roof_condition": "none",
            "warning_lights": "none",
            "windscreen_condition": "none",
            "wheels_condition": "1_minor",
            "mechanical_faults": "minor",
            "brand_slug": "volkswagen",
            "body_style_slug": "hatchback",
            "transmission_category": "manual",
            "fuel_category": "petrol",
            "postcode_region": "london",
            "urban_status": "urban",
            "colour": "grey",
            "photos_car_generic_count": 0,
            "photos_damage_count": 0,
            "photos_dashboard_count": 1,
            "photos_driver_front_side_count": 1,
            "photos_driver_front_side_tyre_count": 1,
            "photos_driver_rear_side_count": 1,
            "photos_driver_rear_side_tyre_count": 1,
            "photos_interior_front_count": 1,
            "photos_passenger_front_side_count": 1,
            "photos_passenger_front_side_tyre_count": 1,
            "photos_passenger_rear_side_count": 1,
            "photos_passenger_rear_side_tyre_count": 1,
            "photos_service_history_count": 0,
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
