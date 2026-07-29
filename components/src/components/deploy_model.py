from typing import NamedTuple

from kfp.dsl import Artifact, Input, Model, component


@component(
    base_image="python:3.12.8",
    packages_to_install=[
        "google-cloud-aiplatform==1.135.0",
        "google-cloud-pipeline-components==2.22.0",
    ],
)
def deploy_model(
    vertex_model: Input[Model],
    project: str,
    location: str,
    endpoint_name: str,
    service_account: str,
    machine_type: str = "n2-highcpu-2",
    traffic_percentage: int = 100,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
) -> NamedTuple("Outputs", [("endpoint_id", str)]):
    """
    Deploy a model to a Vertex AI endpoint using a rolling strategy.

    Finds an existing endpoint by display name or creates a new one, then
    deploys the model. If there are already models on the endpoint, the new
    model receives ``traffic_percentage`` and the previous model gets the
    remainder. Models older than the two most recent are undeployed.

    Args:
        vertex_model: VertexModel artifact from the upload_model component.
        project: GCP project ID.
        location: GCP region (e.g. europe-west2).
        endpoint_name: Display name for the endpoint.
        service_account: Service account for the deployed model.
        machine_type: Machine type for serving.
        traffic_percentage: Percentage of traffic for the new deployment.
        min_replica_count: Minimum replica count.
        max_replica_count: Maximum replica count.
    """

    import logging
    import google.cloud.aiplatform as aip

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
        logger.info("No endpoint found, creating: %s", endpoint_name)
        endpoint = aip.Endpoint.create(
            display_name=endpoint_name,
            project=project,
            location=location,
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

    return (endpoint_id,)
