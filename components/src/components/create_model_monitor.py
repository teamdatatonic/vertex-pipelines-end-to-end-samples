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

from typing import NamedTuple

from kfp.dsl import Input, Model, component


@component(
    base_image="python:3.12.8",
    packages_to_install=["google-cloud-aiplatform==1.135.0"],
)
def create_model_monitor(
    vertex_model: Input[Model],
    project: str,
    location: str,
    display_name: str,
    monitored_features: dict,
) -> NamedTuple("Outputs", [("model_monitor_name", str)]):
    """
    Create or reuse a Vertex AI Model Monitoring v2 ModelMonitor.

    Lists existing monitors by display name and reuses the first match.
    If none match the name, reuses a monitor already attached to the same
    model version (Vertex allows only one). A new monitor is created only
    when neither exists.

    Args:
        vertex_model: Vertex Model artifact with a resourceName in metadata.
        project: GCP project ID.
        location: GCP region (e.g. europe-west2).
        display_name: Display name used to find or create the monitor.
        monitored_features: Maps each feature name to its Model Monitoring v2
            schema data type ("float", "integer", "boolean", "string",
            "categorical").
    """

    import logging

    import vertexai
    from vertexai.resources.preview import ml_monitoring
    from vertexai.resources.preview.ml_monitoring.spec import schema as schema_spec

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def _monitor_model_target(monitor):
        gca = getattr(monitor, "gca_resource", None) or getattr(
            monitor, "_gca_resource", None
        )
        sources = [src for src in (gca, monitor) if src is not None]
        for src in sources:
            target = getattr(src, "model_monitoring_target", None)
            vertex = (
                getattr(target, "vertex_model", None) if target is not None else None
            )
            model = getattr(vertex, "model", None) if vertex is not None else None
            version = (
                getattr(vertex, "model_version_id", None)
                if vertex is not None
                else None
            )
            if isinstance(model, str) and model:
                return model, str(version or "")
        return "", ""

    vertexai.init(project=project, location=location)

    resource_name = vertex_model.metadata["resourceName"]
    metadata_version = vertex_model.metadata.get("versionId")
    if "@" in resource_name:
        model_name, model_version_id = resource_name.rsplit("@", 1)
    else:
        model_name = resource_name
        model_version_id = str(metadata_version or "1")

    def _same_model_version(monitor) -> bool:
        candidate_model, candidate_version = _monitor_model_target(monitor)
        if not candidate_model:
            return True
        return candidate_model == model_name and candidate_version == str(
            model_version_id
        )

    existing_monitors = ml_monitoring.ModelMonitor.list(
        project=project,
        location=location,
        filter=f'display_name="{display_name}"',
    )
    if existing_monitors:
        monitor = existing_monitors[0]
        if _same_model_version(monitor):
            logger.info("Reusing existing ModelMonitor: %s", monitor.resource_name)
            return (monitor.resource_name,)
        logger.info(
            "Monitor %s has display name %s but targets a different model "
            "version; not reusing.",
            monitor.resource_name,
            display_name,
        )

    # Vertex allows only one ModelMonitor per model version. An older batch
    # monitor on the same champion must be reused rather than creating a
    # second one under the endpoint display name.
    for candidate in ml_monitoring.ModelMonitor.list(
        project=project,
        location=location,
    ):
        candidate_model, candidate_version = _monitor_model_target(candidate)
        if candidate_model == model_name and candidate_version == str(model_version_id):
            logger.info(
                "Reusing ModelMonitor already attached to this model version: "
                "%s (display_name=%s)",
                candidate.resource_name,
                getattr(candidate, "display_name", ""),
            )
            return (candidate.resource_name,)

    logger.info("Creating new ModelMonitor: %s", display_name)
    model_monitoring_schema = schema_spec.ModelMonitoringSchema(
        feature_fields=[
            schema_spec.FieldSchema(name=name, data_type=data_type)
            for name, data_type in monitored_features.items()
        ]
    )
    monitor = ml_monitoring.ModelMonitor.create(
        project=project,
        location=location,
        display_name=display_name,
        model_name=model_name,
        model_version_id=model_version_id,
        model_monitoring_schema=model_monitoring_schema,
    )
    logger.info("Created ModelMonitor: %s", monitor.resource_name)
    return (monitor.resource_name,)
