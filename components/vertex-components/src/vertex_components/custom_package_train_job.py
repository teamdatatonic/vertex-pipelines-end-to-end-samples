# Copyright 2022 Google LLC
from typing import Dict

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp.v2.dsl import Input, component, Metrics, Output, Artifact, Dataset


@component(
    base_image="python:3.7",
    packages_to_install=["google-cloud-aiplatform==1.24.1"],
)
def custom_package_train_job(
    python_package_uri: str,
    python_module_name: str,
    train_data: Input[Dataset],
    valid_data: Input[Dataset],
    test_data: Input[Dataset],
    project_id: str,
    project_location: str,
    model_display_name: str,
    train_container_uri: str,
    serving_container_uri: str,
    model: Output[Artifact],
    metrics: Output[Metrics],
    staging_bucket: str,
    job_name: str = None,
    hparams: Dict[str, str] = None,
    replica_count: int = 1,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED",
    accelerator_count: int = 0,
    parent_model: str = None,
):
    import json
    import logging
    import os.path
    import time
    import google.cloud.aiplatform as aip

    logging.info(f"Using train script: {python_package_uri}")
    package_path = "/gcs/" + python_package_uri[5:]
    if not os.path.exists(package_path):
        raise ValueError(
            "Train package was not found. "
            f"Check if the path is correct: {python_package_uri}"
        )

    job = aip.CustomPythonPackageTrainingJob(
        display_name=job_name if job_name else f"Custom job {int(time.time())}",
        python_package_gcs_uri=python_package_uri,
        python_module_name=python_module_name,
        project=project_id,
        location=project_location,
        staging_bucket=staging_bucket,
        container_uri=train_container_uri,
        model_serving_container_image_uri=serving_container_uri,
    )

    cmd_args = [
        f"--train_data={train_data.path}",
        f"--valid_data={valid_data.path}",
        f"--test_data={test_data.path}",
        f"--metrics={metrics.path}",
        f"--hparams={json.dumps(hparams if hparams else {})}",
    ]

    uploaded_model = job.run(
        model_display_name=model_display_name,
        parent_model=parent_model,
        is_default_version=(not parent_model),
        args=cmd_args,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
    )

    resource_name = f"{uploaded_model.resource_name}@{uploaded_model.version_id}"
    model.metadata["resourceName"] = resource_name
    model.metadata["containerSpec"] = {"imageUri": serving_container_uri}
    model.uri = uploaded_model.uri
    model.TYPE_NAME = "google.VertexModel"

    with open(metrics.path, "r") as fp:
        parsed_metrics = json.load(fp)

    logging.info(parsed_metrics)
    for k, v in parsed_metrics.items():
        if type(v) is float:
            metrics.log_metric(k, v)
