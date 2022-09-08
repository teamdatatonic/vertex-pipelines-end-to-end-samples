# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp.v2.dsl import Output, HTML, component

DL_IMAGE_URI = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu110:latest"
)


def generate_notebook_component_definition(
    component_name: str,
    notebook: str,
    input_parameters: dict,
    output_parameters: dict,
) -> str:
    kwargs_code = ", ".join(
        [f"{p}: {input_parameters[p].__name__}" for p in input_parameters]
    )
    pm_parameters_dict = (
        "{" + ", ".join([f"'{p}': {p}" for p in input_parameters]) + "}"
    )
    outputs = ",".join(
        [f"('{p}', {output_parameters[p].__name__})" for p in output_parameters]
    )
    returns = (
        "(" + ",".join([f"scraps.data_dict['{p}']" for p in output_parameters]) + ",)"
        if output_parameters
        else ""
    )
    return f"""
from typing import NamedTuple
from kfp.v2.dsl import Output, HTML, component


@component(base_image="{DL_IMAGE_URI}", packages_to_install=["scrapbook==0.5.0"])
def {component_name}(
    output_notebook: Output[HTML],
    {kwargs_code}
) -> NamedTuple("Outputs", [{outputs}]):
    import papermill
    import nbformat
    import nbconvert
    import scrapbook
    papermill_output = output_notebook.path.replace('.html', '.ipynb')
    try:
        # execute notebook using papermill
        papermill.execute_notebook(
            "{notebook}",
            papermill_output,
            parameters={pm_parameters_dict}
        )
    # no except block; we want errors to be handled by the pipeline runner, and the component to show as failed
    finally:
        # read output notebook and export to html
        with open(papermill_output) as f:
            nb = nbformat.read(f, as_version=4)
        html_exporter = nbconvert.HTMLExporter()
        html_data, resources = html_exporter.from_notebook_node(nb)
        with open(output_notebook.path, "w") as f:
            f.write(html_data)
    scraps = scrapbook.read_notebook(papermill_output).scraps
    return {returns}
    """


@component(base_image=DL_IMAGE_URI, packages_to_install=["scrapbook==0.5.0"])
def run_notebook(notebook: str, output_notebook: Output[HTML], **kwargs) -> dict:
    import papermill
    import nbformat
    import nbconvert
    import scrapbook

    papermill_output = output_notebook.path.replace(".html", ".ipynb")

    try:
        # execute notebook using papermill
        papermill.execute_notebook(notebook, papermill_output, parameters=kwargs)

    # no except block; we want errors to be handled by the pipeline runner, and the component to show as failed
    finally:
        # read output notebook and export to html
        with open(papermill_output) as f:
            nb = nbformat.read(f, as_version=4)
        html_exporter = nbconvert.HTMLExporter()
        html_data, resources = html_exporter.from_notebook_node(nb)
        with open(output_notebook.path, "w") as f:
            f.write(html_data)

    scraps = scrapbook.read_notebook(papermill_output).scraps
    return scraps.data_dict


if __name__ == "__main__":
    """
    I've temporarily made this a python script where I'm generating component definitions for the existing pipelines.
    TODO: in future, this should be a callable library or CLI, not a python script.
    """
    notebook_component_definition = generate_notebook_component_definition(
        "get_current_time",
        "gs://dt-harrycai-sandbox-dev-pipelines/pipelines/training/assets/notebooks/get_current_time.ipynb",
        {"timestamp": str},
        {"current_time": str},
    )
    with open("pipelines/kfp_components/aiplatform/get_current_time.py", "w") as f:
        f.write(notebook_component_definition)

    notebook_component_definition = generate_notebook_component_definition(
        "bq_query_to_table",
        "gs://dt-harrycai-sandbox-dev-pipelines/pipelines/training/assets/notebooks/query_to_table.ipynb",
        {
            "query": str,
            "bq_client_project_id": str,
            "destination_project_id": str,
            "dataset_id": str,
            "table_id": str,
            "dataset_location": str,
            "query_job_config": dict,
        },
        {},
    )
    with open("pipelines/kfp_components/bigquery/query_to_table.py", "w") as f:
        f.write(notebook_component_definition)
