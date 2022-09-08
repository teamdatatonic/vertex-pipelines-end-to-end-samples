from typing import NamedTuple
from kfp.v2.dsl import Output, HTML, component


@component(
    base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu110:latest",
    packages_to_install=["scrapbook==0.5.0"],
)
def bq_query_to_table(
    output_notebook: Output[HTML],
    query: str,
    bq_client_project_id: str,
    destination_project_id: str,
    dataset_id: str,
    table_id: str,
    dataset_location: str,
    query_job_config: dict,
) -> NamedTuple("Outputs", []):
    import papermill
    import nbformat
    import nbconvert
    import scrapbook

    papermill_output = output_notebook.path.replace(".html", ".ipynb")
    try:
        # execute notebook using papermill
        papermill.execute_notebook(
            "gs://dt-harrycai-sandbox-dev-pipelines/pipelines/training/assets/notebooks/query_to_table.ipynb",
            papermill_output,
            parameters={
                "query": query,
                "bq_client_project_id": bq_client_project_id,
                "destination_project_id": destination_project_id,
                "dataset_id": dataset_id,
                "table_id": table_id,
                "dataset_location": dataset_location,
                "query_job_config": query_job_config,
            },
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
    return
