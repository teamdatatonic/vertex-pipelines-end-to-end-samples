from typing import NamedTuple
from kfp.v2.dsl import Output, HTML, component


@component(
    base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu110:latest",
    packages_to_install=["scrapbook==0.5.0"],
)
def get_current_time(
    output_notebook: Output[HTML], timestamp: str
) -> NamedTuple("Outputs", [("current_time", str)]):
    import papermill
    import nbformat
    import nbconvert
    import scrapbook

    papermill_output = output_notebook.path.replace(".html", ".ipynb")
    try:
        # execute notebook using papermill
        papermill.execute_notebook(
            "gs://dt-harrycai-sandbox-dev-pipelines/pipelines/training/assets/notebooks/get_current_time.ipynb",
            papermill_output,
            parameters={"timestamp": timestamp},
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
    return (scraps.data_dict["current_time"],)
