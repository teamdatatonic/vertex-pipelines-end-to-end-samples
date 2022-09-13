from pipelines.kfp_components.papermill import generate_notebook_component

get_current_time = generate_notebook_component(
    "get_current_time",
    "gs://dt-harrycai-sandbox-dev-pipelines/pipelines/training/assets/notebooks/get_current_time.ipynb",
    input_parameters={"timestamp": str},
    output_parameters={"current_time": str},
)
