from pipelines.kfp_components.notebook_component import notebook_component

bq_query_to_table = notebook_component(
    "bq_query_to_table",
    "gs://dt-harrycai-sandbox-dev-pipelines/pipelines/training/assets/notebooks/query_to_table.ipynb",
    input_parameters={
        "query": str,
        "bq_client_project_id": str,
        "destination_project_id": str,
        "dataset_id": str,
        "table_id": str,
        "dataset_location": str,
        "query_job_config": dict,
    },
    output_parameters=dict(),
)
