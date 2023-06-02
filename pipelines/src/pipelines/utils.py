from importlib import import_module
from pathlib import Path
from typing import Callable

from kfp.v2 import compiler
from jinja2 import Template


def generate_query(file_name: str, folder: Path = None, **replacements) -> str:
    """
    Read input file and replace placeholder using Jinja.

    Args:
        input_file (Path): input file to read
        replacements: keyword arguments to use to replace placeholders
    Returns:
        str: replaced content of input file
    """

    if folder is None:
        folder = Path(__file__).parent.parent.parent / "queries"

    with open(folder / file_name, "r") as f:
        query_template = f.read()

    return Template(query_template).render(**replacements)


def load_pipeline(module_name: str, function_name: str = "pipeline") -> Callable:
    module = import_module(f"pipelines.pipelines.{module_name}")
    return getattr(module, function_name)


def compile_pipeline(func: Callable, output_file: str):
    compiler.Compiler().compile(
        pipeline_func=func,
        package_path=output_file,
        type_check=False,
    )
