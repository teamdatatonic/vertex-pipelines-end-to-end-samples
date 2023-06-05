import logging
from importlib import import_module
from pathlib import Path
from typing import Callable

from kfp.v2 import compiler
from jinja2 import Template


def generate_query(file_name: str, folder: Path = None, **replacements) -> str:
    """
    Read input file and replace placeholder using Jinja.

    Args:
        file_name (str): input file to read
        folder (Path): folder which contains file (optional)
        replacements (dict): keyword arguments to use to replace placeholders
    Returns:
        str: replaced content of input file
    """

    if folder is None:
        folder = Path(__file__).parent.parent.parent / "queries"

    with open(folder / file_name, "r") as f:
        query_template = f.read()

    return Template(query_template).render(**replacements)


def load_pipeline(module_name: str, function_name: str = "pipeline") -> Callable:
    """
    Load pipeline by importing from a given module and function name.

    Args:
        module_name (str): name of module
        function_name (str): name of function

    Returns:
        Callable: imported pipeline function
    """
    name = f"pipelines.pipelines.{module_name}"
    logging.debug(f"import pipeline '{name}'")
    module = import_module(name)
    logging.debug(f"import function '{function_name}'")
    return getattr(module, function_name)


def compile_pipeline(func: Callable, output_file: str) -> None:
    """
    Compile pipeline into KubeFlow pipeline job json.

    Args:
        func: pipeline function
        output_file: pipeline job json
    """
    logging.info(f"compile pipeline to '{output_file}'")
    compiler.Compiler().compile(
        pipeline_func=func,
        package_path=output_file,
        type_check=False,
    )
