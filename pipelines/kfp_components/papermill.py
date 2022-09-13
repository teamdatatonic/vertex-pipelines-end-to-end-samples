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

import yaml
import kfp

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


def generate_notebook_component(
    component_name: str,
    notebook: str,
    input_parameters: dict,
    output_parameters: dict,
):
    yml_tmpl = f"""
outputs:
- name: output_notebook
  type: HTML
implementation:
  container:
    image: us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu110:latest
    command:
    - sh
    - -c
    - |
      if ! [ -x "$(command -v pip)" ]; then
          python3 -m ensurepip || python3 -m ensurepip --user || apt-get install python3-pip
      fi
      PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --quiet --no-warn-script-location \
        'scrapbook==0.5.0' 'kfp==1.8.13' && "$0" "$@"
    - sh
    - -ec
    - |
      program_path=$(mktemp -d)
      printf "%s" "$0" > "$program_path/ephemeral_component.py"
      python3 -m kfp.v2.components.executor_main --component_module_path "$program_path/ephemeral_component.py" "$@"

    args:
    - --executor_input
    - executorInput: null
    - --function_to_execute
    - {component_name}
"""
    entrypoint = f"""
import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import *
from typing import *
{generate_notebook_component_definition(component_name, notebook, input_parameters, output_parameters)}
"""
    component = yaml.safe_load(yml_tmpl)

    component["inputs"] = [
        {"name": key, "type": python_type_to_kfp_type(value)}
        for (key, value) in input_parameters.items()
    ]
    component["outputs"].extend(
        [
            {"name": key, "type": python_type_to_kfp_type(value)}
            for (key, value) in output_parameters.items()
        ]
    )
    component["implementation"]["container"]["command"].append(entrypoint)

    component_yaml = yaml.dump(component)
    return kfp.components.load_component_from_text(component_yaml)


def python_type_to_kfp_type(type: type):
    if type == str:
        return "String"
    return type.__name__
