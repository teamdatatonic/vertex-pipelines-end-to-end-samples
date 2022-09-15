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

import textwrap
import yaml
import kfp

DL_IMAGE_URI = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu110:latest"
)


def notebook_component(
    component_name: str,
    notebook: str,
    input_parameters: dict,
    output_parameters: dict,
):
    """
    A factory function that returns a component that runs a parameterised notebook.
    The notebook component uses papermill under the hood.
    See https://papermill.readthedocs.io/en/latest/usage-workflow.html
    Args:
        component_name (str): the name of the notebook component.
        notebook (str): the location of the parameterised notebook.
        input_parameters (dict): the input parameters of the notebook, as a python dict.
            Keys are the parameter names and values are their corresponding types.
            Eg. {'param_1': str, 'param_2': int}
        output_parameters (dict): the return values of the notebook, as a python dict.
            Keys are the return value names and values are their corresponding types.
            Eg. {'output_1': str, 'output_2': int}
    """
    return NotebookComponentFactory(
        component_name, notebook, input_parameters, output_parameters
    ).create_notebook_component()


class NotebookComponentFactory:
    def __init__(
        self,
        component_name: str,
        notebook: str,
        input_parameters: dict,
        output_parameters: dict,
    ):
        self.component_name = component_name
        self.notebook = notebook
        self.input_parameters = input_parameters
        self.output_parameters = output_parameters

    def create_notebook_component(self):
        """Creates the notebook component by assembling its yaml definition as a string then loading it."""
        # we start with the following yaml which contains the common attributes of all notebook components
        component_yml = textwrap.dedent(
            f"""\
            outputs:
            - name: output_notebook
              type: HTML
            implementation:
              container:
                image: {DL_IMAGE_URI}
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
                  python3 -m kfp.v2.components.executor_main \
                    --component_module_path "$program_path/ephemeral_component.py" "$@"
                args:
                - --executor_input
                - executorInput: null
                - --function_to_execute
                - {self.component_name}
            """
        )
        component = yaml.safe_load(component_yml)

        component["inputs"] = [
            {"name": key, "type": _python_type_to_kfp_type(value)}
            for (key, value) in self.input_parameters.items()
        ]
        component["outputs"].extend(
            [
                {"name": key, "type": _python_type_to_kfp_type(value)}
                for (key, value) in self.output_parameters.items()
            ]
        )
        component["implementation"]["container"]["command"].append(
            self._get_notebook_component_definition()
        )

        component_yaml = yaml.dump(component)
        return kfp.components.load_component_from_text(component_yaml)

    def _get_notebook_component_definition(self):
        """Returns the code for the underlying python function to be run by the notebook component."""
        kwargs_code = ", ".join(
            [f"{p}: {self.input_parameters[p].__name__}" for p in self.input_parameters]
        )
        pm_parameters_dict = (
            "{" + ", ".join([f"'{p}': {p}" for p in self.input_parameters]) + "}"
        )
        outputs = ",".join(
            [
                f"('{p}', {self.output_parameters[p].__name__})"
                for p in self.output_parameters
            ]
        )
        returns = (
            "("
            + ",".join([f"scraps.data_dict['{p}']" for p in self.output_parameters])
            + ",)"
            if self.output_parameters
            else ""
        )
        return textwrap.dedent(
            f"""\
            from typing import NamedTuple
            from kfp.v2.dsl import Output, HTML, component

            def {self.component_name}(
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
                        "{self.notebook}",
                        papermill_output,
                        parameters={pm_parameters_dict}
                    )
                # no except block because we want errors to be handled by the pipeline runner
                # and the component to show as failed
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
        )


def _python_type_to_kfp_type(type: type):
    """
    Converts python types to their equivalent type when defined in a kfp component spec.
    Eg. str -> String
    """
    if type == str:
        return "String"
    return type.__name__
