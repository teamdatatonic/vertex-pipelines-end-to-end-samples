# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load per-environment configuration from variables.yml."""

import os
import pathlib

import yaml


_ENVIRONMENTS = ("dev", "staging", "prod")


def detect_environment(config: dict) -> str:
    """Match VERTEX_PROJECT_ID against vertex_project_* values in config."""
    project_id = os.environ.get("VERTEX_PROJECT_ID", "")
    for env in _ENVIRONMENTS:
        if config.get(f"vertex_project_{env}") == project_id:
            return env
    raise ValueError(
        f"Could not detect environment from VERTEX_PROJECT_ID='{project_id}'. "
        f"No matching vertex_project_* entry found in variables.yml."
    )


def load_variables() -> dict:
    """Load variables.yml and return the block for the current environment."""
    config_path = (
        pathlib.Path(__file__).parent.parent.parent.parent
        / "variables"
        / "variables.yml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    environment = detect_environment(config)
    return config.get(environment, {})
