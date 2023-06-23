# Copyright 2022 Google LLC
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
import os

import pytest

from pipelines import training, prediction
from e2e.utils import assert_pipeline

enable_caching = os.environ.get("ENABLE_PIPELINE_CACHING")


@pytest.mark.training
def test_training_pipeline() -> None:
    """Tests if pipeline is run successfully."""
    assert_pipeline(
        training.pipeline,
        enable_caching=enable_caching,
        common_tasks={},
    )


@pytest.mark.prediction
def test_prediction_pipeline() -> None:
    """Tests if pipeline is run successfully."""
    assert_pipeline(
        prediction.pipeline,
        enable_caching=enable_caching,
        common_tasks={},
    )
