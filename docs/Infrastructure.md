<!-- 
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 -->
# Infrastructure

We recommend using [`tfswitch`](https://tfswitch.warrensbox.com/) to automatically choose and download an appropriate version for you (run `tfswitch` from the [`terraform/envs/dev`](terraform/envs/dev/) directory).

The cloud infrastructure is managed using Terraform and is defined in the [`terraform`](terraform) directory. The Terraform modules used by each environment under [`terraform/envs`](terraform/envs/) are:

- `vertex_deployment` - deploys Cloud infrastructure required for running Vertex Pipelines, including enabling APIs, creating buckets, Artifact Registry repos, service accounts, and IAM permissions.
- `bigquery` - deploys BigQuery resources used by the example pipelines.

There is a Terraform configuration for each environment (dev/test/prod) under [`terraform/envs`](terraform/envs/).

## Schedule pipelines

Vertex Pipelines are scheduled with the Vertex AI [`PipelineJobSchedule`](https://cloud.google.com/vertex-ai/docs/pipelines/schedule-pipeline-run) API.

Scheduler settings live in [`pipelines/variables/variables.yml`](../pipelines/variables/variables.yml) under each environment (`dev`, `staging`, `prod`). When you run a training or prediction pipeline (for example via `make training` / `make prediction`), [`pipelines/src/pipelines/utils/trigger_pipeline.py`](../pipelines/src/pipelines/utils/trigger_pipeline.py) loads the config for the current environment and creates or removes the schedule.

| Setting | Description |
|---------|-------------|
| `enable_training_scheduler` / `enable_prediction_scheduler` | Set to `true` to create a schedule, `false` to remove any existing one |
| `training_cron` / `prediction_cron` | Cron expression for the schedule (see [crontab.guru](https://crontab.guru/)) |
| `training_max_concurrent_run_count` / `prediction_max_concurrent_run_count` | Max number of pipeline runs that can execute concurrently |
| `training_max_run_count` / `prediction_max_run_count` | Total number of runs before the schedule is paused (`0` for infinite) |

When scheduling is enabled, any previous schedule for that pipeline type is deleted before the new one is created, so only one active schedule exists at a time.

Example:

```yaml
scheduler:
  enable_training_scheduler: true
  training_cron: "0 0 1 * *"  # first of every month
  training_max_concurrent_run_count: 1
  training_max_run_count: 0

  enable_prediction_scheduler: true
  prediction_cron: "0 0 * * *"  # every night
  prediction_max_concurrent_run_count: 1
  prediction_max_run_count: 0
```

## Tear down infrastructure

To tear down the infrastructure you have created with Terraform, run these commands:

```bash
make undeploy env=dev VERTEX_PROJECT_ID=<DEV PROJECT ID>
```
