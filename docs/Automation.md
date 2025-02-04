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

# Automation

This repo includes the following CI/CD pipelines which can be used in Cloud Build:

1. `pr-checks.yaml` - runs pre-commit checks and unit tests on the custom KFP components, and checks that the ML pipelines (training and prediction) can compile.
1. `e2e-test.yaml` - runs end-to-end tests of the training and prediction pipeline.
1. `release.yaml` - compiles training and prediction pipelines, then copies the compiled pipelines to the chosen GCS destination (versioned by git tag).
1. `terraform-plan.yaml` - Checks the Terraform configuration under `terraform/envs/<env>` (e.g. `terraform/envs/test`), and produces a summary of any proposed changes that will be applied on merge to the main branch.
1. `terraform-apply.yaml` - Applies the Terraform configuration under `terraform/envs/<env>` (e.g. `terraform/envs/test`).

## Choose the right project

We recommend to use a separate `admin` project, since the CI/CD pipelines operate across all the different environments (dev/test/prod).

Before you run the below commands, set the environment variables `GCP_PROJECT_ID` and `GCP_REGION` as follows:

```
export GCP_PROJECT_ID=my-gcp-project
export GCP_REGION=us-central1
```

## Create service accounts

Two service accounts are required

- One for running Cloud Build jobs
- One for running Vertex Pipelines - **already included in terraform**

Run the command below:

```
gcloud iam service-accounts create cloud-build \
--description="Service account for running Cloud Build" \
--display-name="Custom Cloud Build SA" \
--project=${GCP_PROJECT_ID}
```

## Create IAM permissions

The service account we have created for Cloud Build requires the following project roles:

- roles/logging.logWriter
- roles/storage.admin
- roles/aiplatform.user
- roles/artifactregistry.writer

```
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID --member="serviceAccount:cloud-build@${GCP_PROJECT_ID}.iam.gserviceaccount.com" --role="roles/logging.logWriter" --condition=None
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID --member="serviceAccount:cloud-build@${GCP_PROJECT_ID}.iam.gserviceaccount.com" --role="roles/storage.admin" --condition=None
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID --member="serviceAccount:cloud-build@${GCP_PROJECT_ID}.iam.gserviceaccount.com" --role="roles/aiplatform.user" --condition=None
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID --member="serviceAccount:cloud-build@${GCP_PROJECT_ID}.iam.gserviceaccount.com" --role="roles/artifactregistry.writer" --condition=None
```

It also requires the "Service Account User" role for the Vertex Pipelines service account ([docs here](https://cloud.google.com/iam/docs/impersonating-service-accounts#impersonate-sa-level)):

```
gcloud iam service-accounts add-iam-policy-binding vertex-pipelines@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
--member="serviceAccount:cloud-build@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
--role="roles/iam.serviceAccountUser" \
--project=${GCP_PROJECT_ID}
```

## Connect to GitHub

Follow the [Google Cloud documentation](https://cloud.google.com/build/docs/automating-builds/github/connect-repo-github?generation=2nd-gen) to connect the GitHub repository to Cloud Build.

## Set up triggers

There are three Cloud Build triggers to set up.

1. `pr-checks.yaml`
2. `trigger-tests.yaml` 
3. `e2e-test.yaml`

For each of the above, create a Cloud Build trigger with the following settings:

- Each one should be triggered on Pull Request to the `main` branch
- Enable comment control (select `Required` under `Comment Control`)
- Service account email: `cloud-build@<PROJECT ID>.iam.gserviceaccount.com`
- Configuration -> Type: `Cloud Build configuration file (yaml or json)`
- Configuration -> Location: Repository
- Cloud Build configuration file location: `cloudbuild/pr-checks.yaml` / `cloudbuild/trigger-tests.yaml` / `cloudbuild/e2e-test.yaml`
- Substitution variables - per table below

|  Cloud Build Trigger          |  Substitution variables             |
|-------------------------------|-------------------------------------|
| `pr-checks.yaml`              |                                     |
| `trigger-tests.yaml`          |                                     |
| `e2e-test.yaml`               |  _TEST_ENABLE_PIPELINE_CACHING = `False`<br>_TEST_VERTEX_LOCATION = `<GCP REGION (same as buckets etc above)>`<br>_TEST_VERTEX_PIPELINE_ROOT = `gs://<GCP PROJECT ID>-pl-root`<br>_TEST_VERTEX_PROJECT_ID = `<GCP PROJECT ID>`<br>_TEST_VERTEX_SA_EMAIL = `vertex-pipelines@<GCP PROJECT ID>.iam.gserviceaccount.com` |


## Recommended triggers

### On Pull Request to `main` / `master` branch

Set up a trigger for the `pr-checks.yaml` pipeline. 
We recommend to add `make pre-commit` (which is already part of the `Makefile`), to keep your ML use case code clean.
By default pull requests don't execute pre-commit hooks to improve the ease of use for new users of the template.

Set up a trigger for the `e2e-test.yaml` pipeline, and provide substitution values for the following variables:

| Variable | Description | Suggested value |
|---|---|---|
| `_TEST_VERTEX_CMEK_IDENTIFIER` | Optional. ID of the CMEK (Customer Managed Encryption Key) that you want to use for the ML pipeline runs in the E2E tests as part of the CI/CD pipeline with the format `projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key` | Leave blank |
| `_TEST_VERTEX_LOCATION` | The Google Cloud region where you want to run the ML pipelines in the E2E tests as part of the CI/CD pipeline. | Your chosen Google Cloud region |
| `_TEST_VERTEX_NETWORK` | Optional. The full name of the Compute Engine network to which the ML pipelines should be peered during the E2E tests as part of the CI/CD pipeline with the format `projects/<project number>/global/networks/my-vpc` |
| `_TEST_VERTEX_PIPELINE_ROOT` | The GCS folder (i.e. path prefix) that you want to use for the pipeline artifacts and for passing data between stages in the pipeline. Used during the pipeline runs in the E2E tests as part of the CI/CD pipeline. | `gs://<Project ID for dev environment>-pl-root` |
| `_TEST_VERTEX_PROJECT_ID` | Google Cloud project ID in which you want to run the ML pipelines in the E2E tests as part of the CI/CD pipeline. | Project ID for the DEV environment |
| `_TEST_VERTEX_SA_EMAIL` | Email address of the service account you want to use to run the ML pipelines in the E2E tests as part of the CI/CD pipeline. | `vertex-pipelines@<Project ID for dev environment>.iam.gserviceaccount.com` |
| `_TEST_ENABLE_PIPELINE_CACHING` | Override the default caching behaviour of the ML pipelines. Leave blank to use the default caching behaviour. | `False` |
| `_TEST_BQ_LOCATION` | The location of BigQuery datasets used in training and prediction pipelines. | `US` or `EU` if using multi-region datasets |

We recommend to enable comment control for this trigger (select `Required` under `Comment Control`). This will mean that the end-to-end tests will only run once a repository collaborator or owner comments `/gcbrun` on the pull request.
This will help to avoid unnecessary runs of the ML pipelines while a Pull Request is still being worked on, as they can take a long time (and can be expensive to run on every Pull Request!)

Set up three triggers for `terraform-plan.yaml` - one for each of the dev/test/prod environments. Set the Cloud Build substitution variables as follows:

| Environment | Cloud Build substitution variables |
|---|---|
| dev | **\_PROJECT_ID**=\<Google Cloud Project ID for the dev environment><br>**\_REGION**=\<Google Cloud region to use for the dev environment><br>**\_ENV_DIRECTORY**=terraform/envs/dev |
| test | **\_PROJECT_ID**=\<Google Cloud Project ID for the test environment><br>**\_REGION**=\<Google Cloud region to use for the test environment><br>**\_ENV_DIRECTORY**=terraform/envs/test |
| prod | **\_PROJECT_ID**=\<Google Cloud Project ID for the prod environment><br>**\_REGION**=\<Google Cloud region to use for the prod environment><br>**\_ENV_DIRECTORY**=terraform/envs/prod |

### On push of new tag

Set up a trigger for the `release.yaml` pipeline, and provide substitution values for the following variables:

| Variable | Description | Suggested value |
|---|---|---|
| `_PIPELINE_PUBLISH_AR_PATHS` | The (space separated) Artifact Registry repositories (plural!) where the compiled pipelines will be copied to - one for each environment (dev/test/prod). | `https://europe-west2-kfp.pkg.dev/<Project ID for dev environment>/vertex-pipelines https://europe-west2-kfp.pkg.dev/<Project ID for test environment>/vertex-pipelines https://europe-west2-kfp.pkg.dev/<Project ID for prod environment>/vertex-pipelines` |

### On merge to `main` / `master` branch

Set up three triggers for `terraform-apply.yaml` - one for each of the dev/test/prod environments. Set the Cloud Build substitution variables as follows:

| Environment | Cloud Build substitution variables |
|---|---|
| dev | **\_PROJECT_ID**=\<Google Cloud Project ID for the dev environment><br>**\_REGION**=\<Google Cloud region to use for the dev environment><br>**\_ENV_DIRECTORY**=terraform/envs/dev |
| test | **\_PROJECT_ID**=\<Google Cloud Project ID for the test environment><br>**\_REGION**=\<Google Cloud region to use for the test environment><br>**\_ENV_DIRECTORY**=terraform/envs/test |
| prod | **\_PROJECT_ID**=\<Google Cloud Project ID for the prod environment><br>**\_REGION**=\<Google Cloud region to use for the prod environment><br>**\_ENV_DIRECTORY**=terraform/envs/prod |
