/**
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


terraform {
  required_version = ">= 0.13"
  required_providers {

    google = {
      source  = "hashicorp/google"
      version = "~> 4.56.0"
    }

    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.56.0"
    }

  }

  # Terraform state stored in GCS
  backend "gcs" {}
}

# Core Vertex Pipelines infrastructure
module "vertex_deployment" {
  source     = "../../modules/vertex_deployment"
  project_id = var.project_id
  region     = var.region
}

# BigQuery dataset for ML data
module "bigquery" {
  source      = "../../modules/bigquery"
  project_id  = var.project_id
  region      = var.region
  dataset_id  = var.dataset_id
  description = "Dataset for ML pipeline data"
  depends_on  = [module.vertex_deployment]
}
