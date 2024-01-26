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

resource "google_service_account" "cloud_scheduler" {
  project      = var.project_id
  account_id   = "cloud-scheduler"
  display_name = "SA for Cloud Scheduler"
  description  = "SA for scheduled jobs"
}

resource "google_project_iam_member" "cloud_scheduler_iam" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cloud_scheduler.email}"
}

resource "google_service_account_iam_binding" "admin-account-iam" {
  service_account_id = var.pipeline_service_account_name
  role               = "roles/iam.serviceAccountUser"
  members = [
    "serviceAccount:${google_service_account.cloud_scheduler.email}",
  ]
}
