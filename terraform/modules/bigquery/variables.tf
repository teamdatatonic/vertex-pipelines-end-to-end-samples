variable "project_id" {
  description = "The ID of the Google Cloud project."
  type        = string
}

variable "region" {
  description = "Google Cloud region for the dataset."
  type        = string
}

variable "dataset_id" {
  description = "The BigQuery dataset ID."
  type        = string
}

variable "description" {
  description = "Description of the BigQuery dataset."
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to apply to the dataset."
  type        = map(string)
  default     = {}
}
