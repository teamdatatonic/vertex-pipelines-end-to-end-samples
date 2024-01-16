# MLOps Hackathon

*Learn about MLOps by deploying your own ML pipelines in Google Cloud. 
You'll solve a number of exercises and challenges to run pipelines in Vertex AI, continuously monitor your models, and promote your artifacts to a production environment.*

## Getting started 

As a hackathon attendee, simply follow this notebook series in your Vertex AI Workbench instance:

1. **[Health check](./hackathon/01_health_check.ipynb) - start here**
1. [Run pipelines](./hackathon/02_run_pipelines.ipynb)
1. [Promote model](./hackathon/03_promote_model.ipynb)
1. [Challenge: Model monitoring](./hackathon/04_monitoring_challenge.ipynb)
1. [Challenge: Real-time predictions](./hackathon/05_realtime_challenge.ipynb)

**❗Note:** This workshop has been designed to be run in Vertex AI Workbench. 
Support for running the workshop locally is provided, but we recommend Vertex AI Workbench for the best experience.

## For instructors

![Shell](https://github.com/teamdatatonic/vertex-pipelines-end-to-end-samples/wiki/images/shell.gif)

## Introduction

The notebooks are self-contained but instructors of this hackathon are asked to prepare the following for hackathon attendees.

1. Create 3x Google Cloud projects (dev, test, prod)
1. Use `make deploy` to deploy resources in each of them. It's advised to follow the [infrastructure setup notebook](./docs/notebooks/01_infrastructure_setup.ipynb) for each environment
1. Create an E2E test trigger in the test project
1. Create a release trigger in the prod project
1. Add each user with their own Google account with the following IAM roles:
    - `Vertex AI User` (roles/aiplatform.user)
    - `Storage Object Viewer` (roles/storage.objectViewer)
    - `Service Usage Consumer` (roles/serviceusage.serviceUsageConsumer)
1. Create one Vertex Workbench instance per user.
1. Confirm that users can access the GCP resources.
1. ❗Post workshop remember to delete all the users from the project and to clean up branches and releases in this repository