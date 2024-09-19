
# CogFlow

Cogflow module sets up a pipeline for handling datasets and machine learning models
using multiple plugins. It includes functions for creating, registering, evaluating,
and serving models, as well as managing datasets.

**Key components include:**

Mlflow Plugin: For model tracking, logging, and evaluation.
Kubeflow Plugin: For pipeline management and serving models.
Dataset Plugin: For dataset registration and management.
Model Plugin: For saving model details.
Configurations: Constants for configuration like tracking URIs, database credentials, etc.

**Key Functions:**

**Model Management**

register_model: Register a new model.
log_model: Log a model.
load_model: Load a model.
delete_registered_model: Delete a registered model.
create_registered_model: Create a new registered model.
create_model_version: Create a new version of a registered model.


**Run Management**

start_run: Start a new.
end_run: End the current.
log_param: Log a parameter to the current run.
log_metric: Log a metric to the current run.


**Evaluation and Autologging**

evaluate: Evaluate a model.
autolog: Enable automatic logging of parameters, metrics, and models.


**Search and Query**

search_registered_models: Search for registered models.
search_model_versions: Search for model versions.
get_model_latest_version: Get the latest version of a registered model.
get_artifact_uri: Get the artifact URI of the current or specified run.


**Dataset Management**

link_model_to_dataset: Link a model to a dataset.
save_dataset_details: Save dataset details.
save_model_details_to_db: Save model details to the database.


**Pipeline and Component Management**

pipeline: Create a new Kubeflow pipeline.
create_component_from_func: Create a Kubeflow component from a function.
client: Get the Kubeflow client.
load_component_from_url: Load a Kubeflow component from a URL.


**Model Serving**

serve_model_v1: Serve a model using Kubeflow V1.
serve_model_v2: Serve a model using Kubeflow V2.
get_model_url: Get the URL of a served model.
delete_served_model: Delete a served model.


**MinIO Operations**

create_minio_client: Create a MinIO client.
query_endpoint_and_download_file: Query an endpoint and download a file from MinIO.
save_to_minio: Save file content to MinIO.
delete_from_minio: Delete an object from MinIO.


**Dataset Registration**

register_dataset: Register a dataset.
## Getting Started

To begin, import cogflow from the CogFlow module:

```python
import cogflow

```

### Explore the Capabilities of `cogflow`

- **List Attributes and Methods**: Understand the `cogflow` module better with:
    ```python
    print(dir(cogflow))
    ```

- **Get Documentation**: For a comprehensive guide on the `cogflow`, use:
    ```python
    help(cogflow)
    ```

## Environment Variables

To maximize the functionality of CogFlow, set the following environment variables:

- **Mlflow Configuration**:
    - `MLFLOW_TRACKING_URI`: The URI of the Mlflow tracking server.
    - `MLFLOW_S3_ENDPOINT_URL`: The endpoint URL for the AWS S3 service.
    - `ACCESS_KEY_ID`: The access key ID for AWS S3 authentication.
    - `SECRET_ACCESS_KEY`: The secret access key for AWS S3 authentication.

- **Machine Learning Database**:
    - `ML_DB_USERNAME`: Username for connecting to the machine learning database.
    - `ML_DB_PASSWORD`: Password for connecting to the machine learning database.
    - `ML_DB_HOST`: Host address for the machine learning database.
    - `ML_DB_PORT`: Port number for the machine learning database.
    - `ML_DB_NAME`: Name of the machine learning database.

- **CogFlow Database**:
    - `COGFLOW_DB_USERNAME`: Username for connecting to the CogFlow database.
    - `COGFLOW_DB_PASSWORD`: Password for connecting to the CogFlow database.
    - `COGFLOW_DB_HOST`: Host address for the CogFlow database.
    - `COGFLOW_DB_PORT`: Port number for the CogFlow database.
    - `COGFLOW_DB_NAME`: Name of the CogFlow database.

- **MinIO Configuration**:
    - `MINIO_ENDPOINT_URL`: The endpoint URL for the MinIO service.
    - `MINIO_ACCESS_KEY`: The access key for MinIO authentication.
    - `MINIO_SECRET_ACCESS_KEY`: The secret access key for MinIO authentication.

---

By setting the environment variables correctly, you can fully utilize the features and functionalities of the CogFlow framework for your cognitive and machine learning tasks.