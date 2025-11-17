"""
Cogflow module sets up a pipeline for handling datasets and machine learning models
using multiple plugins. It includes functions for creating, registering, evaluating,
and serving models, as well as managing datasets.

Key components include:

Mlflow Plugin: For model tracking, logging, and evaluation.
Kubeflow Plugin: For pipeline management and serving models.
Dataset Plugin: For dataset registration and management.
Model Plugin: For saving model details.
Configurations: Constants for configuration like tracking URIs, database credentials, etc.

Key Functions:

Model Management:

register_model: Register a new model.
log_model: Log a model.
load_model: Load a model.
delete_registered_model: Delete a registered model.
create_registered_model: Create a new registered model.
create_model_version: Create a new version of a registered model.

Run Management

start_run: Start a new.
end_run: End the current.
log_param: Log a parameter to the current run.
log_metric: Log a metric to the current run.

Evaluation and Autologging

evaluate: Evaluate a model.
autolog: Enable automatic logging of parameters, metrics, and models.

Search and Query

search_registered_models: Search for registered models.
search_model_versions: Search for model versions.
get_model_latest_version: Get the latest version of a registered model.
get_artifact_uri: Get the artifact URI of the current or specified run.

Dataset Management

link_model_to_dataset: Link a model to a dataset.
save_dataset_details: Save dataset details.
save_model_details_to_db: Save model details to the database.

Pipeline and Component Management

pipeline: Create a new Kubeflow pipeline.
create_component_from_func: Create a Kubeflow component from a function.
client: Get the Kubeflow client.
load_component: Load a Kubeflow component from a URL/file/text.

Model Serving

serve_model_v1: Serve a model using Kubeflow V1.
serve_model_v2: Serve a model using Kubeflow V2.
get_model_url: Get the URL of a served model.
delete_served_model: Delete a served model.

MinIO Operations

create_minio_client: Create a MinIO client.
query_endpoint_and_download_file: Query an endpoint and download a file from MinIO.
save_to_minio: Save file content to MinIO.
delete_from_minio: Delete an object from MinIO.

Dataset Registration

register_dataset: Register a dataset.
"""

import inspect
import json
import os
from collections import defaultdict
from datetime import datetime
from types import FunctionType
from typing import Callable, Union, Any, List, Optional, Dict, Mapping
import time
from uuid import UUID

import boto3
import yaml
from botocore.exceptions import NoCredentialsError, ClientError
import psutil
import numpy as np
import pandas as pd
import requests
from kfp_server_api import ApiException
from mlflow.models import ModelSignature, ModelInputExample
from scipy.sparse import csr_matrix, csc_matrix
from kfp.components import InputPath, OutputPath
from kfp.dsl import ParallelFor
from .kafka.consumer import stop_consumer, start_consumer_thread
from .plugins.message_broker_dataset_plugin import MessageBrokerDatasetPlugin
from .v2 import *
from . import pluginmanager, plugin_config
from .plugin_config import (
    TRACKING_URI,
    TIMER_IN_SEC,
    ML_TOOL,
    ACCESS_KEY_ID,
    SECRET_ACCESS_KEY,
    S3_ENDPOINT_URL,
    MINIO_ENDPOINT_URL,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_ACCESS_KEY,
    API_BASEPATH,
)
from .pluginmanager import PluginManager, ConfigException
from .plugins.component_plugin import ComponentPlugin
from .plugins.dataset_plugin import DatasetMetadata, DatasetPlugin
from .plugins.kubeflowplugin import CogContainer, KubeflowPlugin
from .plugins.knative_plugin import KnativePlugin
from .plugins.mlflowplugin import MlflowPlugin
from .plugins.notebook_plugin import NotebookPlugin
from .util import (
    make_post_request,
    is_valid_s3_uri,
    uuid_to_hex,
    uuid_to_canonical,
    make_get_request,
)

pyfunc = MlflowPlugin().pyfunc
mlflow = MlflowPlugin().mlflow
sklearn = MlflowPlugin().sklearn
cogclient = MlflowPlugin().cogclient
tensorflow = MlflowPlugin().tensorflow
pytorch = MlflowPlugin().pytorch
models = MlflowPlugin().models
lightgbm = MlflowPlugin().lightgbm
xgboost = MlflowPlugin().xgboost

add_model_access = CogContainer().add_model_access
kfp = KubeflowPlugin().kfp


def create_minio_client():
    """
    Creates a MinIO client object.

    Returns:
        Minio: The MinIO client object.
    """
    return DatasetPlugin().create_minio_client()


def query_endpoint_and_download_file(url, output_file, bucket_name):
    """
    Queries an endpoint and downloads a file from MinIO.

    Args:
        url (str): The URL to query.
        output_file (str): The output file path.
        bucket_name (str): The MinIO bucket name.

    Returns:
        bool: True if the file was successfully downloaded, False otherwise.
    """
    return DatasetPlugin().query_endpoint_and_download_file(
        url=url, output_file=output_file, bucket_name=bucket_name
    )


def save_to_minio(file_content, output_file, bucket_name):
    """
    Saves file content to MinIO.

    Args:
        file_content (bytes): The content of the file to save.
        output_file (str): The output file path.
        bucket_name (str): The MinIO bucket name.

    Returns:
        bool: True if the file was successfully saved, False otherwise.
    """
    return DatasetPlugin().save_to_minio(
        file_content=file_content, output_file=output_file, bucket_name=bucket_name
    )


def delete_from_minio(object_name, bucket_name):
    """
    Deletes an object from MinIO.

    Args:
        object_name (str): The name of the object to delete.
        bucket_name (str): The MinIO bucket name.

    Returns:
        bool: True if the object was successfully deleted, False otherwise.
    """
    return DatasetPlugin().delete_from_minio(
        object_name=object_name, bucket_name=bucket_name
    )


def register_dataset(
    dataset_type: int, name: str, file_path: str, description: str = None
):
    """
    Register a dataset by uploading a file.

    Parameters:
    -----------
    dataset_type : int
        Type of the dataset:
        - 0: Training dataset
        - 1: Inference dataset
        - 2: Both training and inference
    name : str
        Name of the dataset to register.
    file_path : str
        Full path to the dataset file to upload.
    description : str, optional
        A brief description of the dataset. Defaults to None.

    Returns:
    --------
    dict
        JSON response from the API.

    Example:
    --------
    >>> result = register_dataset(
    ...     dataset_type=0,
    ...     name="train dataset",
    ...     description="First training dataset",
    ...     file_path="/home/Dataset...."
    ... )
    >>> print(result)
    """
    return DatasetPlugin().register_dataset(
        dataset_type=dataset_type,
        name=name,
        file_path=file_path,
        description=description,
    )


def get_dataset(dataset_id: int, endpoint: str = plugin_config.DATASETS):
    """
    Generic method to call dataset API endpoints like /datasets/prometheus/{id}.

    :param dataset_id: Dataset ID to fetch
    :param endpoint: API endpoint path (e.g., "/datasets/prometheus")
    :return: API JSON response
    """
    return DatasetPlugin().get_dataset(dataset_id=dataset_id, endpoint=endpoint)


def download_dataset(dataset_id: int, output_file_path: str = None):
    """
    Downloads a dataset by its ID.

    Args:
        dataset_id (int): The ID of the dataset to download.
        output_file_path (str, optional): The path to save the downloaded dataset file.
            If not provided, a default filename will be used.

    Returns:
        str: The path to the downloaded dataset file.
    """
    return DatasetPlugin().download_dataset(
        dataset_id=dataset_id, output_file_path=output_file_path
    )


def delete_registered_model(model_name):
    """
    Deletes a registered model.

    Args:
        model_name (str): The name of the model to delete.

    Returns:
        bool: True if the model was successfully deleted, False otherwise.
    """
    return MlflowPlugin().delete_registered_model(model_name=model_name)


def evaluate(
    data,
    *,
    model_uri: str,
    targets,
    model_type: str,
    dataset_path=None,
    feature_names: list = None,
    evaluators=None,
    evaluator_config=None,
    custom_metrics=None,
    custom_artifacts=None,
    validation_thresholds=None,
    baseline_model=None,
    env_manager=plugin_config.ENV_MANAGER,
):
    """
    Evaluates a model.

    Args:
        model_uri (str): The URI of the model.
        data: The data to evaluate the model on.
        model_type: The type of the model.
        targets: The targets of the model.
        dataset_path: The path to the dataset.
        feature_names: The names of the features.
        evaluators: The evaluators to use.
        evaluator_config: The configuration for the evaluator.
        custom_metrics: Custom metrics to use.
        custom_artifacts: Custom artifacts to use.
        validation_thresholds: Validation thresholds to use.
        baseline_model: The baseline model to compare against.
        env_manager: The environment manager to use.

    Returns:
        dict: The evaluation results.
    """
    result = MlflowPlugin().evaluate(
        model=model_uri,
        data=data,
        model_type=model_type,
        targets=targets,
        dataset_path=dataset_path,
        feature_names=feature_names,
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        custom_metrics=custom_metrics,
        custom_artifacts=custom_artifacts,
        validation_thresholds=validation_thresholds,
        baseline_model=baseline_model,
        env_manager=env_manager,
    )

    PluginManager().load_config()
    time_out = plugin_config.TIME_OUT
    # Construct URLs
    run_id = model_uri.split("/")[4]
    model_id = uuid_to_canonical(run_id)
    url_metrics = (
        os.getenv(plugin_config.API_BASEPATH)
        + f"/models/{model_id}"
        + PluginManager().load_path("validation_metrics")
    )
    url_artifacts = (
        os.getenv(plugin_config.API_BASEPATH)
        + f"/models/{model_id}"
        + PluginManager().load_path("validation_artifacts")
    )
    # Capture final CPU and memory usage metrics
    final_cpu_percent = psutil.cpu_percent(interval=1)
    final_memory_info = psutil.virtual_memory()
    final_memory_used_mb = round(final_memory_info.used / (1024**2), 2)  # Convert to MB

    # Attempt to make POST requests, continue regardless of success or failure
    headers = {"kubeflow-userid": KubeflowPlugin().get_current_user_from_namespace()}
    try:
        metrics = result.metrics
        metrics.update(
            {
                "cpu_consumption": final_cpu_percent,
                "memory_utilization": final_memory_used_mb,
            }
        )

        requests.post(url=url_metrics, json=metrics, headers=headers, timeout=time_out)

    except Exception as exp:
        print(f"Failed to post metrics: {exp}")

    serialized_artifacts = NotebookPlugin().serialize_artifacts(result.artifacts)
    # Now you can use serialized_artifacts in your HTTP request
    try:
        requests.post(
            url=url_artifacts,
            json=serialized_artifacts,
            headers=headers,
            timeout=time_out,
        )
    except Exception as exp:
        print(f"Failed to post artifacts: {exp}")

    return result


def search_registered_models(
    filter_string: Optional[str] = None,
    max_results: int = plugin_config.MAX_RESULTS,
    order_by: Optional[List[str]] = None,
    page_token: Optional[str] = None,
):
    """
    Searches for registered models.

    This method allows you to search for registered models using optional filtering criteria,
    and retrieve a list of registered models that match the specified criteria.

    Args:
        filter_string (Optional[str]): A string used to filter the registered models. The filter
                string can include conditions on model name, tags and other attributes. For example,
                "name='my_model' AND tags.key='value'". If not provided, all registered
                models are returned.
        max_results (int): The maximum number of results to return. Defaults to 100.
        order_by (Optional[List[str]]): A list of property keys to order the results by.
            For example, ["name ASC", "version DESC"].
        page_token (Optional[str]): A token to specify the page of results to retrieve. This is
            useful for pagination when there are more results than can be returned in a single call.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a registered model that
        matches the search criteria. Each dictionary contains details about the registered model,
        such as its name, creation timestamp, last updated timestamp, tags, and description.
    """
    PluginManager().load_config()
    return MlflowPlugin().search_registered_models(
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


def load_model(model_uri: str, dst_path=None):
    """
    Loads a model from the specified URI.

    Args:
        model_uri (str): The URI of the model to load.
        dst_path (str, optional): The destination path to load the model to.

    Returns:
        Any: The loaded model object.
    """
    PluginManager().load_config()
    return MlflowPlugin().load_model(model_uri=model_uri, dst_path=dst_path)


def register_model(
    model_uri: str,
    model_name: str,
    await_registration_for: int = plugin_config.AWAIT_REGISTRATION_FOR,
    *,
    tags: Optional[Dict[str, Any]] = None,
):
    """
    Registers the given model with Mlflow.

    This method registers a model with Mlflow using the specified model URI. Optionally,
    tags can be added to the registered model for better organization and metadata tracking.

    Args:
        model_uri (str): The URI of the Mlflow model to register.
        model_name (str): The name under which to register the model in the Mlflow Model Registry.
        await_registration_for (int, optional): The duration, in seconds, to wait for the model
        version to finish being created and be in the READY status. Defaults to 300 seconds.
        tags (Optional[Dict[str, Any]], optional): A dictionary of key-value pairs to tag the
        registered model with. Tags can be useful for organizing and filtering models in the
         registry.

    Returns:
        ModelVersion: An instance of `ModelVersion` representing the registered model version.
    """
    PluginManager().load_config()
    return MlflowPlugin().register_model(
        model=model_name,
        model_uri=model_uri,
        await_registration_for=await_registration_for,
        tags=tags,
    )


def autolog():
    """
    Enables automatic logging of parameters, metrics, and models.
    """
    PluginManager().load_config()
    return MlflowPlugin().autolog()


def create_registered_model(
    model: str, tags: Optional[Dict[str, Any]] = None, description: Optional[str] = None
):
    """
    Create a registered model in the Mlflow Model Registry.

    This method creates a new registered model in the Mlflow Model Registry with the given name.
    Optionally, tags and a description can be added to provide additional metadata about the model.

    Args:
        model (str): The name of the registered model.
        tags (Optional[Dict[str, Any]], optional): A dictionary of key-value pairs to tag
        the registered model with. Tags can be useful for organizing and filtering models in the
        registry.
        description (Optional[str], optional): A description of the registered model. This can
        provide additional context about the model's purpose, usage, or any other relevant
        information.

    Returns:
        RegisteredModel: An instance of `RegisteredModel` representing the created registered model.
    """
    PluginManager().load_config()
    return MlflowPlugin().create_registered_model(
        model=model, tags=tags, description=description
    )


def create_model_version(
    model: str,
    source: str,
    run_id: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    run_link: Optional[str] = None,
    description: Optional[str] = None,
    await_creation_for: int = plugin_config.AWAIT_REGISTRATION_FOR,
):
    """
    Create a model version for a registered model in the Mlflow Model Registry.

    This method registers a new version of an existing registered model with the given
    source path or URI.
    Optionally, additional metadata such as run ID, tags, run link, and description can be provided.
    The `await_creation_for` parameter allows specifying a timeout for waiting for the model version
    creation to complete.

    Args:
        model (str): The name of the registered model.
        source (str): The source path or URI of the model. This is the location where the model
        artifacts are stored.
        run_id (Optional[str], optional): The ID of the run associated with this model version.
            This can be useful for tracking the lineage of the model version.
        tags (Optional[Dict[str, Any]], optional): A dictionary of key-value pairs to tag the
        model version with. Tags can help in organizing and filtering model versions.
        run_link (Optional[str], optional): A URI link to the run. This can provide quick access to
        the run details.
        description (Optional[str], optional): A description of the model version. This can provide
        additional context
            about the changes or improvements in this version.
        await_creation_for (int, optional): The time in seconds to wait for the model version
        creation to complete.
            Defaults to 300 seconds.

    Returns:
        ModelVersion: An instance of `ModelVersion` representing the created model version.
    """
    PluginManager().load_config()
    return MlflowPlugin().create_model_version(
        model=model,
        source=source,
        run_id=run_id,
        tags=tags,
        run_link=run_link,
        description=description,
        await_creation_for=await_creation_for,
    )


def set_tracking_uri(tracking_uri):
    """
    Sets the tracking URI.

    Args:
        tracking_uri (str): The tracking URI to set.
    """

    return MlflowPlugin().set_tracking_uri(tracking_uri=tracking_uri)


def set_experiment(
    experiment_name: Optional[str] = None, experiment_id: Optional[str] = None
):
    """
    Set the active experiment.

    This method sets the specified experiment as the active experiment.
    The active experiment is the one to which subsequent runs will be logged.
    You can specify the experiment by name or by ID.

    Args:
        experiment_name (Optional[str]): The name of the experiment to set as active.
            If `experiment_name` is provided, it takes precedence over `experiment_id`.
        experiment_id (Optional[str]): The ID of the experiment to set as active.
            If `experiment_name` is not provided, `experiment_id` will be used to set
            the active experiment.

    Returns:
        None
    """
    PluginManager().load_config()
    return MlflowPlugin().set_experiment(
        experiment_name=experiment_name, experiment_id=experiment_id
    )


def get_artifact_uri(artifact_path: Optional[str] = None):
    """
    Get the artifact URI of the current or specified  run.

    This method returns the URI of the artifact directory for the current run or for
    the specified artifact path.

    Args:
        artifact_path (Optional[str]): The path of the artifact within the run's artifact directory.
            If not provided, the method returns the URI of the current run's artifact directory.

    Returns:
        str: The URI of the specified artifact path or the current run's artifact directory.
    """
    PluginManager().load_config()

    return MlflowPlugin().get_artifact_uri(artifact_path=artifact_path)


def start_run(
    run_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
):
    """
    Starts a run.

    This method starts a new run or resumes an existing run if a run_id is provided.

    Args:
        run_id (Optional[str]): The ID of the run to resume. If not provided, a new run is started.
        experiment_id (Optional[str]): The ID of the experiment under which to create the run.
        run_name (Optional[str]): The name of the Mlflow run.
        nested (bool): Whether to create the run as a nested run of the parent run.
        tags (Optional[Dict[str, Any]]): A dictionary of tags to set on the run.
        description (Optional[str]): A description for the run.

    Returns:
        The Run object corresponding to the started or resumed run.
    """
    PluginManager().load_config()
    return MlflowPlugin().start_run(
        run_id=run_id,
        experiment_id=experiment_id,
        run_name=run_name,
        nested=nested,
        tags=tags,
        description=description,
    )


def end_run():
    """
    Ends the current run.

    Returns:
        str: The ID of the ended run.
    """
    PluginManager().load_config()
    return MlflowPlugin().end_run()


def log_param(key: str, value: Any):
    """
    Logs a parameter to the current run.

    Args:
        key (str): The key of the parameter.
        value (Any): The value of the parameter.
    """
    PluginManager().load_config()
    return MlflowPlugin().log_param(key=key, value=value)


def log_params(params: Dict[str, Any]) -> None:
    """
    Log a batch of params for the current run. If no run is active, this method will create a
    new active run.

    :param params: Dictionary of param_name: String -> value: (String, but will be string-ified if
                   not)
    :returns: None

    . test-code-block:: python
        :caption: Example

        import cogflow

        params = {"learning_rate": 0.01, "n_estimators": 10}

        # Log a batch of parameters
        with cogflow.start_run():
            cogflow.log_params(params)
    """
    PluginManager().load_config()
    return MlflowPlugin().log_params(params=params)


def log_metric(
    key: str,
    value: float,
    step: Optional[int] = None,
):
    """
    Logs a metric to the current run.

    Args:
        key (str): The key of the metric.
        value (float): The value of the metric.
        step (int, optional): The step at which the metric was logged.
    """
    return MlflowPlugin().log_metric(
        key=key,
        value=value,
        step=step,
    )


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log multiple metrics for the current run. If no run is active, this method will create a new
    active run.

    :param metrics: Dictionary of metric_name: String -> value: Float. Note that some special
                    values such as +/- Infinity may be replaced by other values depending on
                    the store. For example, sql based store may replace +/- Infinity with
                    max / min float values.
    :param step: A single integer step at which to log the specified
                 Metrics. If unspecified, each metric is logged at step zero.

    :returns: None

    . test-code-block:: python
        :caption: Example

        import cogflow

        metrics = {"mse": 2500.00, "rmse": 50.00}

        # Log a batch of metrics
        with cogflow.start_run():
            cogflow.log_metrics(metrics)
    """
    return MlflowPlugin().log_metrics(metrics=metrics, step=step)


def log_model(
    model,
    artifact_path,
    model_type: str = None,
    registered_model_name=None,
    conda_env=None,
    code_paths=None,
    serialization_format=plugin_config.SERIALIZATION_FORMAT,
    signature: ModelSignature = None,
    input_example: Union[
        pd.DataFrame,
        np.ndarray,
        dict,
        list,
        csr_matrix,
        csc_matrix,
        str,
        bytes,
        tuple,
    ] = None,
    await_registration_for=plugin_config.AWAIT_REGISTRATION_FOR,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn=plugin_config.PYFUNC_PREDICT_FN,
    metadata=None,
):
    """
    Logs a model.

    Args:
        model: The model to log.
        artifact_path (str): The artifact path to log the model to.
        model_type (str, optional): The type of the model.
        registered_model_name (str, optional): The name to register the model under.
        conda_env (str, optional): The conda environment to use.
        code_paths (list, optional): List of paths to include in the model.
        serialization_format (str, optional): The format to use for serialization.
        signature (ModelSignature, optional): The signature of the model.
        input_example (Union[pd.DataFrame, np.ndarray, dict, list, csr_matrix, csc_matrix, str,
         bytes, tuple], optional): Example input.
        await_registration_for (int, optional): Time to wait for registration.
        pip_requirements (list, optional): List of pip requirements.
        extra_pip_requirements (list, optional): List of extra pip requirements.
        pyfunc_predict_fn (str, optional): The prediction function to use.
        metadata (dict, optional): Metadata for the model.
    """
    PluginManager().load_config()
    # --- Safe detection logic (no torch / sklearn import required) ---
    cls_hierarchy = [cls.__name__.lower() for cls in type(model).mro()]

    is_pyfunc = isinstance(model, pyfunc.PythonModel) or (
        inspect.isclass(model) and issubclass(model, pyfunc.PythonModel)
    )

    # Detect PyTorch models, including scvi / lightning / wrappers
    is_pytorch = (
        "module" in cls_hierarchy
        or hasattr(model, "module")
        or any("torch" in str(base.__module__).lower() for base in type(model).mro())
    )

    is_sklearn = "baseestimator" in cls_hierarchy

    # ----------------------------------------------------------
    # Apply optional user override (model_type)
    # ----------------------------------------------------------
    if model_type:
        model_type = str(model_type).lower().strip()
        if model_type in ("pytorch", "torch", "scvi", "lightning"):
            is_pyfunc, is_pytorch, is_sklearn = False, True, False
        elif model_type in ("sklearn", "xgboost", "rf", "tree"):
            is_pyfunc, is_pytorch, is_sklearn = False, False, True
        elif model_type in ("pyfunc", "python_function", "custom"):
            is_pyfunc, is_pytorch, is_sklearn = True, False, False

    if is_pyfunc:
        # Log using pyfunc flavor
        result = custom_log_model(
            artifact_path=artifact_path,
            python_model=model,
            code_path=code_paths,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            metadata=metadata,
        )
    elif is_pytorch:
        # Log using PyTorchPlugin
        result = pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            code_paths=code_paths,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            metadata=metadata,
        )
    elif is_sklearn:
        # Log using MLflowPlugin (e.g., sklearn, XGBoost, etc.)
        result = MlflowPlugin().log_model(
            sk_model=model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            code_paths=code_paths,
            serialization_format=serialization_format,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            await_registration_for=await_registration_for,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            pyfunc_predict_fn=pyfunc_predict_fn,
            metadata=metadata,
        )
    else:
        raise ValueError("Unsupported model type for logging")

    try:
        active_run = mlflow.active_run()
        if not active_run:
            raise RuntimeError("No active MLflow run found")
        model_id = active_run.info.run_id

        model_details = MlflowPlugin().get_full_model_uri_from_run_or_registry(
            model_id=model_id,
        )
        model_type = MlflowPlugin().detect_model_type(model_details["model_uri"])

        model_dict = {
            "model_id": uuid_to_canonical(model_id),
            "model_name": str(
                model_details.get("model_name")
                or active_run.data.tags.get("mlflow.runName")
            ),
            "model_version": int(model_details.get("model_version") or 0),
            "register_date": datetime.fromtimestamp(
                active_run.info.start_time / 1000
            ).isoformat(),
            "type": model_type,
            "description": str(
                active_run.data.tags.get("mlflow.note.content") or "log_model"
            ),
            "user_id": KubeflowPlugin().get_current_user_from_namespace(),
        }

        path = PluginManager().load_path(path_name="log_model")
        url = f"{os.getenv(API_BASEPATH)}{path}"

        headers = {
            "kubeflow-userid": KubeflowPlugin().get_current_user_from_namespace()
        }

        make_post_request(url=url, data=model_dict, headers=headers)
    except Exception as exp:
        print(f"Failed to log model details to DB: {exp}")

    return result


def log_model_with_dataset(
    model,
    artifact_path,
    dataset: DatasetMetadata,
    conda_env=None,
    code_paths=None,
    serialization_format=plugin_config.SERIALIZATION_FORMAT,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: Union[
        pd.DataFrame,
        np.ndarray,
        dict,
        list,
        csr_matrix,
        csc_matrix,
        str,
        bytes,
        tuple,
    ] = None,
    await_registration_for=plugin_config.AWAIT_REGISTRATION_FOR,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn=plugin_config.PYFUNC_PREDICT_FN,
    metadata=None,
):
    """
    Logs a model along with its dataset.

    Args:
        model: The scikit-learn model to log.
        artifact_path (str): The artifact path to log the model to.
        dataset (DatasetMetadata): The dataset metadata.
        conda_env (str, optional): The conda environment to use.
        code_paths (list, optional): List of paths to include in the model.
        serialization_format (str, optional): The format to use for serialization.
        registered_model_name (str, optional): The name to register the model under.
        signature (ModelSignature, optional): The signature of the model.
        input_example (Union[pd.DataFrame, np.ndarray, dict, list, csr_matrix, csc_matrix, str,
         bytes, tuple], optional): Example input.
        await_registration_for (int, optional): Time to wait for registration.
        pip_requirements (list, optional): List of pip requirements.
        extra_pip_requirements (list, optional): List of extra pip requirements.
        pyfunc_predict_fn (str, optional): The prediction function to use.
        metadata (dict, optional): Metadata for the model.
    """
    PluginManager().load_config()
    return DatasetPlugin().log_model_with_dataset(
        sk_model=model,
        artifact_path=artifact_path,
        dataset=dataset,
        conda_env=conda_env,
        code_paths=code_paths,
        serialization_format=serialization_format,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        pyfunc_predict_fn=pyfunc_predict_fn,
        metadata=metadata,
    )


def link_model_to_dataset(dataset_id, model_id):
    """
    Links a model to a dataset.

    Args:
        dataset_id (str): The ID of the dataset.
        model_id (str): The ID of the model.
    """
    PluginManager().load_config()
    return NotebookPlugin().link_model_to_dataset(
        dataset_id=dataset_id, model_id=uuid_to_canonical(model_id)
    )


def save_model_uri_to_db(model_id, model_uri):
    """
    Save the model URI to the database.

    :param model_id: ID of the model to update.
    :param model_uri: URI of the model to save.
    :return: Response from the database save operation.
    """
    PluginManager().load_config()
    return NotebookPlugin().save_model_uri_to_db(
        model_id=uuid_to_canonical(model_id), model_uri=model_uri
    )


def save_model_details_to_db(registered_model_name):
    """
    Saves model details to the database.

    Args:
        registered_model_name (str): The name of the registered model.

    Returns:
        str: Information message confirming the model details are saved.
    """
    return NotebookPlugin().save_model_details_to_db(
        registered_model_name=registered_model_name
    )


def get_model_latest_version(registered_model_name):
    """
    Gets the latest version of a registered model.

    Args:
        registered_model_name (str): The name of the registered model.

    Returns:
        str: The latest version of the registered model.
    """
    PluginManager().load_config()
    return NotebookPlugin().get_model_latest_version(
        registered_model_name=registered_model_name
    )


def search_model_versions(
    filter_string: Optional[str] = None,
):
    """
    Searches for model versions in the model registry based on the specified filters.

    Args:
        filter_string (Optional[str], optional): A string specifying the conditions
        that the model versions must meet.
            It is used to filter the model versions. Examples of filter strings
            include "name='my-model'" or "name='my-model' and version='1'".
            If not provided, all model versions are returned.
            Defaults to None.

    Returns:
        List[dict]: A list of dictionaries, each representing a model version that meets
        the filter criteria. Each dictionary contains information about the model version,
        including its name, version number, creation time, run ID, and other metadata.
    """
    PluginManager().load_config()
    return MlflowPlugin().search_model_versions(filter_string=filter_string)


def pipeline(name=None, description=None):
    """
    Creates a new Kubeflow pipeline.

    Args:
        name (str, optional): The name of the pipeline.
        description (str, optional): The description of the pipeline.

    Returns:
        str: Information message confirming the pipeline creation.
    """
    return KubeflowPlugin().pipeline(name=name, description=description)


def create_component_from_func(
    func,
    output_component_file=None,
    base_image=plugin_config.BASE_IMAGE,
    packages_to_install=None,
    annotations: Optional[Mapping[str, str]] = None,
):
    """
    Creates a Kubeflow component from a function.

    Args:
        func: The function to create the component from.
        output_component_file (str, optional): The output file for the component.
        base_image (str, optional): The base image to use. Defaults to
        "hiroregistry/cogflow:dev".
        packages_to_install (list, optional): List of packages to install.
        annotations: Optional. Allows adding arbitrary key-value data to the
        component specification.

    Returns:
        str: Information message confirming the component creation.
    """
    return KubeflowPlugin().create_component_from_func(
        func=func,
        output_component_file=output_component_file,
        base_image=base_image,
        packages_to_install=packages_to_install,
        annotations=annotations,
    )


def client(
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
) -> kfp.Client:
    """
    Gets the Kubeflow client.

    Returns:
        KubeflowClient: The Kubeflow client object.
    """
    return KubeflowPlugin().client(
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def serve_model_v2(model_uri: str, isvc_name: str = None):
    """
    Serves a model using Kubeflow V2.

    Args:
        model_uri (str): The URI of the model to serve.
        isvc_name (str, optional): The name of the model to serve.

    Returns:
        str: Information message confirming the model serving.
    """
    return KubeflowPlugin().serve_model_v2(model_uri=model_uri, isvc_name=isvc_name)


def serve_model_v1(model_uri: str, isvc_name: str = None):
    """
    Serves a model using Kubeflow V1.

    Args:
        model_uri (str): The URI of the model to serve.
        isvc_name (str, optional): The name of the model to serve.

    Returns:
        str: Information message confirming the model serving.
    """
    return KubeflowPlugin().serve_model_v1(model_uri=model_uri, isvc_name=isvc_name)


def load_component(file_path=None, url=None, text=None, id: UUID = None):
    """Loads component from text, file or URL and creates a task factory
    function.

    Only one argument should be specified.

    Args:
        file_path: Path of local file containing the component definition.
        url: The URL of the component file data.
        text: A string containing the component file data.
        id: The ID of the component.

    Returns:
        A factory function with a strongly-typed signature.
        Once called with the required arguments, the factory constructs a
        pipeline task instance (ContainerOp).
    """
    PluginManager().load_config()
    # --- Sanity check ---
    non_null_args_count = len([v for v in [file_path, url, text, id] if v is not None])
    if non_null_args_count != 1:
        raise ValueError("Need to specify exactly one source")

    # --- Load base component factory ---
    if file_path:
        base_comp = KubeflowPlugin().load_component_from_file(file_path=file_path)
    elif url:
        base_comp = KubeflowPlugin().load_component_from_url(url=url)
    elif text:
        base_comp = KubeflowPlugin().load_component_from_text(text=text)
    elif id:
        base_comp = ComponentPlugin().load_component_from_id(component_id=id)
    else:
        raise ValueError("Need to specify a source")

    # --- Wrap with runtime env injection ---
    def wrapped_component(*args, **kwargs):
        component_op = base_comp(*args, **kwargs)
        component_op = CogContainer.add_model_access(component_op)
        return component_op

    # Preserve metadata for KFP compatibility
    wrapped_component.__signature__ = inspect.signature(base_comp)
    wrapped_component.component_spec = getattr(base_comp, "component_spec", None)
    wrapped_component.__name__ = getattr(base_comp, "__name__", "wrapped_component")

    return wrapped_component


def delete_pipeline(
    pipeline_id,
    api_url=None,
    skip_tls_verify=True,
    session_cookies=None,
    namespace=None,
):
    """
    method deletes the pipeline
    :param pipeline_id: pipeline id
    :param api_url: kfp api url
    :param skip_tls_verify: skip tls verify or not
    :param session_cookies: session cookies for kfp api
    :param namespace: user namespace
    :return:
    """
    # list the runs based on pipeline_id
    run_info = NotebookPlugin.list_runs_by_pipeline_id(pipeline_id)
    run_ids = [run["uuid"] for run in run_info["data"]]

    # delete the runs from kfp and db based on pipeline_id
    try:
        KubeflowPlugin().delete_runs(
            run_ids=run_ids,
            api_url=api_url,
            skip_tls_verify=skip_tls_verify,
            session_cookies=session_cookies,
            namespace=namespace,
        )
        NotebookPlugin.delete_run_details_from_db(pipeline_id)
    except ApiException as exp:
        print(f"Failed to delete run for the pipeline id {pipeline_id}: {exp}")

    # list the pipeline versions and delete from kfp
    pipeline_version_response = KubeflowPlugin().list_pipeline_versions(
        pipeline_id=pipeline_id,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )
    if pipeline_version_response.versions:
        pipeline_version_details = pipeline_version_response.versions

        pipeline_version_ids = [version.id for version in pipeline_version_details]
        print("Pipeline Version IDs to delete:", pipeline_version_ids)

        # Delete each pipeline version
        for version_id in pipeline_version_ids:
            try:
                KubeflowPlugin().delete_pipeline_version(
                    version_id=version_id,
                    api_url=api_url,
                    skip_tls_verify=skip_tls_verify,
                    session_cookies=session_cookies,
                    namespace=namespace,
                )
                print(f"Deleted pipeline version: {version_id}")
            except ApiException as exp:
                print(f"Failed to delete pipeline version {version_id}: {exp}")
    else:
        print(
            f"No pipeline versions found for the specified pipeline ID {pipeline_id}."
        )

    # Delete the pipeline itself
    try:
        KubeflowPlugin().delete_pipeline(
            pipeline_id=pipeline_id,
            api_url=api_url,
            skip_tls_verify=skip_tls_verify,
            session_cookies=session_cookies,
            namespace=namespace,
        )
        print(f"Deleted pipeline: {pipeline_id}")
    except ApiException as exp:
        print(f"Failed to delete pipeline {pipeline_id}: {exp}")

    NotebookPlugin.delete_pipeline_details_from_db(pipeline_id)


def cogcomponent(
    output_component_file: Optional[str] = None,
    base_image: str = plugin_config.BASE_IMAGE,
    packages_to_install: Optional[List[str]] = None,
    annotations: Optional[Mapping[str, str]] = None,
    name: Optional[str] = None,
    category: Optional[str] = None,
    register: bool = False,
    overwrite: bool = False,
):
    """
    Decorator to create a Kubeflow component and optionally register it.

    Args:
        output_component_file (str, optional): Path to save component YAML locally.
        base_image (str): Base Docker image for the component.
        packages_to_install (List[str], optional): Extra Python packages.
        annotations (Mapping[str, str], optional): Custom metadata.
        name (str, optional): Name for registration.
        category (str, optional): Category for registration.
        register (bool): Register the component automatically.
        overwrite (bool): Overwrite existing MinIO object if true.
    """

    def decorator(func: Callable):
        # Step 1: Build the component spec
        component_op = create_component_from_func(
            func=func,
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

        # Step 2: Optional registration
        if register and category:
            try:
                # ✅ Directly get YAML string from the ComponentSpec object
                yaml_data = yaml.safe_dump(
                    component_op.component_spec.to_dict(),
                    sort_keys=False,
                )

                print(
                    f"Registering component '{component_op.component_spec.name}' "
                    f"under category '{category}'..."
                )

                register_component(
                    name=name,
                    yaml_data=yaml_data,
                    category=category,
                    overwrite=overwrite,
                )

            except Exception as e:
                print(f"Component registration failed: {e}")

        return component_op

    return decorator


def create_run_from_pipeline_func(
    pipeline_func,
    arguments: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    namespace: Optional[str] = None,
    pipeline_root: Optional[str] = None,
    enable_caching: Optional[bool] = None,
    service_account: Optional[str] = None,
):
    """
        method to create a run from pipeline function
    :param pipeline_func:
    :param arguments:
    :param run_name:
    :param experiment_name:
    :param namespace:
    :param pipeline_root:
    :param enable_caching:
    :param service_account:
    :return:
    """
    run_details = KubeflowPlugin().create_run_from_pipeline_func(
        pipeline_func=pipeline_func,
        arguments=arguments,
        run_name=run_name,
        experiment_name=experiment_name,
        namespace=namespace,
        pipeline_root=pipeline_root,
        enable_caching=enable_caching,
        service_account=service_account,
    )
    # Poll the run status
    while not KubeflowPlugin().is_run_finished(
        run_details.run_id,
    ):
        status = KubeflowPlugin().get_run_status(
            run_details.run_id,
        )
        print(f"Run {run_details.run_id} status: {status}")
        time.sleep(plugin_config.TIMER_IN_SEC)

    # details = get_pipeline_and_experiment_details(run_details.run_id)
    # print("details of upload pipeline", details)
    # NotebookPlugin().save_pipeline_details_to_db(details)
    return run_details


def get_pipeline_and_experiment_details(
    run_id, api_url=None, skip_tls_verify=True, session_cookies=None, namespace=None
):
    """
        method to return pipeline,run_details,task_details,experiments details based on run_id
    :param run_id: run_id of the run
    :param api_url: kfp api url
    :param skip_tls_verify: skip tls verify or not
    :param session_cookies: session cookies for kfp api
    :param namespace: user namespace
    :return: dictionary with all the details of pipeline,run_details,task_details,experiments
    """
    try:
        # Get the run details using the run_id
        run_detail = (
            KubeflowPlugin()
            .client(
                api_url=api_url,
                skip_tls_verify=skip_tls_verify,
                session_cookies=session_cookies,
                namespace=namespace,
            )
            .get_run(run_id=run_id)
        )
        # Extract run details
        run = run_detail.run
        pipeline_id = run.pipeline_spec.pipeline_id
        workflow_manifest = run_detail.pipeline_runtime.workflow_manifest

        # Try to parse the workflow manifest if it's a valid JSON string
        try:
            # Parse the string to a dictionary (if it's a valid JSON string)
            workflow_manifest_dict = json.loads(workflow_manifest)

            # Get the workflow name from the metadata section
            workflow_name = workflow_manifest_dict.get("metadata", {}).get("name", None)

            # If pipeline_id is None, set it to the workflow_name
            if pipeline_id is None:
                pipeline_id = workflow_name

            # print(f"Pipeline ID: {pipeline_id}")

        except json.JSONDecodeError:
            print("Error: workflow_manifest is not a valid JSON string")
        experiment_id = run.resource_references[0].key.id
        run_details = {
            "uuid": run.id,
            "display_name": run.name,
            "name": run.name,
            "description": run.description,
            "experiment_uuid": experiment_id,
            "pipeline_uuid": pipeline_id,
            "createdAt_in_sec": run.created_at,
            "scheduledAt_in_sec": run.scheduled_at,
            "finishedAt_in_sec": run.finished_at,
            "state": run.status,
        }

        # Get experiment details using the experiment_id
        experiment = (
            KubeflowPlugin()
            .client(
                api_url=api_url,
                skip_tls_verify=skip_tls_verify,
                session_cookies=session_cookies,
                namespace=namespace,
            )
            .get_experiment(experiment_id=experiment_id)
        )

        experiment_details = {
            "uuid": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "createdAt_in_sec": experiment.created_at,
        }

        if run.pipeline_spec.pipeline_id:
            # Get pipeline details using the pipeline_id
            pipeline_info = (
                KubeflowPlugin()
                .client(
                    api_url=api_url,
                    skip_tls_verify=skip_tls_verify,
                    session_cookies=session_cookies,
                    namespace=namespace,
                )
                .get_pipeline(pipeline_id=pipeline_id)
            )

            pipeline_details = {
                "uuid": pipeline_info.id,
                "createdAt_in_sec": pipeline_info.created_at,
                "name": pipeline_info.name,
                "description": pipeline_info.description,
                "experiment_uuid": experiment.id,
                "status": run.status,
            }

        pipeline_spec = json.loads(
            workflow_manifest_dict["metadata"].get(
                "pipelines.kubeflow.org/pipeline_spec", "{}"
            )
        )
        pipeline_details = {
            "uuid": pipeline_id,
            "createdAt_in_sec": workflow_manifest_dict["metadata"].get(
                "creationTimestamp", None
            ),
            "name": workflow_manifest_dict["metadata"].get("name", None),
            "description": pipeline_spec.get("description", "No description available"),
            "experiment_uuid": experiment.id,
            "status": run.status,
        }

        workflow_manifest = run_detail.pipeline_runtime.workflow_manifest
        workflow = json.loads(workflow_manifest)

        # Extract the task details
        tasks = workflow["status"]["nodes"]

        task_details = []
        for task_id, task_info in tasks.items():
            task_detail = {
                "uuid": task_id,
                "name": task_info.get("displayName"),
                "state": task_info.get("phase"),
                "runuuid": run.id,
                "startedtimestamp": task_info.get("startedAt"),
                "finishedtimestamp": task_info.get("finishedAt"),
                "createdtimestamp": task_info.get("createdAt"),
            }
            task_details.append(task_detail)

        steps = workflow["status"]["nodes"]
        model_uris = []

        for step_info in steps.items():
            # print(f"step={step_name}")
            if step_info["type"] == "Pod":
                outputs = step_info.get("outputs", {}).get("parameters", [])
                for output in outputs:
                    # print(f"Artifact: {output['name']}")
                    # print(f"URI: {output['value']}")
                    if is_valid_s3_uri(output["value"]):
                        model_uris.append(output["value"])
                    else:
                        print("Not valid model-uri")
        model_uris = list(set(model_uris))

        model_ids = []
        for model_uri in model_uris:
            PluginManager().load_config()
            # Define the URL
            url = os.getenv(plugin_config.API_BASEPATH) + PluginManager().load_path(
                "models_uri"
            )
            query_params = {"uri": model_uri}
            # Make the GET request
            response = requests.get(url, params=query_params, timeout=100)

            # Check if the request was successful
            if response.status_code == 200:
                # Print the response content
                # print('Response Content:')
                model_ids.append(response.json()["data"])
            else:
                print(f"Failed to retrieve data: {response.status_code}")

        return {
            "run_details": run_details,
            "experiment_details": experiment_details,
            "pipeline_details": pipeline_details,
            "task_details": task_details,
            "model_ids": model_ids,
        }
    except Exception as e:
        return e


def log_artifact(
    local_path: str, artifact_path: Optional[str] = None, run_id: Optional[str] = None
):
    """
    Log a local file as an artifact of a run.

    Behavior:
      - If `run_id` is provided → logs the artifact(s) to that specific run
        (works even if the run has already finished).
      - If `run_id` is not provided → logs to the currently active run.
        If no run is active, a new run will automatically be created.

    Args:
        local_path (str): Path to the local file to log.
        artifact_path (str, optional): Subdirectory within the run's
            ``artifact_uri`` where the artifact(s) should be stored.
            If None, the artifact(s) are logged to the root.
        run_id (str, optional): The ID of the run to log the artifact(s) to.
            If not provided, logs to the active run or creates a new one.

    Returns:
        str or None: The artifact URI, depending on backend implementation.

    Examples:
        # Case 1: Log to a specific run (using run_id)
        >>> log_artifact(
        ...     local_path="reports/metrics.txt",
        ...     artifact_path="reports",
        ...     run_id="<run_id>"
        ... )
        # → stores as s3://mlflow/0/<run_id>/artifacts/reports/metrics.txt

        # Case 2: Log to the currently active run (or auto-starts one)
        >>> with cogflow.start_run() as run:
        ...     log_artifact(local_path="plots/chart.png", artifact_path="images")
        # → stores as s3://mlflow/0/<active_run_id>/artifacts/images/chart.png
    """
    PluginManager().load_config()

    if run_id is not None:
        return cogclient.log_artifact(
            run_id=uuid_to_hex(run_id),
            local_path=local_path,
            artifact_path=artifact_path,
        )
    else:
        return MlflowPlugin().log_artifact(
            local_path=local_path, artifact_path=artifact_path
        )


def log_artifacts(
    local_dir: str, artifact_path: Optional[str] = None, run_id: Optional[str] = None
) -> None:
    """
    Log all the contents of a local directory as artifacts of the run. If no run is active,
    this method will create a new active run.

    :param local_dir: Path to the directory of files to write.
    :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.
    :param run_id: The ID of the run to log the artifact(s) to.
    If not provided, logs to the active run or creates a new one.
    :return: None

    . test-code-block:: python
        :caption: Example

        import json
        import os
        import cogflow

        # Create some files to preserve as artifacts
        features = "rooms, zipcode, median_price, school_rating, transport"
        data = {"state": "TX", "Available": 25, "Type": "Detached"}

        # Create a couple of artifact files under the directory "data"
        os.makedirs("data", exist_ok=True)
        with open("data/data.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        with open("data/features.txt", 'w') as f:
            f.write(features)

        # Write all files in "data" to root artifact_uri/states
        with cogflow.start_run():
            cogflow.log_artifacts("data", artifact_path="states")
    """
    PluginManager().load_config()
    if run_id is not None:
        return cogclient.log_artifacts(
            run_id=uuid_to_hex(run_id), local_dir=local_dir, artifact_path=artifact_path
        )
    else:
        return MlflowPlugin().log_artifacts(
            local_dir=local_dir, artifact_path=artifact_path
        )


original_pyfunc_log_model = pyfunc.log_model


def custom_log_model(
    artifact_path,
    registered_model_name=None,
    loader_module=None,
    data_path=None,
    code_path=None,
    conda_env=None,
    python_model=None,
    artifacts=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=plugin_config.AWAIT_REGISTRATION_FOR,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Custom wrapper around cogflow.pyfunc.log_model with extended signature.

    Args:
        artifact_path (str): The location where model artifacts should be saved.
        loader_module (str, optional): The module that defines how to load the model.
        data_path (str, optional): Path to the data used by the model.
        code_path (str or list, optional): Path(s) to custom code dependencies.
        conda_env (str or dict, optional): Conda environment specification.
        python_model (object, optional): Custom Python model class.
        artifacts (dict, optional): Additional artifacts to log.
        registered_model_name (str, optional): Name of the registered model.
        signature (ModelSignature, optional): Model signature (input/output schema).
        input_example (ModelInputExample, optional): Example input for the model.
        await_registration_for (int, optional): Time to wait for model registration.
        pip_requirements (list, optional): List of pip requirements.
        extra_pip_requirements (list, optional): Additional pip requirements.
        metadata (dict, optional): Additional metadata to log.
        **kwargs: Additional arguments for cogflow.pyfunc.log_model.
    """

    # Call the original cogflow.pyfunc.log_model
    result = original_pyfunc_log_model(
        artifact_path=artifact_path,
        loader_module=loader_module,
        data_path=data_path,
        code_path=code_path,
        conda_env=conda_env,
        python_model=python_model,
        artifacts=artifacts,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )

    return result


# Reassign the custom function to cogflow.pyfunc.log_model
pyfunc.log_model = custom_log_model


def get_served_models(
    isvc_name: str = None,
    namespace: str = None,
):
    """
    Gets information about inference service of served models

    Args:
        namespace (str): Namespace where isvc is deployed.
        isvc_name (str, optional): Name of served model. If None, returns all served models.

    Returns:
        list: List of model information dictionaries. Each dict contains:
              model_name, model_id, model_version, creation_timestamp,
              served_model_url, status, traffic_percentage.
    """
    return KubeflowPlugin().get_served_models(namespace=namespace, isvc_name=isvc_name)


def delete_served_model(isvc_name: str, namespace: str = None):
    """
    Deletes a served model.

    Args:
        isvc_name (str): The name of the model to delete.
        namespace (str, optional): The namespace where the model is served.

    Returns:
        str: Information message confirming the deletion of the served model.
    """
    return KubeflowPlugin().delete_served_model(
        isvc_name=isvc_name, namespace=namespace
    )


def serve_model_v2_url(model_uri: str, name: str = None):
    """
    Serves a model using Kubeflow V2.

    Args:
        model_uri (str): The URI of the model to serve.
        name (str, optional): The name of the model to serve.

    Returns:
        str: Information message confirming the model serving.
    """
    try:
        KubeflowPlugin().serve_model_v2(model_uri=model_uri, isvc_name=name)
        return get_served_models(isvc_name=name)
    except Exception as exp:
        return f"Failed to serve model: {exp}"


def serve_model_v1_url(model_uri: str, isvc_name: str = None):
    """
    Serves a model using Kubeflow V1.

    Args:
        model_uri (str): The URI of the model to serve.
        isvc_name (str, optional): The name of the model to serve.

    Returns:
        str: Information message confirming the model serving.
    """
    try:
        KubeflowPlugin().serve_model_v1(model_uri=model_uri, isvc_name=isvc_name)
        return get_served_models(isvc_name=isvc_name)
    except Exception as exp:
        return f"Failed to serve model: {exp}"


def log_model_by_model_file(model_file_path, model_name):
    """
        log_model in cogflow with the model_file
    :param model_file_path: file_path of model
    :param model_name: name of the model
    :return:
        data = {
            "artifact_uri" : 'artifact_uri of the model',
            "version" : "model version"
        }
    """

    return NotebookPlugin().log_model_by_model_file(
        model_file_path=model_file_path, model_name=model_name
    )


def deploy_model(model_name, model_version, isvc_name):
    """

    :param model_name: name of the model
    :param model_version: version of the model
    :param isvc_name: service name to be created for the deployed model
    :return:
    """
    return NotebookPlugin().deploy_model(
        model_name=model_name, model_version=model_version, isvc_name=isvc_name
    )


def list_pipelines_by_name(
    pipeline_name,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Lists all versions and runs of the specified pipeline by name.

    Args:
        pipeline_name (str): The name of the pipeline to fetch details for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The namespace to use for the Kubeflow Pipelines API.

    Returns:
        dict: A dictionary containing the pipeline ID, versions,
         and runs of the specified pipeline.

    Raises:
        ValueError: If the pipeline name is invalid or not found.
        Exception: For any other issues encountered during the fetch operations.
    """

    return NotebookPlugin().list_pipelines_by_name(
        pipeline_name=pipeline_name,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def model_recommender(model_name=None, classification_score=None):
    """
    Calls the model recommender API and returns the response.

    Args:
    - model_name (str): The name of the model to recommend (optional).
    - classification_score (list): A list of classification scores to consider(e.g., accuracy_score, f1_score,
     recall_score, log_loss, roc_auc, precision_score, example_count, score.). (optional).

    Returns:
    - dict: The response from the model recommender API.
    """

    return NotebookPlugin().model_recommender(
        model_name=model_name, classification_score=classification_score
    )


def get_pipeline_task_sequence_by_run_id(
    run_id,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches the pipeline workflow and task sequence for a given run in Kubeflow.

    Args:
        run_id (str): The ID of the pipeline run to fetch details for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The namespace to use for the Kubeflow Pipelines API.

    Returns:
        tuple: A tuple containing:
            - pipeline_workflow_name (str): The name of the pipeline's workflow (root node of the DAG).
            - task_structure (dict): A dictionary representing the task structure of the pipeline, with each node
                                     containing information such as task ID, pod name, status, inputs, outputs,
                                     and resource duration.

    The task structure contains the following fields for each node:
        - id (str): The unique ID of the task (node).
        - podName (str): The name of the pod associated with the task.
        - name (str): The display name of the task.
        - inputs (list): A list of input parameters for the task.
        - outputs (list): A list of outputs produced by the task.
        - status (str): The phase/status of the task (e.g., 'Succeeded', 'Failed').
        - startedAt (str): The timestamp when the task started.
        - finishedAt (str): The timestamp when the task finished.
        - resourcesDuration (dict): A dictionary representing the resources used (e.g., CPU, memory).
        - children (list): A list of child tasks (if any) in the DAG.

    Example:
        >>> run_id = "afcf98bb-a9af-4a34-a512-1236110150ae"
        >>> pipeline_name, task_structure = get_pipeline_and_task_sequence_by_run(run_id)
        >>> print(f"Pipeline Workflow Name: {pipeline_name}")
        >>> print("Task Structure:", task_structure)

    Raises:
        ValueError: If the root node (DAG) is not found in the pipeline.
    """
    return NotebookPlugin().get_pipeline_task_sequence_by_run_id(
        run_id=run_id,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def list_all_pipelines(
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Lists all pipelines along with their IDs, handling pagination.
    Args:
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        list: A list of tuples containing (pipeline_name, pipeline_id).
    """
    return NotebookPlugin().list_all_pipelines(
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_pipeline_task_sequence_by_pipeline_id(
    pipeline_id,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches the task structures of all pipeline runs based on the provided pipeline_id.

    Args:
        pipeline_id (str): The ID of the pipeline to fetch task structures for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        list: A list of dictionaries containing pipeline workflow names and task structures for each run.
    Example:
        >>>pipeline_id = "1000537e-b101-4432-a779-768ec479c2b0"  # Replace with your actual pipeline_id
        >>>all_task_structures = get_pipeline_task_sequence_by_pipeline_id(pipeline_id)
        >>>for details in all_task_structures:
            >>>print(f'Run ID: {details["run_id"]}')
            >>>print(f'Pipeline Workflow Name: {details["pipeline_workflow_name"]}')
            >>>print("Task Structure:")
            >>>print(json.dumps(details["task_structure"], indent=4))
    """
    return NotebookPlugin().get_pipeline_task_sequence_by_pipeline_id(
        pipeline_id=pipeline_id,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_latest_run_id_by_pipeline_id(
    pipeline_id,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches the run_id of the latest pipeline run by its pipeline_id.

    Args:
        pipeline_id (str): The ID of the pipeline to search for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        str: The run_id of the latest run if found, otherwise None.
    """
    return NotebookPlugin().get_run_ids_by_pipeline_id(
        pipeline_id=pipeline_id,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_pipeline_task_sequence_by_run_name(
    run_name,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches the task structure of a pipeline run based on its name.

    Args:
        run_name (str): The name of the pipeline run to fetch task structure for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        tuple: (pipeline_workflow_name, task_structure)
    Example:
        >>>run_name = "Run of test_pipeline (ad001)"
        >>>pipeline_name, task_structure = get_pipeline_task_sequence_by_name(run_name)
        >>>print(f'Pipeline Workflow Name: {pipeline_name}')
        >>>print("Task Structure:")
        >>>print(json.dumps(task_structure, indent=4))
    """
    return NotebookPlugin().get_pipeline_task_sequence_by_run_name(
        run_name=run_name,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_run_id_by_run_name(
    run_name,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches the run_id of a pipeline run by its name, traversing all pages if necessary.

    Args:
        run_name (str): The name of the pipeline run to search for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        str: The run_id if found, otherwise None.
    """
    return NotebookPlugin().get_run_id_by_run_name(
        run_name=run_name,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_run_ids_by_pipeline_name(
    pipeline_name,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches all run_ids for a given pipeline name.

    Args:
        pipeline_name (str): The name of the pipeline to search for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        list: A list of run_ids for the matching pipeline name.
    """
    return NotebookPlugin().get_run_ids_by_pipeline_name(
        pipeline_name=pipeline_name,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_pipeline_task_sequence(
    pipeline_name=None,
    pipeline_workflow_name=None,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches the task structures of all pipeline runs based on the provided pipeline name or pipeline workflow name.

    Args:
        pipeline_name (str, optional): The name of the pipeline to fetch task structures for.
        pipeline_workflow_name (str, optional): The workflow name of the pipeline to fetch task structures for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        list: A list with details of task structures for each run.
    Example:
        >>> pipeline_workflow_name = "pipeline-vzn7z"
        >>> all_task_structures = get_pipeline_task_sequence(pipeline_workflow_name=pipeline_workflow_name)
        >>> for details in all_task_structures:
                >>> print(f'Run ID: {details["run_id"]}')
                >>> print(f'Pipeline Workflow Name: {details["pipeline_workflow_name"]}')
                >>> print("Task Structure:")
                >>> print(json.dumps(details["task_structure"], indent=4))

    Raises:
        ValueError: If neither pipeline_name nor pipeline_workflow_name is provided.
    """
    return NotebookPlugin().get_pipeline_task_sequence(
        pipeline_name=pipeline_name,
        pipeline_workflow_name=pipeline_workflow_name,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_all_run_ids(
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches all run_ids available in the system.
    Args:
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        list: A list of all run_ids.
    """
    return NotebookPlugin().get_all_run_ids(
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_run_ids_by_name(
    run_name,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches run_ids by run name.

    Args:
        run_name (str): The name of the run to search for.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        list: A list of run_ids matching the run_name.
    """
    return NotebookPlugin().get_run_ids_by_name(
        run_name=run_name,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def get_task_structure_by_task_id(
    task_id,
    run_id=None,
    run_name=None,
    api_url: str = None,
    skip_tls_verify: bool = True,
    session_cookies: str = None,
    namespace: str = None,
):
    """
    Fetches the task structure of a specific task ID, optionally filtered by run_id or run_name.

    Args:
        task_id (str): The task ID to look for.
        run_id (str, optional): The specific run ID to filter by. Defaults to None.
        run_name (str, optional): The specific run name to filter by. Defaults to None.
        api_url (str, optional): The URL of the Kubeflow Pipelines API endpoint.
        skip_tls_verify (bool, optional): Whether to skip TLS verification for the API requests.
            Defaults to True.
        session_cookies (str, optional): Session cookies for authentication if required.
        namespace (str, optional): The user namespace to use for the Kubeflow Pipelines API.

    Returns:
        list: A list of dictionaries containing run IDs and their corresponding task info if found.
    Example:
        >>>task_id = "test-pipeline-749dn-2534915009"
        >>>run_id = None  # "afcf98bb-a9af-4a34-a512-1236110150ae"
        >>>run_name = "Run of test_pipeline (ad001)"
    """
    return NotebookPlugin().get_task_structure_by_task_id(
        task_id=task_id,
        run_id=run_id,
        run_name=run_name,
        api_url=api_url,
        skip_tls_verify=skip_tls_verify,
        session_cookies=session_cookies,
        namespace=namespace,
    )


def register_message_broker(
    dataset_name: str,
    broker_name: str,
    broker_ip: str,
    broker_port: int,
    topic_name: str,
):
    """
    Registers a Message Broker dataset by creating and submitting a registration request.

    This function constructs a `Request` object with details about the dataset,
    such as dataset name, Kafka host name, server IP, and topic details, then submits
    this request to register the dataset using the `KafkaDatasetPlugin`. If any error
    occurs during the process, it logs the exception message.

    Parameters:
    - dataset_name (str): The name of the dataset to be registered.
    - broker_name (str): Host name of the Broker server.
    - broker_ip (str): IP address of the Broker server.
    - broker_port (int): Port address of the Broker server.
    - topic_name (str): Name of the Broker topic associated with this dataset.

    Returns:
    - Response from the `BrokerDatasetPlugin` upon successful registration, or None if an error occurs.

    Exceptions:
    - Catches any exceptions and logs an error message detailing the failure.

    """
    try:
        print(f"Start creating dataset {dataset_name}")
        message_broker_dataset_plugin = MessageBrokerDatasetPlugin()
        message_broker_dataset_plugin.register_message_broker_dataset(
            dataset_name, broker_name, broker_ip, topic_name, broker_port
        )
    except Exception as ex:
        print(f"Error registering message broker server dataset details: {str(ex)}")


def read_message_broker_data(dataset_id: int):
    """
    Initiates reading messages from a specified message broker topic.

    This function calls `start_consumer_thread` with the provided message broker URL,
    topic name, and consumer group ID to initiate the reading process. It starts a
    consumer thread to continuously listen for and process incoming messages on the
    specified topic, managed under the provided consumer group for load balancing
    and offset tracking.

    Parameters:
    - dataset_id (str): The name of the dataset that need to be read data from.

    Returns:
    - None
    """
    print(f"Reading message from dataset {dataset_id}")
    message_broker_dataset_plugin = MessageBrokerDatasetPlugin()
    message_broker_topic_detail = (
        message_broker_dataset_plugin.get_message_broker_details(dataset_id)
    )
    print(f"start reading message from topic {message_broker_topic_detail}")
    kafka_broker_url = (
        message_broker_topic_detail.broker_ip
        + ":"
        + str(message_broker_topic_detail.broker_port)
    )
    read_from_kafka_topic(
        kafka_broker_url,
        message_broker_topic_detail.topic_name,
        "aces_metrics_consumer",
    )


def read_from_kafka_topic(kafka_broker_url, topic_name, group_id):
    """
    Initiates reading messages from a specified Kafka topic.

    This function calls `start_consumer_thread` with the provided Kafka broker URL,
    topic name, and consumer group ID to initiate the reading process. It starts a
    consumer thread to continuously listen for and process incoming messages on the
    specified topic, managed under the provided consumer group for load balancing
    and offset tracking.

    Parameters:
    - kafka_broker_url (str): The URL of the Kafka broker to connect to.
    - topic_name (str): The name of the Kafka topic to read messages from.
    - group_id (str): The consumer group ID for managing offsets and load balancing.

    Returns:
    - None
    """

    start_consumer_thread(kafka_broker_url, topic_name, group_id)


def stop_kafka_consumer():
    """
    Stops the Kafka consumer gracefully.

    This function initiates the process to stop the Kafka consumer by printing a log
    message and calling the `stop_consumer` function. This is useful in scenarios where
    the consumer needs to be terminated without abruptly closing the connection,
    ensuring any active resources are released appropriately.

    Behavior:
    - Logs a message to indicate that a stop request for the Kafka consumer has been received.
    - Calls `stop_consumer()` to perform the stop operation.

    Returns:
    - None
    """
    print("Stop kafka consumer request received....")
    stop_consumer()


def get_model_uri(model_name, version):
    """
        return the model_uri given the model name and version
    :param model_name: name of the model
    :param version: version of the model
    :return: model_uri
    """
    return MlflowPlugin().get_model_uri(model_name=model_name, version=version)


def get_artifacts(model_name, version):
    """
        return the model_uri given the model name and version
    :param model_name: name of the model
    :param version: version of the model
    :return: model_uri
    """
    artifacts_complete = MlflowPlugin().get_model_uri(
        model_name=model_name, version=version
    )
    artifacts = "/".join(artifacts_complete.split("/")[:-1])

    return artifacts


def get_deployments(namespace=KubeflowPlugin().get_default_namespace()):
    """
    Fetches details of all InferenceServices in the given namespace and formats them.

    Args:
    - namespace (str): The Kubernetes namespace where InferenceServices are deployed. Defaults to "default".

    Returns:
    - list of dicts: A list of dictionaries with InferenceService details.
    """
    return NotebookPlugin().get_deployments(namespace=namespace)


def create_fl_component_from_func(
    func,
    output_component_file=None,
    base_image=plugin_config.FL_COGFLOW_BASE_IMAGE,
    packages_to_install=None,
    annotations: Optional[Mapping[str, str]] = None,
    container_port=plugin_config.CONTAINER_PORT,
):
    """
    Create a component from a Python function with additional configurations
    for ports and pod labels.
    Args:
        func (Callable): Python function to convert into a component.
        output_component_file (str, optional): Path to save the component YAML file. Defaults to None.
        base_image (str, optional): Base Docker image for the component. Defaults to None.
        packages_to_install (List[str], optional): List of additional Python packages to install.
        annotations: Optional. Adds arbitrary key-value data to the component specification.
        container_port (int, optional): Container port to expose. Defaults to 8080.
    Returns:
        kfp.components.ComponentSpec: Component specification.
    """
    return KubeflowPlugin().create_fl_component_from_func(
        func=func,
        output_component_file=output_component_file,
        base_image=base_image,
        packages_to_install=packages_to_install,
        annotations=annotations,
        container_port=container_port,
    )


def fl_server_component(
    output_component_file=None,
    base_image=plugin_config.FL_COGFLOW_BASE_IMAGE,
    packages_to_install=None,
    annotations: Optional[Mapping[str, str]] = None,
    container_port=plugin_config.CONTAINER_PORT,
):
    """
    Decorator to create a Kubeflow component from a Python function.

    Args:
        output_component_file (str, optional): Path to save the component YAML file. Defaults to None.
        base_image (str, optional): Base Docker image for the component. Defaults to None.
        packages_to_install (List[str], optional): List of additional Python packages to install.
        annotations: Optional. Adds arbitrary key-value data to the component specification.
        container_port (int, optional): Container port to expose. Defaults to 8080.
    Returns:
        Callable: A wrapped function that is now a Kubeflow component.
    """

    def decorator(func):
        return create_fl_component_from_func(
            func=func,
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
            container_port=container_port,
        )

    return decorator


def create_fl_pipeline(
    fl_client: Callable,
    fl_server: Callable,
    connectors: list,
    node_enforce: bool = True,
):
    """
    Returns a KFP pipeline function that wires up:
    setup_links → fl_server → many fl_client → release_links

    fl_client must accept at minimum:
    - server_address: str
    - local_data_connector

    fl_server must accept at minimum:
    - number_of_iterations: int

    Any other parameters that fl_client/ fl_server declare will automatically
    become pipeline inputs and be forwarded along.
    """
    return KubeflowPlugin().create_fl_pipeline(
        fl_client=fl_client,
        fl_server=fl_server,
        connectors=connectors,
        node_enforce=node_enforce,
    )


def extract_data_products_from_json(data_input: Any) -> List[Dict[str, str]]:
    """
    Extracts data products (region and access URLs) from either:
    - A JSON string, or
    - A Python list/dict (already parsed JSON)

    Args:
        data_input (str | list | dict): JSON string or already parsed JSON.

    Returns:
        List[Dict[str, str]]: A list of {region, access_url} dictionaries.
    """
    # ✅ Step 1: Parse only if it's a string
    if isinstance(data_input, str):
        try:
            data = json.loads(data_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    else:
        data = data_input  # Already parsed

    # ✅ Step 2: Ensure we have a list of datasets
    if isinstance(data, dict):
        data = [data]

    data_products = []

    # ✅ Step 3: Extract connectors
    for ds in data:
        region = ds.get("region")
        for dist in ds.get("distribution", []):
            access_url = dist.get("accessURL")
            if region and access_url:
                data_products.append({"region": region, "access_url": access_url})

    return data_products


def create_fl_pipeline_dataspace(
    fl_client: Union[Callable, object],
    fl_server: Union[Callable, object],
    data_products: list,
    node_enforce: bool = False,
):
    """
    Returns a KFP pipeline function that wires up:
    setup_links → fl_server → many fl_client → release_links

    Args:
        fl_client: Callable or KFP component factory function.
        fl_server: Callable or KFP component factory function.
        data_products (list): List of dicts:
            [{"region": "...", "access_url": "..."}, ...]
        node_enforce (bool): Whether to enforce region-based node scheduling.

    Any other parameters that fl_client/ fl_server declare will automatically
    become pipeline inputs and be forwarded along.
    """

    # --- Helper to normalize to component ops ---
    def _to_component(obj, label: str):

        # Case 1: already a Kubeflow component (from load_component_from_id)
        if hasattr(obj, "component_spec"):
            print(f"Using preloaded {label} component.")
            return obj

        # Case 2: function — wrap it into a KFP component
        if isinstance(obj, FunctionType):
            print(f"Wrapping {label} function into Kubeflow component...")
            return create_component_from_func(
                obj,
                base_image="hiroregistry/cogflow_lite:latest",
            )

        # Case 3: unsupported
        raise TypeError(
            f"Invalid {label} type: {type(obj)} — must be a function or Kubeflow component."
        )

    # --- Normalize both ---
    fl_client_comp = _to_component(fl_client, "fl_client")
    fl_server_comp = _to_component(fl_server, "fl_server")

    # # --- Validate data products ---
    # if not isinstance(data_products, list) or not all(
    #     isinstance(dp, dict) and "region" in dp and "access_url" in dp
    #     for dp in data_products
    # ):
    #     raise ValueError(
    #         "data_products must be a list of dicts with keys 'region' and 'access_url'."
    #     )
    data_products = extract_data_products_from_json(data_products)

    return KubeflowPlugin().create_fl_pipeline_dataspace(
        fl_client=fl_client_comp,
        fl_server=fl_server_comp,
        data_products=data_products,
        node_enforce=node_enforce,
    )


def create_fl_client_component(
    func,
    output_component_file=None,
    base_image=plugin_config.FL_COGFLOW_BASE_IMAGE,
    packages_to_install=None,
    annotations: Optional[Mapping[str, str]] = None,
):
    """
    Create a component from a Python function with additional configurations.
    Args:
        func (Callable): Python function to convert into a component.
        output_component_file (str, optional): Path to save the component YAML file. Defaults to None.
        base_image (str, optional): Base Docker image for the component. Defaults to None.
        packages_to_install (List[str], optional): List of additional Python packages to install.
        annotations: Optional. Adds arbitrary key-value data to the component specification.
    Returns:
        kfp.components.ComponentSpec: Component specification.
    """
    return KubeflowPlugin().create_fl_client_component(
        func=func,
        output_component_file=output_component_file,
        base_image=base_image,
        packages_to_install=packages_to_install,
        annotations=annotations,
    )


def fl_client_component(
    output_component_file=None,
    base_image=plugin_config.FL_COGFLOW_BASE_IMAGE,
    packages_to_install=None,
    annotations: Optional[Mapping[str, str]] = None,
):
    """
    Creates a Kubeflow component from a function.

    Args:
        output_component_file (str, optional): The output file for the component.
        base_image (str, optional): The base image to use. Defaults to
        "hiroregistry/flcogflow:latest".
        packages_to_install (list, optional): List of packages to install.
        annotations: Optional. Allows adding arbitrary key-value data to the
        component specification.

    Returns:
        str: Information message confirming the component creation.
    """

    def decorator(func):
        return create_fl_client_component(
            func=func,
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

    return decorator


def get_current_user_from_namespace() -> str:
    """
    Fetch the current Kubeflow user ID by reading the owner annotation
    from the user's namespace.

    Returns:
        str: The user ID of the notebook owner.

    Raises:
        RuntimeError: If the owner annotation is not found.
    """
    return KubeflowPlugin().get_current_user_from_namespace()


def register_component(
    name: str = None,
    yaml_path: str = None,
    yaml_data: str = None,
    category: str = None,
    overwrite: bool = False,
):
    """
    Registers a component by uploading its YAML definition (from file or memory)
    to MinIO and posting its metadata to the registry API.

    Args:
        name (str, optional): Name of the component. If not provided, it will be
            extracted from the YAML content.
        yaml_path (str, optional): Local path to the component YAML file.
        yaml_data (str, optional): Raw YAML string of the component.
        category (str): Category / logical namespace.
        overwrite (bool, optional): Overwrite existing MinIO object if true.

    Returns:
        dict: {
            "registry_response": <dict>,
            "minio_url": "s3://{category}/{bucket}/{object_name}",
            "object_name": "{object_name}.yaml"
        }

    Raises:
        ValueError: If neither yaml_path nor yaml_data is provided.
        requests.HTTPError: If the registry API returns an error.
    """
    return ComponentPlugin().register_component(
        name=name,
        yaml_path=yaml_path,
        yaml_data=yaml_data,
        bucket_name=plugin_config.COMPONENTS_BUCKET_NAME,
        category=category,
        creator=get_current_user_from_namespace(),
        overwrite=overwrite,
    )


def get_full_model_uri_from_run_or_registry(
    model_id: str = None,
    artifact_path: str = None,
    model_name: str = None,
    model_version: str = None,
) -> str:
    """
    Returns the full model URI from either run_id or a model registry entry.

    Args:
        model_id (str, optional): The run ID or model_id.
        artifact_path (str, optional): Specific artifact_path name (e.g., 'model').
        model_name (str, optional): Name of the registered model.
        model_version (str, optional): Version of the registered model.

    Returns:
        str: Full model URI like s3://.../artifacts/artifact_path

    Raises:
        Exception: If neither valid run_id nor model registry info is provided or valid.
    """

    return MlflowPlugin().get_full_model_uri_from_run_or_registry(
        model_id=uuid_to_hex(model_id),
        artifact_path=artifact_path,
        model_name=model_name,
        model_version=model_version,
    )


def serve_model(
    model_id: str = None,
    isvc_name: str = None,
    artifact_path: str = None,
    model_name: str = None,
    model_version: str = None,
    dataset_id: str = None,
    transformer_image: str = plugin_config.TRANSFORMER_BASE_IMAGE,
    transformer_parameters: dict = None,
    protocol_version: str = None,
    model_format: str = None,
    namespace: str = None,
):
    """
    Resolve a model and create a KServe InferenceService.

    Args:
        model_id (str, optional): Unique identifier for the model/run.
        isvc_name (str, optional): Name of the KServe InferenceService.
            If not provided, one will be auto-generated.
        model_name (str, optional): Registered model name (alternative to model_id).
        model_version (str, optional): Registered model version.
        artifact_path (str, optional): Specific artifact path (e.g., "model").
        dataset_id (str, optional): Dataset linked to the model.
        transformer_image (str): Image of the transformer.
            Required if transformer_parameters is provided.
        transformer_parameters (dict, optional): Parameters for the transformer.
        protocol_version (str, optional): Protocol version for the model server (e.g., "v1", "v2").
        model_format (str, optional): Model format (e.g., "mlflow", "sklearn").
        namespace (str, optional): Kubernetes namespace to deploy the InferenceService.

    Examples:
        # Serve using run ID (with optional artifact path)
        >>> serve_model(model_id="abcd1234", isvc_name="my-model", artifact_path="model")

        # Serve using registered model name + version
        >>> serve_model(isvc_name="my-model", model_name="XGBoost", model_version="5")

        # Serve with dataset and transformer
        >>> serve_model(
        ...     isvc_name="my-model-tkg",
        ...     model_name="TKG",
        ...     model_version="3",
        ...     dataset_id="ds-123",
        ...     transformer_image="my-transformer:latest",
        ...     transformer_parameters={
        ...         "PROMETHEUS_URL": "http://prometheus:9090",
        ...         "PROMETHEUS_METRICS": "metric1,metric2"
        ...     },
        ...     protocol_version="v2",
        ...     model_format="mlflow"
        ... )

    Raises:
        ValueError: If no model_id or (model_name + model_version) is provided.
        Exception: For any errors during model resolution or deployment.
    """
    try:
        if not model_id and not (model_name and model_version):
            raise ValueError(
                "Must provide either model_id or (model_name and model_version)."
            )

        # Resolve model details
        model_details = MlflowPlugin().get_full_model_uri_from_run_or_registry(
            model_id=uuid_to_hex(model_id),
            artifact_path=artifact_path,
            model_name=model_name,
            model_version=model_version,
        )

        transformer_parameters = transformer_parameters or {}

        if dataset_id is not None and not transformer_parameters:
            dataset = get_dataset(
                dataset_id=dataset_id, endpoint=PluginManager().load_path("dataset")
            )
            if dataset.get("data_source_type") == 20:
                if not transformer_image:
                    raise ValueError(
                        "Dataset is of Prometheus type. You must provide a 'transformer_image' "
                        "to handle preprocessing for the transformer."
                    )
                dataset_response = get_dataset(
                    dataset_id=dataset_id,
                    endpoint=PluginManager().load_path("prometheus_dataset"),
                )
                transformer_parameters = {
                    "PROMETHEUS_URL": dataset_response.get("connection_type", {}).get(
                        "prometheus_url"
                    ),
                    "PROMETHEUS_METRICS": dataset_response.get("metric_list", {}).get(
                        "METRIC_FEATURES"
                    ),
                }

        # Set model_format if not provided
        if model_format is None:
            model_format = MlflowPlugin().detect_model_format(
                model_details["model_uri"]
            )

        # Serve via KubeflowPlugin
        KubeflowPlugin().serve_model(
            model_uri=model_details["model_uri"],
            isvc_name=isvc_name,
            model_id=uuid_to_canonical(model_details["model_id"]),
            model_name=model_details["model_name"],
            model_version=model_details["model_version"],
            dataset_id=dataset_id,
            transformer_image=transformer_image,
            transformer_parameters=transformer_parameters,
            protocol_version=protocol_version,
            model_format=model_format,
            namespace=namespace,
        )

    except Exception as e:
        print(f"[ERROR] Failed to serve model: {e}")
        raise


def connect(source_dataset, model_isvc, destination_dataset):
    """
    normally in cogflow you check each dataset ,
    1) all of them should be streaming type
    2) then create sink and source and sequence for them
    3) for source and destination if the type is nats , create bridge for each of them as well
    """
    try:
        PluginManager().load_config()
    except ConfigException as e:
        print(f"[config] ERROR: {e}")

    try:
        KnativePlugin().connect(
            source_dataset=source_dataset,
            model_isvc=model_isvc,
            destination_dataset=destination_dataset,
        )
    except Exception as e:
        print(f"Failed to connect datasets: {e}")
        raise e


def register_model_api(
    model_name,
    artifact_path,
    registered_model_name=None,
    conda_env=None,
    code_paths=None,
    serialization_format=plugin_config.SERIALIZATION_FORMAT,
    signature: ModelSignature = None,
    input_example: Union[
        pd.DataFrame,
        np.ndarray,
        dict,
        list,
        csr_matrix,
        csc_matrix,
        str,
        bytes,
        tuple,
    ] = None,
    await_registration_for=plugin_config.AWAIT_REGISTRATION_FOR,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn=plugin_config.PYFUNC_PREDICT_FN,
    metadata=None,
):
    """
    Logs a model.

    Args:
        model_name: The model to log.
        artifact_path (str): The artifact path to log the model to.
        registered_model_name (str, optional): The name to register the model under.
        conda_env (str, optional): The conda environment to use.
        code_paths (list, optional): List of paths to include in the model.
        serialization_format (str, optional): The format to use for serialization.
        signature (ModelSignature, optional): The signature of the model.
        input_example (Union[pd.DataFrame, np.ndarray, dict, list, csr_matrix, csc_matrix, str,
         bytes, tuple], optional): Example input.
        await_registration_for (int, optional): Time to wait for registration.
        pip_requirements (list, optional): List of pip requirements.
        extra_pip_requirements (list, optional): List of extra pip requirements.
        pyfunc_predict_fn (str, optional): The prediction function to use.
        metadata (dict, optional): Metadata for the model.
    """
    PluginManager().load_config()
    is_custom_pyfunc_model = isinstance(model_name, pyfunc.PythonModel) or (
        inspect.isclass(model_name) and issubclass(model_name, pyfunc.PythonModel)
    )

    if is_custom_pyfunc_model:
        # Log using pyfunc flavor
        result = custom_log_model(
            artifact_path=artifact_path,
            python_model=model_name,
            code_path=code_paths,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            metadata=metadata,
        )
    else:
        # Log using MLflowPlugin (e.g., sklearn, XGBoost, etc.)
        result = MlflowPlugin().log_model(
            sk_model=model_name,
            artifact_path=artifact_path,
            conda_env=conda_env,
            code_paths=code_paths,
            serialization_format=serialization_format,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            await_registration_for=await_registration_for,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            pyfunc_predict_fn=pyfunc_predict_fn,
            metadata=metadata,
        )

    display_name = registered_model_name or model_name

    reg = register_model(model_uri=result.model_uri, model_name=display_name)
    result.model_name = reg.name
    result.model_version = reg.version

    # Fetch run tags
    run = cogclient.get_run(result.run_id)
    tags = run.data.tags

    # Traverse all tags and append each one individually as an attribute
    if tags:
        for key, value in tags.items():
            # normalize key: replace invalid characters with underscores
            safe_key = key.replace(".", "_").replace("-", "_")
            setattr(result, safe_key, value)

    return result


def update_served_model(
    isvc_name: str,
    model_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    dataset_id: Optional[str] = None,
    transformer_image: Optional[str] = plugin_config.TRANSFORMER_BASE_IMAGE,
    transformer_parameters: Optional[dict] = None,
    protocol_version: Optional[str] = None,
    namespace: Optional[str] = None,
    model_format: Optional[str] = None,
    canary_traffic_percent: Optional[int] = None,
    enable_tag_routing: Optional[bool] = False,
) -> str:
    """
    Update or roll out a model on an existing KServe InferenceService.

        Supports:
            - Normal update (no canary)
            - Canary rollout (traffic split)
            - Canary promotion (increase or full switch)
            - Disable canary
    Args:
        isvc_name (str): Name of the KServe InferenceService to update.
        model_id (str, optional): Unique identifier for the model/run.
        model_name (str, optional): Registered model name (alternative to model_id).
        model_version (str, optional): Registered model version.
        artifact_path (str, optional): Specific artifact path (e.g., "model").
        dataset_id (str, optional): Dataset linked to the model.
        transformer_image (str, optional): Image of the transformer.
            Required if transformer_parameters is provided.
        transformer_parameters (dict, optional): Parameters for the transformer.

        protocol_version (str, optional): Protocol version for the model server (e.g., "v1", "v2").
        namespace (str, optional): Kubernetes namespace of the InferenceService.
        model_format (str, optional): Model format (e.g., "mlflow", "sklearn").
        canary_traffic_percent (int, optional): % of traffic routed to canary model.
        enable_tag_routing (bool, optional): Explicitly enable tag routing (no auto).

    Returns:
        str: Success message.

    Raises:
        ValueError: If neither model_id nor (model_name + model_version) is provided.
        RuntimeError: If the InferenceService does not exist.
        PermissionError: If RBAC/namespace access is forbidden.
        Exception: For any errors during model resolution or patching.
    """
    try:
        # ---------------------------------------------------------------------
        # 🧩 CASE 1: Only promote or adjust traffic (no new model involved)
        # ---------------------------------------------------------------------
        if canary_traffic_percent is not None and not (
            model_id or model_name or model_version
        ):
            # Validate the traffic value based on current ISVC state
            KubeflowPlugin().validate_canary_traffic_percent(
                isvc_name=isvc_name, canary_traffic_percent=canary_traffic_percent
            )
            return KubeflowPlugin().update_served_model(
                isvc_name=isvc_name,
                namespace=namespace,
                canary_traffic_percent=canary_traffic_percent,
            )

        # ---------------------------------------------------------------------
        # 🧩 CASE 2: Model rollout or update (new model introduced)
        # ---------------------------------------------------------------------
        if not model_id and not (model_name and model_version):
            raise ValueError(
                "Must provide either model_id or (model_name and model_version) "
                "when performing a model update or canary rollout."
            )

        # Resolve full model URI and metadata
        model_details = MlflowPlugin().get_full_model_uri_from_run_or_registry(
            model_id=uuid_to_hex(model_id),
            artifact_path=artifact_path,
            model_name=model_name,
            model_version=model_version,
        )

        transformer_parameters = transformer_parameters or {}

        # Handle Prometheus dataset preprocessor config
        if dataset_id is not None and not transformer_parameters:
            dataset = get_dataset(
                dataset_id=dataset_id, endpoint=PluginManager().load_path("dataset")
            )
            if dataset.get("data_source_type") == 20:
                if not transformer_image:
                    raise ValueError(
                        "Dataset is of Prometheus type. You must provide a 'transformer_image' "
                        "to handle preprocessing for the transformer."
                    )
                dataset_response = get_dataset(
                    dataset_id=dataset_id,
                    endpoint=PluginManager().load_path("prometheus_dataset"),
                )
                transformer_parameters = {
                    "PROMETHEUS_URL": dataset_response.get("connection_type", {}).get(
                        "prometheus_url"
                    ),
                    "PROMETHEUS_METRICS": dataset_response.get("metric_list", {}).get(
                        "METRIC_FEATURES"
                    ),
                }

        # Auto-detect model format if not provided
        if model_format is None:
            model_format = MlflowPlugin().detect_model_format(
                model_details["model_uri"]
            )

        # Validate canary range if requested
        if canary_traffic_percent is not None:
            KubeflowPlugin().validate_canary_traffic_percent(
                isvc_name=isvc_name, canary_traffic_percent=canary_traffic_percent
            )

        # ---------------------------------------------------------------------
        # 🧩 Update or Rollout model using KubeflowPlugin
        # ---------------------------------------------------------------------
        return KubeflowPlugin().update_served_model(
            isvc_name=isvc_name,
            model_name=model_details["model_name"],
            model_version=model_details["model_version"],
            model_uri=model_details["model_uri"],
            model_id=uuid_to_canonical(model_details["model_id"]),
            dataset_id=dataset_id,
            transformer_image=transformer_image,
            transformer_parameters=transformer_parameters,
            protocol_version=protocol_version,
            namespace=namespace,
            model_format=model_format,
            canary_traffic_percent=canary_traffic_percent,
            enable_tag_routing=enable_tag_routing,
        )

    except Exception as e:
        print(f"[ERROR] Failed to update served model: {e}")
        raise


def set_tag(key: str, value: Any) -> None:
    """
    Set a tag under the current run. If no run is active, this method will create a
    new active run.

    :param key: Tag name (string). This string may only contain alphanumerics, underscores
                (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                All backend stores will support keys up to length 250, but some may
                support larger keys.
    :param value: Tag value (string, but will be string-ified if not).
                  All backend stores will support values up to length 5000, but some
                  may support larger values.
    """
    return MlflowPlugin().set_tag(key=key, value=value)


def get_model_url(
    isvc_name: str,
    namespace: str = None,
) -> str:
    """
    Gets information about inference service of served models

    Args:
        namespace (str): Namespace where isvc is deployed.
        isvc_name (str, optional): Name of served model.

    Returns:
        list: List of model information dictionaries. Each dict contains:
              model_name, model_id, model_version, creation_timestamp,
              served_model_url, status, traffic_percentage.
    """
    info = KubeflowPlugin().get_served_models(isvc_name=isvc_name, namespace=namespace)
    if isinstance(info, list):  # sometimes returns [ { ... } ]
        info = info[0]
    return info.get("served_model_url", None)


def update_artifact(
    run_id: str,
    local_path: str,
    artifact_path: str = None,
):
    """
    Update (overwrite) an artifact in S3/MinIO for a given run_id.

    Behavior:
        - File name is always inferred from ``local_path``.
        - If ``artifact_path`` is provided → file is stored inside that folder.
        - If ``artifact_path`` is None → file is stored at the root of artifacts/.

    Args:
        run_id (str): run ID.
        local_path (str): Path to the local file to upload.
        artifact_path (str, optional): Subdirectory under artifacts/.
            If None, file is stored directly under artifacts/.

    Returns:
        str: The full S3 URI of the updated artifact.

    Examples:
        # Case 1: Update inside a folder
        >>> update_artifact(
        ...     run_id="<run_id>",
        ...     local_path="<local_path>",
        ...     artifact_path="<artifact_path>"
        ... )
        # → s3://mlflow/<experiment_id>/<run_id>/artifacts/<artifact_path>/<local_path_file_name>

        # Case 2: Update at root/base_path
        >>> update_artifact(
        ...     run_id="<run_id>",
        ...     local_path="<local_path>"
        ... )
        # → s3://mlflow/<experiment_id>/<run_id>/artifacts/<local_path_file_name>
    """
    PluginManager().load_config()

    exp_id = MlflowPlugin().get_experiment_id_from_run(run_id)

    # Validate local file exists
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Infer file name from local_path
    file_name = os.path.basename(local_path)

    # Build the object key
    if artifact_path:
        key = f"{exp_id}/{run_id}/artifacts/{artifact_path}/{file_name}"
    else:
        key = f"{exp_id}/{run_id}/artifacts/{file_name}"

    # Load environment variables
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket = plugin_config.BUCKET_NAME

    if not endpoint_url or not access_key or not secret_key:
        raise EnvironmentError(
            "Missing one or more required environment variables: "
            "MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
        )

    # Init S3 client
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize S3 client: {e}") from e

    # Upload file (overwrite if exists)
    try:
        s3.upload_file(local_path, bucket, key)
    except NoCredentialsError:
        raise RuntimeError("Invalid or missing credentials.")
    except ClientError as e:
        raise RuntimeError(f"S3 upload failed: {e.response['Error']['Message']}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during S3 upload: {e}") from e

    s3_uri = f"s3://{bucket}/{key}"
    print(f"Artifact updated: {s3_uri}")
    return s3_uri


def delete_artifact(
    run_id: str,
    file_name: str,
    artifact_path: str = None,
):
    """
    Delete an artifact (object) from S3/MinIO for a given MLflow run.

    Behavior:
        - If ``artifact_path`` is provided → deletes file inside that folder.
        - If ``artifact_path`` is None → deletes file directly under ``artifacts/``.

    Args:
        run_id (str): MLflow run ID.
        file_name (str): File name with extension (e.g., "kafka-sink.yaml").
        artifact_path (str, optional): Subdirectory under artifacts/.
            If None, file is deleted from the root of artifacts/.

    Returns:
        str or None: The full S3 URI of the deleted artifact,
                     or None if the object was not found.

    Raises:
        EnvironmentError: If required environment variables are missing.
        RuntimeError: If S3 client creation or deletion fails.

    Examples:
        # Case 1: Delete inside a folder
        >>> delete_artifact(
        ...     run_id="<run_id>",
        ...     file_name="<file_name>",
        ...     artifact_path="<artifact_path>"
        ... )
        # → deletes s3://mlflow/<experiment_id>/<run_id>/artifacts/<artifact_path>/<file_name>

        # Case 2: Delete at root/base_path
        >>> delete_artifact(
        ...     run_id="<run_id>",
        ...     file_name="<file_name>"
        ... )
        # → deletes s3://mlflow/<experiment_id>/<run_id>/artifacts/<file_name>
    """

    PluginManager().load_config()

    exp_id = MlflowPlugin().get_experiment_id_from_run(run_id)

    # Build the object key
    if artifact_path:
        key = f"{exp_id}/{run_id}/artifacts/{artifact_path}/{file_name}"
    else:
        key = f"{exp_id}/{run_id}/artifacts/{file_name}"

    # Load environment variables
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket = plugin_config.BUCKET_NAME

    if not endpoint_url or not access_key or not secret_key:
        raise EnvironmentError(
            "Missing one or more required environment variables: "
            "MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
        )

    # Init S3 client safely, create as new method later
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize S3 client: {e}") from e

    # Delete with existence check
    try:
        s3.head_object(Bucket=bucket, Key=key)  # verify object exists
        s3.delete_object(Bucket=bucket, Key=key)
        s3_uri = f"s3://{bucket}/{key}"
        print(f"Deleted: {s3_uri}")
        return s3_uri
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Not found: s3://{bucket}/{key}")
            return None
        raise RuntimeError(
            f"S3 deletion failed: {e.response['Error']['Message']}"
        ) from e
    except NoCredentialsError:
        raise RuntimeError("Invalid or missing AWS credentials.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during S3 deletion: {e}") from e


def search_runs(
    experiment_ids: List[str],
    filter_string: str = "",
    max_results: int = None,
    order_by: Optional[List[str]] = None,
    page_token: Optional[str] = None,
):
    """
    Search for Runs that fit the specified criteria.

    :param experiment_ids: List of experiment IDs, or a single int or string id.
    :param filter_string: Filter query string, defaults to searching all runs.
    :param max_results: Maximum number of runs desired.
    :param order_by: List of columns to order by (e.g., "metrics.rmse"). The ``order_by`` column
                 can contain an optional ``DESC`` or ``ASC`` value. The default is ``ASC``.
                 The default ordering is to sort by ``start_time DESC``, then ``run_id``.
    :param page_token: Token specifying the next page of results. It should be obtained from
        a ``search_runs`` call.

    :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
        :py:class:`Run <mlflow.entities.Run>` objects that satisfy the search expressions.
        If the underlying tracking store supports pagination, the token for the next page may
        be obtained via the ``token`` attribute of the returned object.

    . code-block:: python
        :caption: Example

        import cogflow
        from cogflow import cogclient

        def print_run_info(runs):
            for r in runs:
                print("run_id: {}".format(r.info.run_id))
                print("lifecycle_stage: {}".format(r.info.lifecycle_stage))
                print("metrics: {}".format(r.data.metrics))

                # Exclude cogflow system tags

                tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
                print("tags: {}".format(tags))

        # Create an experiment and log two runs with metrics and tags under the experiment
        experiment_id = cogflow.create_experiment("Social NLP Experiments")
        with cogflow.start_run(experiment_id=experiment_id) as run:
            cogflow.log_metric("m", 1.55)
            cogflow.set_tag("s.release", "1.1.0-RC")
        with cogflow.start_run(experiment_id=experiment_id):
            cogflow.log_metric("m", 2.50)
            cogflow.set_tag("s.release", "1.2.0-GA")

        # Search all runs under experiment id and order them by
        # descending value of the metric 'm'
        client = cogclient
        runs = client.search_runs(experiment_id, order_by=["metrics.m DESC"])
        print_run_info(runs)
        print("--")

        # Delete the first run
        client.delete_run(run_id=run.info.run_id)

        # Search only deleted runs under the experiment id and use a case-insensitive pattern
        # in the filter_string for the tag.
        filter_string = "tags.s.release ILIKE '%rc%'"
        runs = client.search_runs(experiment_id, run_view_type=ViewType.DELETED_ONLY,
                                    filter_string=filter_string)
        print_run_info(runs)

    . code-block:: text
        :caption: Output

        run_id: 0efb2a68833d4ee7860a964fad31cb3f
        lifecycle_stage: active
        metrics: {'m': 2.5}
        tags: {'s.release': '1.2.0-GA'}
        run_id: 7ab027fd72ee4527a5ec5eafebb923b8
        lifecycle_stage: active
        metrics: {'m': 1.55}
        tags: {'s.release': '1.1.0-RC'}
        --
        run_id: 7ab027fd72ee4527a5ec5eafebb923b8
        lifecycle_stage: deleted
        metrics: {'m': 1.55}
        tags: {'s.release': '1.1.0-RC'}
    """
    PluginManager().load_config()

    return MlflowPlugin().search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


def create_experiment(
    name: str,
    artifact_location: str = None,
    tags: dict = None,
) -> str:
    """
    Create a new experiment.

    Args:
        name (str): Name of the experiment to create.
        artifact_location (str, optional): Base location to store artifacts
            for runs in this experiment. If not provided, the default
            artifact root from config is used.
        tags (dict, optional): Dictionary of key-value tags to set on the experiment.

    Returns:
        str: The experiment ID of the newly created experiment.

    Raises:
       exceptions: If experiment creation fails.

    Examples:
        >>> from cogflow import cogclient

        # Create a basic experiment
        >>> exp_id = cogclient.create_experiment("my_experiment")
        >>> print(exp_id)
        '2'

        # Create an experiment with custom artifact location and tags
        >>> exp_id = cogclient.create_experiment(
        ...     "experiment_with_tags",
        ...     artifact_location="s3://mlflow/artifacts",
        ...     tags={"team": "ml", "env": "staging"}
        ... )
        >>> print(exp_id)
        '3'
    """
    PluginManager().load_config()
    return MlflowPlugin().create_experiment(
        name=name, artifact_location=artifact_location, tags=tags
    )


def register_prometheus_dataset(
    dataset_name: str,
    description: str,
    prometheus_url: str,
    metric_features: str,
    dataset_type: int,
    feature_list: Optional[Dict[str, Any]] = None,
    connection_parameter: Optional[Dict[str, Any]] = None,
    target_namespace: str = "default",
    query_duration: str = None,
    frequency: str = None,
    timeout: str = None,
    data_source: str = None,
) -> Dict[str, Any]:
    """Register a new Prometheus dataset in Cogflow.

    Args:
        dataset_name: Unique dataset name.
        description: Dataset description.
        prometheus_url: URL of Prometheus endpoint.
        metric_features: Comma-separated list of metrics to capture.
        dataset_type: The type of the dataset in (train dataset - 0, inference dataset 1, both- 2).
        feature_list: Additional Prometheus label filters or static metadata.
        connection_parameter: Additional connection or auth params.
        target_namespace: Namespace to query metrics from.
        query_duration: PromQL query window.
        frequency: Query collection frequency.
        timeout: Query timeout.
        data_source: Data source label.

    Returns:
        Parsed JSON API response.

    Raises:
        ValueError: If required parameters are missing.
        RuntimeError: On API or connection failure.
    """
    PluginManager().load_config()
    # Validate required parameters
    required_params = [
        ("dataset_name", dataset_name),
        ("description", description),
        ("prometheus_url", prometheus_url),
        ("metric_features", metric_features),
        ("dataset_type", dataset_type),
    ]
    missing = [name for name, value in required_params if not value]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    feature_list = feature_list or {}
    connection_parameter = connection_parameter or {}

    api_url = f"{os.getenv(API_BASEPATH)}" + f"{plugin_config.PROMETHEUS_DATASETS}"

    payload = {
        "connection_type": {
            "prometheus_url": prometheus_url,
            "frequency": frequency,
            "timeout": timeout,
            "data_source": data_source,
        },
        "metric_list": {
            "METRIC_FEATURES": metric_features,
            "TARGET_NAMESPACE": target_namespace,
            "QUERY_DURATION": query_duration,
        },
        "feature_list": feature_list,
        "connection_parameter": connection_parameter,
        "dataset_name": dataset_name,
        "description": description,
        "dataset_type": dataset_type,
    }

    headers = {
        "kubeflow-userid": KubeflowPlugin().get_current_user_from_namespace(),
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        raise RuntimeError(f"HTTP error: {http_err}")
    except Exception as exp:
        raise RuntimeError(f"Error while registering Prometheus dataset: {exp}")


def get_run_artifact_uri(run_id: str) -> str:
    """
    Get the artifact URI for a given run ID.
    Args:
        run_id: run ID.

    Returns:
        str: Artifact URI.

    """
    PluginManager().load_config()
    run = cogclient.get_run(uuid_to_hex(run_id))
    artifact_uri = run.info.artifact_uri
    return artifact_uri


def list_artifacts_grouped(run_id: str):
    """
    Recursively list artifacts for the given run_id,
    grouped by directory.

    Returns:
        dict: {
            "root": [ "file_name", "..."],
            "subdir": [ "file_name","..."]
        }
    """
    PluginManager().load_config()

    grouped = defaultdict(list)

    def _walk(path="", prefix="root"):
        artifacts = cogclient.list_artifacts(uuid_to_hex(run_id), path)
        for a in artifacts:
            if a.is_dir:
                _walk(a.path, prefix=a.path)
            else:
                # Extract directory name, fallback to root
                directory = prefix or "root"
                grouped[directory].append(a.path.split("/")[-1])

    _walk()
    return dict(grouped)


__all__ = [
    # Methods from MlflowPlugin class
    "InputPath",
    "OutputPath",
    "ParallelFor",
    "pyfunc",
    "mlflow",
    "sklearn",
    "cogclient",
    "tensorflow",
    "pytorch",
    "models",
    "lightgbm",
    "xgboost",
    # Method from CogContainer class
    "add_model_access",
    "kfp",
    "v2",
]
