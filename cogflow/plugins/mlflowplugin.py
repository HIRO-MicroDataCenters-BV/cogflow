"""
This module provides functionality related to Mlflow.
"""

import os
from typing import Union, Any, List, Optional, Dict

import mlflow as ml
import numpy as np
import pandas as pd
import requests
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from scipy.sparse import csr_matrix, csc_matrix
from .. import plugin_config
from ..pluginmanager import PluginManager


class MlflowPlugin:
    """
    Class for defining reusable components.
    """

    def __init__(self):
        """
        Initializes the MlFlowPlugin class.
        """
        self.mlflow = ml
        self.sklearn = ml.sklearn
        self.cogclient = MlflowClient()
        self.pyfunc = ml.pyfunc
        self.tensorflow = ml.tensorflow
        self.pytorch = ml.pytorch
        self.models = ml.models
        self.lightgbm = ml.lightgbm
        self.xgboost = ml.xgboost
        self.section = "mlflow_plugin"

    @staticmethod
    def is_alive():
        """
        Check if Mlflow UI is accessible.

        Returns:
            tuple: A tuple containing a boolean indicating if Mlflow UI is accessible
             and the status code of the response.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        try:
            response = requests.get(os.getenv(plugin_config.TRACKING_URI), timeout=300)

            if response.status_code == 200:
                pass
            else:
                print(
                    f"Mlflow UI is not accessible. Status code: {response.status_code}, "
                    f"Message: {response.text}"
                )
            return response.status_code, response.text
        except Exception as exp:
            print(f"An error occurred while accessing Mlflow UI: {str(exp)}, ")
            raise exp

    @staticmethod
    def version():
        """
        Retrieve the version of the Mlflow.

        Returns:
            str: Version of the Mlflow.
        """
        return ml.__version__

    def delete_registered_model(self, model_name):
        """
        Deletes a registered model with the given name.

        Args:
            model_name (str): The name of the registered model to delete.

        Returns:
            bool: True if the model was successfully deleted, False otherwise.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.cogclient.delete_registered_model(model_name)

    def search_registered_models(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
        page_token: Optional[str] = None,
    ):
        """
        Searches for registered models in Mlflow.

        This method allows you to search for registered models using optional filtering criteria,
        and retrieve a list of registered models that match the specified criteria.

        Args:
            filter_string(Optional[str]): A string used to filter the registered models. The filter
                string can include conditions on model name, tags, and other attributes.
                For example, "name='my_model' AND tags.key='value'". If not provided, all registered
                models are returned.
            max_results (int): The maximum number of results to return. Defaults to 100.
            order_by (Optional[List[str]]): A list of property keys to order the results by.
                For example, ["name ASC", "version DESC"].
            page_token (Optional[str]): A token to specify the page of results to retrieve. This is
                useful for pagination when there are more results than can be returned in a
                single call.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a registered model
            that matches the search criteria. Each dictionary contains details about the registered
            model, such as its name, creation timestamp, last updated timestamp, tags,
            and description.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        registered_models = self.cogclient.search_registered_models(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )
        return registered_models

    @staticmethod
    def load_model(model_uri: str, dst_path=None):
        """
        Loads a model from the specified Mlflow model URI.

        Args:
            model_uri (str): The URI of the Mlflow model to load.
            dst_path (str, optional): Optional path where the model will be downloaded and saved.
             If not provided, the model will be loaded without saving.

        Returns:
            loaded_model: The loaded Mlflow model.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        loaded_model = ml.sklearn.load_model(model_uri, dst_path)
        return loaded_model

    def evaluate(
        self,
        model: str,
        data,
        *,
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
        env_manager="local",
    ):
        """
        Evaluate the performance of a machine learning model using various metrics and techniques.

        Args:
            model (object, optional): The machine learning model to evaluate. Default is None.
            data (object, optional): The dataset or data object to evaluate the model on.
            Default is None.
            model_type (str, optional): Type of the model being evaluated. Default is None.
            targets (array-like, optional): The true target values. Default is None.
            dataset_path (str, optional): Path to the dataset if not directly provided.
            Default is None.
            feature_names (list, optional): Names of features in the dataset. Default is None.
            evaluators (list, optional): List of evaluators to use for evaluation. Default is None.
            evaluator_config (dict, optional): Configuration for the evaluators. Default is None.
            custom_metrics (dict, optional): Additional custom metrics to compute. Default is None.
            custom_artifacts (dict, optional): Custom artifacts to save during evaluation.
            Default is None.
            validation_thresholds (dict, optional): Thresholds for validation. Default is None.
            baseline_model (object, optional): Baseline model for comparison. Default is None.
            env_manager (str, optional): Environment manager to use for evaluation.
            Default is 'local'.

        Returns:
            dict: Evaluation results including various metrics and artifacts.
        """
        PluginManager().verify_activation(MlflowPlugin().section)
        return self.mlflow.evaluate(
            model=model,
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

    @staticmethod
    def register_model(
        model_uri: str,
        model: str,
        await_registration_for: int = 300,
        *,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Registers the given model with Mlflow.

        This method registers a model with Mlflow using the specified model URI. Optionally,
        tags can be added to the registered model for better organization and metadata tracking.

        Args:
            model_uri (str): The URI of the Mlflow model to register.
            model (str): The name under which to register the model in the Mlflow Model Registry.
            await_registration_for (int, optional): The duration, in seconds, to wait for the model
            version to finish being created and be in the READY status. Defaults to 300 seconds.
            tags (Optional[Dict[str, Any]], optional): A dictionary of key-value pairs to tag
            the registered model
                with. Tags can be useful for organizing and filtering models in the registry.

        Returns:
            ModelVersion: An instance of `ModelVersion` representing the registered model version.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return ml.register_model(
            name=model,
            model_uri=model_uri,
            await_registration_for=await_registration_for,
            tags=tags,
        )

    def autolog(self):
        """
        Enable automatic logging of parameters, metrics, and models with Mlflow.

        Returns:
            None
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.autolog()

    def create_registered_model(
        self,
        model: str,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        """
        Create a registered model in the Mlflow Model Registry.

        This method creates a new registered model in the Mlflow Model Registry with
        the given name. Optionally,
        tags and a description can be added to provide additional metadata about the model.

        Args:
            model (str): The name of the registered model.
            tags (Optional[Dict[str, Any]], optional): A dictionary of key-value pairs to
            tag the registered model
                with. Tags can be useful for organizing and filtering models in the registry.
            description (Optional[str], optional): A description of the registered model.
            This can provide additional context about the model's purpose, usage, or any other
                relevant information.

        Returns:
            RegisteredModel: An instance of `RegisteredModel` representing the
            created registered model.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.cogclient.create_registered_model(
            name=model, tags=tags, description=description
        )

    def create_model_version(
        self,
        model: str,
        source: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        run_link: Optional[str] = None,
        description: Optional[str] = None,
        await_creation_for: int = 300,
    ):
        """
        Create a model version for a registered model in the Mlflow Model Registry.

        This method registers a new version of an existing registered model with the given
        source path or URI.
        Optionally, additional metadata such as run ID, tags, run link, and description
        can be provided.
        The `await_creation_for` parameter allows specifying a timeout for waiting for the
        model version creation to complete.

        Args:
            model (str): The name of the registered model.
            source (str): The source path or URI of the model. This is the location where the
            model artifacts are stored.
            run_id (Optional[str], optional): The ID of the run associated with this model version.
            This can be useful
                for tracking the lineage of the model version.
            tags (Optional[Dict[str, Any]], optional): A dictionary of key-value pairs to tag
            the model version with.
                Tags can help in organizing and filtering model versions.
            run_link (Optional[str], optional): A URI link to the run. This can provide quick
            access to the run details.
            description (Optional[str], optional): A description of the model version. This can
            provide additional context about the changes or improvements in this version.
            await_creation_for (int, optional): The time in seconds to wait for the model
            version creation to complete.
                Defaults to 300 seconds.

        Returns:
            ModelVersion: An instance of `ModelVersion` representing the created model version.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.cogclient.create_model_version(
            name=model,
            source=source,
            run_id=run_id,
            tags=tags,
            run_link=run_link,
            description=description,
            await_creation_for=await_creation_for,
        )

    def set_tracking_uri(self, tracking_uri):
        """
        Set the Mlflow tracking URI.

        Args:
            tracking_uri (str): The URI of the Mlflow tracking server.

        Returns:
            None
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.set_tracking_uri(tracking_uri)

    def set_experiment(
        self, experiment_name: Optional[str] = None, experiment_id: Optional[str] = None
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
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.set_experiment(
            experiment_name=experiment_name, experiment_id=experiment_id
        )

    def get_artifact_uri(self, artifact_path: Optional[str] = None):
        """
        Get the artifact URI of the current or specified Mlflow run.

        This method returns the URI of the artifact directory for the current run or
        for the specified artifact path.

        Args:
            artifact_path (Optional[str]): The path of the artifact within the run's
                artifact directory.
                If not provided, the method returns the URI of the current run's artifact directory.

        Returns:
            str: The URI of the specified artifact path or the current run's artifact directory.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.get_artifact_uri(artifact_path=artifact_path)

    def start_run(
        self,
        run_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        """
        Start Mlflow run.

        This method starts a new Mlflow run or resumes an existing run if a run_id is provided.

        Args:
            run_id (Optional[str]): The ID of the run to resume. If not provided,
            a new run is started.
            experiment_id (Optional[str]): The ID of the experiment under which to create the run.
            run_name (Optional[str]): The name of the Mlflow run.
            nested (bool): Whether to create the run as a nested run of the parent run.
            tags (Optional[Dict[str, Any]]): A dictionary of tags to set on the run.
            description (Optional[str]): A description for the run.

        Returns:
            mlflow.entities.Run: The Mlflow Run object corresponding to the started or resumed run.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested,
            tags=tags,
            description=description,
        )

    def end_run(self):
        """
        End a Mlflow run.

        Returns:
            Mlflow Run object
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.end_run()

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter to the Mlflow run.

        Args:
            key (str): The key of the parameter to log.
            value (Any): The value of the parameter to log.
                Defaults to True.

        Returns:
            None
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.log_param(key, value)

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a metric to the Mlflow run.

        Args:
            key (str): The name of the metric to log.
            value (float): The value of the metric to log.
            step (Optional[int], optional): Step to log the metric at. Defaults to None.

        Returns:
            None
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.log_metric(
            key,
            value,
            step=step,
        )

    def log_model(
        self,
        sk_model,
        artifact_path,
        conda_env=None,
        code_paths=None,
        serialization_format="cloudpickle",
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
        await_registration_for=300,
        pip_requirements=None,
        extra_pip_requirements=None,
        pyfunc_predict_fn="predict",
        metadata=None,
    ):
        """
        Log a scikit-learn model to Mlflow.

        Args:
            sk_model: The scikit-learn model to be logged.
            artifact_path (str): The run-relative artifact path to which the model artifacts will
            be saved.
            conda_env (str, optional): The path to a Conda environment YAML file. Defaults to None.
            code_paths (list, optional): A list of local filesystem paths to Python files that
            contain code to be
            included as part of the model's logged artifacts. Defaults to None.
            serialization_format (str, optional): The format used to serialize the model. Defaults
            to "cloudpickle".
            registered_model_name (str, optional): The name under which to register the model with
            Mlflow. Defaults to None.
            signature (ModelSignature, optional): The signature defining model input and output
            data types and shapes. Defaults to None.
            input_example (Union[pd.DataFrame, np.ndarray, dict, list, csr_matrix, csc_matrix, str,
            bytes, tuple], optional): An example input to the model. Defaults to None.
            await_registration_for (int, optional): The duration, in seconds, to wait for the
            model version to finish being created and is in the READY status. Defaults to 300.
            pip_requirements (str, optional): A file in pip requirements format specifying
            additional pip dependencies for the model environment. Defaults to None.
            extra_pip_requirements (str, optional): A string containing additional pip dependencies
            that should be added to the environment. Defaults to None.
            pyfunc_predict_fn (str, optional): The name of the function to invoke for prediction,
            when the model is a PyFunc model. Defaults to "predict".
            metadata (dict, optional): A dictionary of metadata to log with the model.
            Defaults to None.

        Returns:
            Model: The logged scikit-learn model.

        Raises:
            Exception: If an error occurs during the logging process.

        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)

        return self.mlflow.sklearn.log_model(
            sk_model=sk_model,
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

    def search_model_versions(
        self,
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

        Raises:
            Exception: If the plugin is not activated.
        """
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.cogclient.search_model_versions(
            filter_string=filter_string,
        )

    def get_model_uri(self, model_name, version):
        """
            return the model_uri given the model name and version
        :param model_name: name of the model
        :param version: version of the model
        :return: model_uri
        """
        model_version = self.cogclient.get_model_version(
            name=model_name, version=version
        )

        # Get the model URI
        model_uri = model_version.source

        return model_uri

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file or directory as an artifact of the currently active run. If no run is
        active, this method will create a new active run.

        :param local_path: Path to the file to write.
        :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.
        """
        return self.mlflow.log_artifact(
            local_path=local_path, artifact_path=artifact_path
        )
