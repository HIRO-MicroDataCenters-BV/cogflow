"""
This module provides functionality related to Mlflow.
"""

import os
from typing import Union, Any, List, Optional, Dict

import mlflow as ml
import numpy as np
import pandas as pd
import requests
from mlflow.entities import ViewType, Run
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.store.entities import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
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

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
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
        # Verify plugin activation
        PluginManager().verify_activation(MlflowPlugin().section)

        return self.mlflow.log_metrics(metrics, step=step)

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

    def get_full_model_uri_from_run_or_registry(
        self,
        model_id: str = None,
        artifact_path: str = None,
        model_name: str = None,
        model_version: str = None,
    ) -> dict:
        """
        Returns model_uri, model_name, model_version, and run_id given either run_id or model_name/version.

        Args:
            model_id (str, optional): MLflow run ID.
            artifact_path (str, optional): Path like 'model'.
            model_name (str, optional): Registered model name.
            model_version (str, optional): Registered model version.

        Returns:
            dict: {
                'model_uri': 's3://.../artifacts/...',
                'model_name': 'your-model',
                'model_version': '1',
                'run_id': 'abc123'
            }

        Raises:
            ValueError / Exception: If inputs are invalid or model path cannot be resolved.
        """
        client = self.cogclient

        # 1. If model_name & model_version provided → get run_id from registry
        if not model_id:
            if not (model_name and model_version):
                raise ValueError(
                    "Either `run_id` or both `model_name` and `model_version` must be provided."
                )

            try:
                mv = client.get_model_version(
                    name=model_name, version=str(model_version)
                )
                model_id = mv.run_id
            except MlflowException as e:
                raise Exception(f"Failed to fetch model version from registry: {e}")

        # 2. Get artifact URI from run
        run = self.mlflow.get_run(model_id)
        artifact_uri = run.info.artifact_uri  # s3://mlflow/...

        # 3. If artifact_path is provided, build URI directly
        if artifact_path:
            model_uri = f"{artifact_uri}/{artifact_path}"
        else:
            artifacts = client.list_artifacts(model_id)
            dirs = [a for a in artifacts if a.is_dir]
            model_files = [
                a
                for a in artifacts
                if a.path.endswith(
                    (".pkl", ".joblib", ".onnx", ".pt", ".sav", ".mlmodel")
                )
            ]

            if len(dirs) == 1:
                model_uri = f"{artifact_uri}/{dirs[0].path}"
            elif len(dirs) > 1:
                dir_names = [d.path for d in dirs]
                raise Exception(
                    f"Multiple artifact paths found in run `{model_id}`: {dir_names}. "
                    f"Please specify the `artifact_path` explicitly."
                )
            elif model_files:
                model_uri = f"{artifact_uri}/{model_files[0].path}"
            else:
                raise Exception(
                    f"No model artifact_path or supported model file found in run `{model_id}`.\n"
                    f"Please provide the artifact_path explicitly using `artifact_path=`."
                )

        # Step 4: If model_name/version is still unknown, resolve from registry
        if not model_name or not model_version:
            results = self.search_model_versions(filter_string=f"run_id = '{model_id}'")
            for mv in results:
                model_name = mv.name
                model_version = mv.version
                break

        return {
            "model_uri": model_uri,
            "model_name": model_name,
            "model_version": model_version,
            "model_id": model_id,
        }

    def set_tag(self, key: str, value: Any) -> None:
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
        return self.mlflow.set_tag(key=key, value=value)

    def create_experiment(
        self,
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
        return self.cogclient.create_experiment(
            name=name, artifact_location=artifact_location, tags=tags
        )

    def get_experiment_id_from_run(self, run_id: str) -> str:
        """
        Fetch the experiment ID associated with a given run ID.

        This function looks up the run metadata in the tracking server
        and returns the `experiment_id` to which the run belongs.

        Args:
            run_id (str): The unique run identifier (UUID-like string).

        Returns:
            str: The experiment ID (as a string) associated with the given run.

        Raises:
            exceptions: If the run cannot be found
                or tracking server is unreachable.

        Examples:
            >>> import cogflow
            >>> from cogflow import cogclient, mlflow
            >>> run_name = cogflow.start_run()
            >>> run_info_id = run_name.info.run_id
            >>> cogflow.end_run()

            # Fetch experiment ID for the run
            >>> exp_id = mlflow.get_experiment_id_from_run(run_info_id)
            >>> print(exp_id)
            '0'   # (default experiment if none was specified)

            # Works with runs from non-default experiments as well
            >>> cogclient.create_experiment("my_exp")
            >>> with cogflow.start_run(experiment_id=1) as run_name:
            ...     print(mlflow.get_experiment_id_from_run(run_name.info.run_id))
            '1'
        """
        run = self.mlflow.get_run(run_id)
        return run.info.experiment_id

    def detect_model_format(self, model_uri: str) -> str:
        """
        Detect the model format (flavor) from an MLflow model URI.

        Args:
            model_uri (str): Path/URI to the MLflow model.

        Returns:
            str: "mlflow" if pyfunc flavor is present,
                 "sklearn" if sklearn flavor is present,
                 otherwise "unknown".
        """
        model_info = self.mlflow.models.get_model_info(model_uri)
        flavors = model_info.flavors.keys()

        if "python_function" in flavors:
            return "mlflow"
        elif "sklearn" in flavors:
            return "sklearn"
        else:
            return "unknown"

    def log_params(self, params: Dict[str, Any]) -> None:
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
        return self.mlflow.log_params(params)

    def log_artifacts(
        self, local_dir: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Log all the contents of a local directory as artifacts of the run. If no run is active,
        this method will create a new active run.

        :param local_dir: Path to the directory of files to write.
        :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.

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
        return self.mlflow.log_artifacts(
            local_dir=local_dir, artifact_path=artifact_path
        )

    def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str = "",
        run_view_type: int = ViewType.ACTIVE_ONLY,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: Optional[List[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[Run]:
        """
        Search for Runs that fit the specified criteria.

        :param experiment_ids: List of experiment IDs, or a single int or string id.
        :param filter_string: Filter query string, defaults to searching all runs.
        :param run_view_type: one of enum values ACTIVE_ONLY, DELETED_ONLY, or ALL runs
                              defined in :py:class:`mlflow.entities.ViewType`.
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
        return self.cogclient.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            run_view_type=run_view_type,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )
