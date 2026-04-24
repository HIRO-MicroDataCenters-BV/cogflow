"""
Core module for managing models within cogflow.

Handles ML integration, model lifecycle, versioning, and registry operations.

-------------------------------------------------------------------------------
📘 MODULE OVERVIEW (Method Index by Category)
-------------------------------------------------------------------------------

🔹 Health & Initialization
    - __init__
    - _check_tracking_server_health
    - _ensure_health
    - _warn_if_unhealthy

🔹 Model Lifecycle
    - load_model
    - register_model
    - create_registered_model
    - create_model_version
    - delete_registered_model
    - evaluate
    - log_model
    - autolog
    - detect_model_format
    - detect_model_type

🔹 Model Registry & Versioning
    - search_registered_models
    - search_model_versions
    - get_model_uri
    - get_full_model_uri_from_run_or_registry

🔹 Run Management
    - start_run
    - end_run
    - set_tag
    - set_experiment
    - get_experiment_id_from_run
    - search_runs

🔹 Experiment Management
    - create_experiment

🔹 Parameter Logging
    - log_param
    - log_params

🔹 Metric Logging
    - log_metric
    - log_metrics

🔹 Artifact Management
    - log_artifact
    - log_artifacts
    - get_artifact_uri

🔹 Tracking Server Configuration
    - set_tracking_uri

-------------------------------------------------------------------------------

"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from ..utils.exceptions import (
    CogflowErrorHandler,
    CogflowModelError,
    CogflowRunError,
    CogflowConnectionError,
    CogflowArtifactError,
    CogflowExperimentError,
    CogflowValidationError,
)
from ..utils.logging import get_logger
from ..utils.imports import lazy_import
from ..utils import network, common
from ..config import config

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix, csc_matrix
    from mlflow.models.signature import ModelSignature

logger = get_logger(__name__)


class ModelManager:
    """
    Central class for all model lifecycle operations within CogFlow.

    Features:
        - Lazy-loads MLflow dependencies for faster import.
        - Performs one-time MLflow tracking server health check.
        - Caches connection health for performance.
        - Revalidates automatically if previously unhealthy.
        - Provides standardized exception handling using CogflowErrorHandler.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize the ModelManager and verify MLflow tracking server availability.

        Args:
            strict (bool): If True, raises CogflowConnectionError when MLflow
                tracking server is unreachable.
        """
        # --- MLflow Core Modules ---
        self.mlflow = lazy_import("mlflow")
        self.client = lazy_import("mlflow.tracking").MlflowClient()

        # Common MLflow modules
        self.sklearn = lazy_import("mlflow.sklearn")
        self.pyfunc = lazy_import("mlflow.pyfunc")
        self.pytorch = lazy_import("mlflow.pytorch")
        # self.tensorflow = lazy_import("mlflow.tensorflow")
        self.xgboost = lazy_import("mlflow.xgboost")
        self.lightgbm = lazy_import("mlflow.lightgbm")

        # --- MLflow Support Types ---
        self.mlflowexception = lazy_import("mlflow.exceptions").MlflowException
        self.modelsignature = lazy_import("mlflow.models.signature").ModelSignature

        # --- Heavy External Libraries (Lazy Loaded) ---
        self.pd = lazy_import("pandas")
        self.np = lazy_import("numpy")
        self.scipy_sparse = lazy_import("scipy.sparse")
        self.os = lazy_import("os")
        self.inspect = lazy_import("inspect")

        self.csr_matrix = self.scipy_sparse.csr_matrix
        self.csc_matrix = self.scipy_sparse.csc_matrix

        # --- Health Check ---
        self._healthy = self._check_tracking_server_health()

        logger.info("Initializing ModelManager...")

        if not self._healthy:
            msg = "MLflow tracking server at %s is not reachable. Some operations may fail."
            if strict:
                CogflowErrorHandler.log_and_raise(
                    msg % config.MLFLOW_TRACKING_URI, CogflowConnectionError
                )
            logger.warning(msg, config.MLFLOW_TRACKING_URI)
        else:
            logger.info("MLflow tracking server is healthy and reachable.")

    @staticmethod
    def _check_tracking_server_health() -> bool:
        """Perform a one-time health check for the MLflow tracking server."""
        uri = config.MLFLOW_TRACKING_URI
        try:
            network.make_health_check_request(uri, timeout=config.DEFAULT_TIMEOUT)
            return True
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e, context="MLflow health check", raise_as=CogflowConnectionError
            )
            return False

    def _ensure_health(self) -> bool:
        """Re-check tracking server health if previously unhealthy."""
        if not self._healthy:
            logger.debug("Rechecking MLflow tracking server health...")
            self._healthy = self._check_tracking_server_health()
        return self._healthy

    def _warn_if_unhealthy(self, action: str):
        """Log a warning if MLflow tracking may be unreachable."""
        if not self._ensure_health():
            logger.warning(
                "MLflow tracking server may be unreachable — %s may fail.", action
            )

    def load_model(self, model_uri: str, dst_path: Optional[str] = None) -> Any:
        """Load a model from the specified MLflow model URI.
        Args:
            model_uri (str): The URI of the model to load.
            dst_path (str, optional): The destination path to load the model to.

        Returns:
            Any: The loaded model object.

        Example:
        >>> from cogflow import models
        >>> model = models.load_model("runs:/a1b2c3d4e5/model")
        >>> y_pred = model.predict(X_test)
        """
        self._warn_if_unhealthy("loading model")
        try:
            logger.info("Loading model from URI: %s", model_uri)
            model = self.mlflow.sklearn.load_model(model_uri, dst_path)
            logger.info("Model loaded successfully from: %s", model_uri)
            return model
        except FileNotFoundError as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Model file not found at {model_uri}",
                raise_as=CogflowArtifactError,
                re_raise=True,
            )
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Loading model from {model_uri}",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        await_registration_for: Optional[int] = None,
        *,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Register a model with the MLflow Model Registry.

        This method registers a trained model artifact (e.g., from a local path or run URI)
        into the MLflow Model Registry. It supports optional tagging and waiting for the
        registration process to complete.

        Args:
            model_uri (str): The URI of the model to register, e.g., ``"runs:/<run_id>/model"``.
            model_name (str): The name under which the model will be registered.
            await_registration_for (Optional[int]): Seconds to wait for registration completion.
                Defaults to ``config.AWAIT_REGISTRATION_FOR``.
            tags (Optional[Dict[str, Any]]): Optional metadata tags to organize registered models.
                Example: ``{"team": "mlops", "stage": "production"}``.

        Returns:
            mlflow.entities.model_registry.ModelVersion: The registered MLflow model version object.

        Raises:
            CogflowConnectionError: If MLflow tracking server is unreachable.
            CogflowModelError: If registration fails or the model URI is invalid.

        Example:
            >>> from cogflow import models
            >>> models.register_model(
            ...     model_uri="runs:/a1b2c3d4e5/model",
            ...     model_name="fraud-detection",
            ...     tags={"stage": "staging", "owner": "ml-team"}
            ... )
        """
        self._warn_if_unhealthy("registering model")

        await_registration_for = await_registration_for or config.AWAIT_REGISTRATION_FOR

        try:
            logger.info("Registering model '%s from URI: %s", model_name, model_uri)
            model_version = self.mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                await_registration_for=await_registration_for,
                tags=tags,
            )
            logger.info(
                "Successfully registered model %s (version=%s).",
                model_name,
                model_version.version,
            )
            return model_version

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Register model '{model_name}' from {model_uri}",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def create_registered_model(
        self,
        model: str,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        """
        Create a new registered model entry in the MLflow Model Registry.

        This method creates a *model namespace* (not a version) in the MLflow Model Registry,
        optionally with descriptive metadata and tags for categorization.

        Args:
            model (str): Name of the registered model to create.
            tags (Optional[Dict[str, Any]]): Optional dictionary of key-value tags.
                Example: ``{"team": "mlops", "env": "production"}``.
            description (Optional[str]): A human-readable description of the model’s purpose.

        Returns:
            mlflow.entities.model_registry.RegisteredModel: The created registered model object.

        Raises:
            CogflowConnectionError: If MLflow tracking server is unreachable.
            CogflowModelError: If model creation fails.

        Example:
            >>> from cogflow import models
            >>> models.create_registered_model(
            ...     model="sentiment-analyzer",
            ...     tags={"owner": "nlp-team"},
            ...     description="Model for detecting sentiment in user reviews."
            ... )
        """
        self._warn_if_unhealthy("creating registered model")

        try:
            logger.info("Creating registered model: %s", model)
            registered_model = self.client.create_registered_model(
                name=model, tags=tags, description=description
            )
            logger.info("Registered model %s created successfully.", model)
            return registered_model

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Create registered model '{model}'",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def create_model_version(
        self,
        model: str,
        source: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        run_link: Optional[str] = None,
        description: Optional[str] = None,
        await_creation_for: Optional[int] = None,
    ):
        """
        Create a new model version for an existing registered model.

        This method registers a new version of a model in MLflow’s Model Registry
        using the provided source URI (e.g., S3 or local artifact path). Metadata such as
        run ID, tags, and description can be attached to this version.

        Args:
            model (str): Name of the registered model.
            source (str): URI or path to the model artifact (e.g., "s3://bucket/model").
            run_id (Optional[str]): The MLflow run ID associated with this version.
            tags (Optional[Dict[str, Any]]): Optional dictionary of tags.
            run_link (Optional[str]): A URL or MLflow run link for traceability.
            description (Optional[str]): Description for this version.
            await_creation_for (Optional[int]): Timeout (in seconds) to wait for model
                version creation to complete. Defaults to ``config.AWAIT_REGISTRATION_FOR``.

        Returns:
            mlflow.entities.model_registry.ModelVersion: The created model version object.

        Raises:
            CogflowConnectionError: If MLflow tracking server is unreachable.
            CogflowModelError: If model version creation fails.

        Example:
            >>> from cogflow import models
            >>> models.create_model_version(
            ...     model="sentiment-analyzer",
            ...     source="s3://mlflow-artifacts/sentiment-model",
            ...     run_id="a1b2c3d4e5f6",
            ...     description="Version 2.0 retrained on 2025 data."
            ... )
        """
        self._warn_if_unhealthy("creating model version")

        await_creation_for = await_creation_for or config.AWAIT_REGISTRATION_FOR

        try:
            logger.info("Creating model version for %s from %s", model, source)
            version = self.client.create_model_version(
                name=model,
                source=source,
                run_id=common.uuid_to_hex(run_id) if run_id else None,
                tags=tags,
                run_link=run_link,
                description=description,
                await_creation_for=await_creation_for,
            )
            logger.info(
                "Model version %s created successfully for %s.",
                version.version,
                model,
            )
            return version

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Create model version for '{model}'",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def delete_registered_model(self, model_name: str) -> bool:
        """
        Delete a registered model from the MLflow registry.

        This operation removes the specified model from the MLflow Model Registry.
        Be cautious—this is irreversible and deletes all associated versions.

        Args:
            model_name (str): The name of the registered model to delete.

        Returns:
            bool: True if deletion succeeds, False otherwise.

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowModelError: If deletion fails for any other reason.

        Example:
            >>> from cogflow import models
            >>> models.delete_registered_model("customer_churn_model")
        """
        self._warn_if_unhealthy("deleting registered model")

        try:
            logger.info("Attempting to delete registered model: %s", model_name)
            self.client.delete_registered_model(model_name)
            logger.info("Successfully deleted registered model %s.", model_name)
            return True

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Delete registered model '{model_name}'",
                raise_as=CogflowModelError,
                re_raise=False,
            )
            return False

    def evaluate(
        self,
        model_uri,
        data,
        *,
        targets,
        model_type: str,
        dataset_path: Optional[str] = None,
        feature_names: Optional[list] = None,
        evaluators: Optional[list] = None,
        evaluator_config: Optional[dict] = None,
        custom_metrics: Optional[dict] = None,
        custom_artifacts: Optional[dict] = None,
        validation_thresholds: Optional[dict] = None,
        baseline_model=None,
        env_manager: Optional[str] = None,
    ) -> Any:
        """
        Evaluate a model and automatically log metrics & artifacts via CogFlow.

        This method extends MLflow's built-in `evaluate()` functionality to:
        - Log system CPU/memory usage
        - POST metrics and serialized artifacts to CogFlow backend endpoints
        - Support multiple evaluators and validation thresholds
        - Provide consistent error handling and health checking

        Args:
            model_uri: The ML model URI.
            data: Dataset used for evaluation (DataFrame, path, or dataset reference).
            targets: True target values corresponding to `data`.
            model_type (str): Model type (e.g., "regressor", "classifier").
            dataset_path (Optional[str]): Dataset location or identifier.
            feature_names (Optional[list]): Optional list of feature column names.
            evaluators (Optional[list]): Evaluators to use (e.g., ["default", "classification"]).
            evaluator_config (Optional[dict]): Evaluator-specific configuration overrides.
            custom_metrics (Optional[dict]): User-defined metrics to include.
            custom_artifacts (Optional[dict]): Custom artifacts to attach in evaluation output.
            validation_thresholds (Optional[dict]): Thresholds for metric validation.
            baseline_model (Any): Optional baseline model for comparative evaluation.
            env_manager (Optional[str]): Environment manager; defaults to `config.ENV_MANAGER`.

        Returns:
            Any: The evaluation result object containing metrics and artifacts.

        Raises:
            CogflowConnectionError: If MLflow tracking or backend API endpoints are unreachable.
            CogflowModelError: If model evaluation fails.
            CogflowValidationError: If result serialization or posting fails.

        Example:
            >>> from cogflow import models
            >>> result = models.evaluate(
            ...     model_uri="runs:/abc123/model",
            ...     data=test_df,
            ...     targets="label",
            ...     model_type="classifier",
            ...     evaluators=["default"],
            ... )
            >>> print(result.metrics)
        """

        self._warn_if_unhealthy("evaluating model")

        env_manager = env_manager or config.ENV_MANAGER

        try:
            logger.info("Starting model evaluation for type %s.", model_type)

            # Evaluate model using MLflow
            result = self.mlflow.evaluate(
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

            logger.info("MLflow model evaluation completed successfully.")

            # # Capture system metrics
            # cpu_usage = psutil.cpu_percent(interval=1)
            # memory_info = psutil.virtual_memory()
            # memory_used_mb = round(memory_info.used / (1024**2), 2)
            #
            # # Append resource metrics
            # metrics = dict(result.metrics)
            # metrics.update(
            #     {
            #         "cpu_consumption": cpu_usage,
            #         "memory_utilization_mb": memory_used_mb,
            #     }
            # )
            #
            # # Prepare API URLs for CogFlow
            # Construct URLs
            #     if model_uri.startswith("runs:/"):
            #         run_id = model_uri.split("/")[1]
            #     elif model_uri.startswith("s3://"):
            #         run_id = model_uri.split("/")[4]
            #     else:
            #         raise ValueError(
            #             f"Unsupported model_uri format. Expected 'runs:/' or 's3://', got: {model_uri}"
            #         )
            # base_url = config.API_PATH
            # model_id = network.uuid_to_canonical(run_id)
            # url_metrics = f"{base_url}/models/{model_id}{config.VALIDATION_METRICS}"
            # url_artifacts = f"{base_url}/models/{model_id}{config.VALIDATION_ARTIFACTS}"
            #
            # # Serialize artifacts
            # serialized_artifacts = network.serialize_artifacts(result.artifacts)
            #
            # # Prepare headers
            # headers = {
            #     "Content-Type": "application/json",
            #     "kubeflow-userid": common.get_current_user(),
            # }
            #
            # # POST metrics
            # try:
            #     network.make_post_request(
            #         url_metrics,
            #         json=metrics,
            #         headers=headers,
            #         timeout=config.DEFAULT_TIMEOUT,
            #     )
            #     logger.info("Metrics successfully posted to %s", url_metrics)
            # except Exception as e:
            #     CogflowErrorHandler.handle_exception(
            #         e,
            #         context="Posting evaluation metrics to CogFlow API",
            #         raise_as=CogflowConnectionError,
            #         re_raise=False,
            #     )
            #
            # # POST artifacts
            # try:
            #     network.make_post_request(
            #         url_artifacts,
            #         json=serialized_artifacts,
            #         headers=headers,
            #         timeout=config.DEFAULT_TIMEOUT,
            #     )
            #     logger.info("Artifacts successfully posted to %s", url_artifacts)
            # except Exception as e:
            #     CogflowErrorHandler.handle_exception(
            #         e,
            #         context="Posting evaluation artifacts to CogFlow API",
            #         raise_as=CogflowArtifactError,
            #         re_raise=False,
            #     )

            # Return result
            return result

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Evaluating model (type={model_type})",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        conda_env: Optional[str] = None,
        code_paths: Optional[List[str]] = None,
        serialization_format: str = config.SERIALIZATION_FORMAT,
        signature: Optional["ModelSignature"] = None,
        input_example: Optional[
            Union[
                "pd.DataFrame",
                "np.ndarray",
                dict,
                list,
                "csr_matrix",
                "csc_matrix",
                str,
                bytes,
                tuple,
            ]
        ] = None,
        await_registration_for: int = config.AWAIT_REGISTRATION_FOR,
        pip_requirements: Optional[str] = None,
        extra_pip_requirements: Optional[str] = None,
        pyfunc_predict_fn: str = config.PYFUNC_PREDICT_FN,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Log a model to MLflow and optionally register it in the CogFlow system.

        This method handles both:
            - Standard models (e.g., scikit-learn, xgboost, etc.)
            - Custom Python models using the PyFunc flavor

        It logs the model into MLflow, detects its type, and posts metadata
        (user, version, run_id, etc.) back to CogFlow’s backend.

        Args:
            model: Trained model object to be logged.
            artifact_path (str): Artifact path where model artifacts are stored.
            registered_model_name (Optional[str]): Optional registered model name.
            conda_env (Optional[str]): Path to Conda environment YAML file.
            code_paths (Optional[List[str]]): List of Python source paths to include.
            serialization_format (str): Serialization format (default: "cloudpickle").
            signature (Optional[ModelSignature]): Schema of model inputs/outputs.
            input_example (Optional[Any]): Example input for schema inference.
            await_registration_for (int): Timeout for registration completion.
            pip_requirements (Optional[str]): Path to pip requirements file.
            extra_pip_requirements (Optional[str]): Additional pip dependencies.
            pyfunc_predict_fn (str): Name of the prediction function (default: "predict").
            metadata (Optional[Dict[str, Any]]): Custom metadata to attach to the model.

        Returns:
            Any: The MLflow model info object from the logging operation.

        Raises:
            CogflowModelError: If model logging or registration fails.

        Example:
            >>> from cogflow import models
            >>> from sklearn.linear_model import LogisticRegression
            >>> model = LogisticRegression().fit(X_train, y_train)
            >>> models.log_model(
            ...     model,
            ...     artifact_path="sklearn-model",
            ...     registered_model_name="my-logreg-model",
            ...     input_example=X_train[:5]
            ... )

        """
        self._warn_if_unhealthy("logging model")
        if signature is not None and not isinstance(signature, self.modelsignature):
            raise TypeError("signature must be a ModelSignature instance")

        if input_example is not None:
            valid_types = (
                self.pd.DataFrame,
                self.np.ndarray,
                dict,
                list,
                self.csr_matrix,
                self.csc_matrix,
                str,
                bytes,
                tuple,
            )
            if not isinstance(input_example, valid_types):
                raise TypeError(
                    "input_example must be one of: pd.DataFrame, np.ndarray, dict, list, csr_matrix, "
                    "csc_matrix, str, bytes, tuple"
                )

        try:
            # --- Safe detection logic (no torch / sklearn import required) ---
            cls_hierarchy = [cls.__name__.lower() for cls in type(model).mro()]

            is_pyfunc = isinstance(model, self.pyfunc.PythonModel) or (
                self.inspect.isclass(model)
                and issubclass(model, self.pyfunc.PythonModel)
            )
            is_pytorch = "module" in cls_hierarchy  # torch.nn.Module base class
            is_sklearn = (
                "baseestimator" in cls_hierarchy
            )  # sklearn.base.BaseEstimator base class
            # ---------------------------------------------------------------

            if is_pyfunc:
                logger.info(
                    "Logging custom PyFunc model using MLflow.pyfunc.log_model()"
                )
                result = self.mlflow.pyfunc.log_model(
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
                logger.info("Logging PyTorch model using MLflow.pytorch.log_model()")
                result = self.pytorch.log_model(
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
                logger.info(
                    "Logging scikit-learn model using MLflow.sklearn.log_model()"
                )
                result = self.sklearn.log_model(
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

            logger.info("Model successfully logged to MLflow.")

            # ----------------------------------------------------------
            # Post model metadata to CogFlow backend (if possible)
            # ----------------------------------------------------------
            try:
                active_run = self.mlflow.active_run()
                if not active_run:
                    raise RuntimeError("No active MLflow run found")

                model_id = active_run.info.run_id
                model_details = self.get_full_model_uri_from_run_or_registry(
                    model_id=model_id
                )
                model_type = self.detect_model_type(model_details["model_uri"])

                model_dict = {
                    "model_id": common.normalize_uuid(model_id),
                    "model_name": str(
                        model_details.get("model_name")
                        or active_run.data.tags.get("mlflow.runName", "UnnamedModel")
                    ),
                    "model_version": int(model_details.get("model_version") or 0),
                    "register_date": datetime.fromtimestamp(
                        active_run.info.start_time / 1000
                    ).isoformat(),
                    "type": model_type,
                    "description": str(
                        active_run.data.tags.get("mlflow.note.content") or "log_model"
                    ),
                    "user_id": common.get_current_user(),
                }

                url = f"{config.API_PATH}{config.LOG_MODEL}"

                headers = {"kubeflow-userid": model_dict["user_id"]}

                network.make_post_request(url=url, data=model_dict, headers=headers)
                logger.info("Logged model metadata to CogFlow backend: %s", url)

            except Exception as post_err:
                logger.warning(
                    "Failed to post model metadata to CogFlow backend: %s",
                    str(post_err),
                )

            return result

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="Logging MLflow model",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def _prepare_llm_catalog_run(
        self,
        *,
        served_model_name: str,
        hf_model_id: Optional[str],
        extra_tags: Optional[Dict[str, str]],
    ):
        """Open the MLflow run that anchors an LLM catalog entry.

        Shared step between the sync and async registration paths
        (:meth:`register_llm_catalog_entry` and
        :meth:`async_register_llm_catalog_entry`). Everything here is
        sync — MLflow's tracking client is sync, but its target is a
        separate service (MLflow server), so it blocks the event loop
        only briefly and does not self-deadlock.

        Returns:
            Tuple of ``(run_id, start_time_ms, description,
            normalized_hf_model_id)``.
        """
        self._warn_if_unhealthy("registering LLM catalog entry")

        # Self-defensive normalization: ``serve_llm`` already runs the
        # id through this validator, but direct callers (notebook users
        # of ``cogflow.models.register_llm_catalog_entry``) shouldn't
        # have to. Strip an accidental ``hf://`` prefix first, then run
        # through the same ``_extract_hf_model_id`` helper the serving
        # path uses so whitespace / stray slashes / empty input are
        # rejected up-front — no orphan MLflow run, no orphan catalog
        # row on bad input.
        if hf_model_id is not None:
            if hf_model_id.startswith("hf://"):
                hf_model_id = hf_model_id[len("hf://") :]
            from cogflow.core.serving import ServingManager as _ServingManager

            hf_model_id = _ServingManager._extract_hf_model_id(
                f"hf://{hf_model_id}"
            )

        description = (
            f"LLM served from HuggingFace: {hf_model_id}"
            if hf_model_id
            else "LLM served from MLflow-backed checkpoint"
        )

        try:
            with self.start_run(run_name=f"register-{served_model_name}") as run_info:
                # Apply caller ``extra_tags`` FIRST so the reserved
                # identity tags below always win on a collision — catalog
                # consumers and downstream tooling rely on them being
                # authoritative. A caller passing
                # ``extra_tags={"type": "foo"}`` must not be able to
                # desync the run from the catalog entry.
                if extra_tags:
                    for k, v in extra_tags.items():
                        self.set_tag(k, v)
                self.set_tag("type", "llm")
                self.set_tag("source", "huggingface" if hf_model_id else "mlflow")
                if hf_model_id:
                    self.set_tag("hf_model_id", hf_model_id)
                self.set_tag("mlflow.note.content", description)
            run_id = run_info.info.run_id
            start_time_ms = run_info.info.start_time
            logger.info(
                "Opened MLflow run %s for LLM '%s' (hf_model_id=%s)",
                run_id,
                served_model_name,
                hf_model_id,
            )
            return run_id, start_time_ms, description, hf_model_id
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Register LLM catalog entry '{served_model_name}'",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def _build_llm_catalog_payload(
        self,
        *,
        run_id: str,
        start_time_ms: int,
        served_model_name: str,
        hf_model_id: Optional[str],
        description: str,
        user_id: Optional[str],
    ):
        """Shape the ``POST /models/log`` payload for an LLM catalog entry.

        Returns ``(url, model_dict, headers, resolved_user)`` so both
        the sync and async registration methods can use the same body
        without duplicating the resolution + dict-shaping logic.
        """
        resolved_user = user_id or common.get_current_user()
        model_dict: Dict[str, Any] = {
            "model_id": common.normalize_uuid(run_id),
            "model_name": served_model_name,
            # HF-sourced LLMs aren't MLflow-registered, so there's no
            # registered-model version number to report. Use 0 as
            # the sentinel for "unknown registry version" — matches
            # the fallback ``log_model`` already uses for classical
            # artifacts when ``model_details.get("model_version")``
            # is missing or falsy.
            "model_version": 0,
            "register_date": datetime.fromtimestamp(
                start_time_ms / 1000
            ).isoformat(),
            "type": "llm",
            "description": description,
            "user_id": resolved_user,
        }
        if hf_model_id:
            # Old catalog schemas (<= the ModelLogBase that predates
            # this field) ignore unknown keys; newer ones will persist
            # the hf_model_id column. Forward-compatible.
            model_dict["hf_model_id"] = hf_model_id

        url = f"{config.API_PATH}{config.LOG_MODEL}"
        headers = {"kubeflow-userid": resolved_user}
        return url, model_dict, headers, resolved_user

    def register_llm_catalog_entry(
        self,
        *,
        served_model_name: str,
        hf_model_id: Optional[str] = None,
        user_id: Optional[str] = None,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create an MLflow run for an LLM and register its catalog entry
        with the CogFlow backend.

        HF-sourced LLMs have no MLflow artifact to log; we open a run
        purely to establish the catalog identity (``model_info.id ==
        MLflow run_id``, the same invariant every other registration
        path respects) and to carry metadata tags.

        Mirrors the ``log_model`` → ``/models/log`` pattern:

        - Opens and closes an MLflow run; sets ``type=llm``,
          ``source=huggingface`` (or ``mlflow`` if no HF id is given),
          ``hf_model_id``, ``mlflow.note.content``, plus any caller-
          supplied ``extra_tags``.
        - Best-effort POST to ``{API_PATH}{LOG_MODEL}``: failures are
          logged as warnings and do **not** raise — matches
          ``log_model``'s behaviour so a transient CogFlow backend
          outage can't abort an LLM deploy that otherwise succeeded.

        For callers already inside an async coroutine, prefer
        :meth:`async_register_llm_catalog_entry` — the POST here is
        blocking ``requests.post`` and will deadlock the event loop if
        the target URL resolves back to the same uvicorn worker.

        Args:
            served_model_name: Logical model name (vLLM ``--model_name``).
                Also becomes the catalog row's ``name``.
            hf_model_id: HuggingFace Hub id (e.g. ``"Qwen/Qwen2.5-Coder-7B-Instruct"``),
                or ``None`` for MLflow-backed LLMs.
            user_id: Override for the ``kubeflow-userid`` / catalog
                ``register_user_id``. Falls back to
                ``common.get_current_user()`` (reads the Kubeflow
                namespace's ``owner`` annotation) which is what notebook
                consumers want.
            extra_tags: Additional MLflow tags to set on the run.

        Returns:
            The MLflow ``run_id`` — which is also the catalog row's
            primary key.

        Raises:
            CogflowModelError: If opening the MLflow run itself fails.
                Backend POST failures are *not* raised (warn-only).
        """
        run_id, start_time_ms, description, hf_model_id = (
            self._prepare_llm_catalog_run(
                served_model_name=served_model_name,
                hf_model_id=hf_model_id,
                extra_tags=extra_tags,
            )
        )

        url, model_dict, headers, _ = self._build_llm_catalog_payload(
            run_id=run_id,
            start_time_ms=start_time_ms,
            served_model_name=served_model_name,
            hf_model_id=hf_model_id,
            description=description,
            user_id=user_id,
        )

        # Best-effort POST — same pattern as log_model. If the catalog
        # service is unreachable we still want the deploy to proceed;
        # the warning surfaces in logs.
        try:
            network.make_post_request(url=url, data=model_dict, headers=headers)
            logger.info(
                "Registered LLM catalog entry via CogFlow backend: %s (run_id=%s)",
                url,
                run_id,
            )
        except Exception as post_err:
            logger.warning(
                "Failed to post LLM catalog entry to CogFlow backend "
                "(deploy proceeds regardless): %s",
                str(post_err),
            )

        return run_id

    async def async_register_llm_catalog_entry(
        self,
        *,
        served_model_name: str,
        hf_model_id: Optional[str] = None,
        user_id: Optional[str] = None,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Async variant of :meth:`register_llm_catalog_entry`.

        Identical semantics, but the backend POST to ``{API_PATH}
        {LOG_MODEL}`` uses :func:`network.make_async_post_request`
        (httpx-backed) so the event loop stays free during the call.
        This is what lets ``cog-api -> cogflow -> cog-api`` self-calls
        terminate — when the POST target is the same uvicorn worker
        that's running the outer coroutine, a sync ``requests.post``
        deadlocks the loop, and the callback can't be accepted.

        MLflow's tracking client is still sync (no first-party async
        MLflow yet), so the run-open/close step blocks the loop
        briefly. That's fine in practice: MLflow server is a separate
        service, so the block never deadlocks — it just delays other
        coroutines on the same worker for the duration of the run
        setup/teardown.

        Return value and exception semantics match the sync variant.
        """
        run_id, start_time_ms, description, hf_model_id = (
            self._prepare_llm_catalog_run(
                served_model_name=served_model_name,
                hf_model_id=hf_model_id,
                extra_tags=extra_tags,
            )
        )

        url, model_dict, headers, _ = self._build_llm_catalog_payload(
            run_id=run_id,
            start_time_ms=start_time_ms,
            served_model_name=served_model_name,
            hf_model_id=hf_model_id,
            description=description,
            user_id=user_id,
        )

        try:
            await network.make_async_post_request(
                url=url, data=model_dict, headers=headers
            )
            logger.info(
                "Registered LLM catalog entry via CogFlow backend: %s (run_id=%s)",
                url,
                run_id,
            )
        except Exception as post_err:
            logger.warning(
                "Failed to post LLM catalog entry to CogFlow backend "
                "(deploy proceeds regardless): %s",
                str(post_err),
            )

        return run_id

    def search_model_versions(self, filter_string: Optional[str] = None):
        """
        Search for model versions in the MLflow Model Registry.

        This method retrieves model version entries that match the specified filter.
        It can search by model name, version, or other metadata attributes.

        Args:
            filter_string (Optional[str]): A SQL-like query string used to filter model versions.
                Examples:
                    - ``"name='my-model'"``

        Returns:
            List[mlflow.entities.model_registry.ModelVersion]: A list of MLflow ModelVersion objects.

        Raises:
            CogflowConnectionError: If MLflow tracking server is unreachable.
            CogflowModelError: If the model search fails.

        Example:
            >>> from cogflow import models
            >>> results = models.search_model_versions("name='sentiment-analyzer'")
            >>> for mv in results:
            ...     print(mv.name, mv.version)
            sentiment-analyzer 1
            sentiment-analyzer 2
        """
        self._warn_if_unhealthy("searching model versions")

        try:
            logger.info("Searching for model versions with filter: %s", filter_string)
            results = self.client.search_model_versions(filter_string=filter_string)
            logger.info("Found %s model version(s).", len(results))
            return results

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Search model versions (filter={filter_string})",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def get_model_uri(self, model_name: str, version: str) -> str:
        """
        Retrieve the model URI for a specific registered model version.

        This method fetches the source URI (e.g., S3, GCS, or local path)
        where the specified model version is stored in the MLflow Model Registry.

        Args:
            model_name (str): Name of the registered model.
            version (str): Version identifier of the model.

        Returns:
            str: The model's artifact URI (e.g., "s3://mlflow-artifacts/.../model").

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowModelError: If model retrieval fails or the version does not exist.

        Example:
            >>> from cogflow import models
            >>> uri = models.get_model_uri("sentiment-analyzer", "3")
            >>> print(uri)
            s3://mlflow-artifacts/sentiment-analyzer/3/model
        """
        self._warn_if_unhealthy("fetching model URI")

        try:
            logger.info("Fetching model URI for %s (version %s)", model_name, version)
            mv = self.client.get_model_version(name=model_name, version=version)
            uri = mv.source
            logger.debug("Resolved model URI: %s", uri)
            return uri

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Get model URI for '{model_name}' (version {version})",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def get_full_model_uri_from_run_or_registry(
        self,
        model_id: Optional[str] = None,
        artifact_path: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Resolve the full model URI from either a run ID or a registered model name/version.

        This method supports dual resolution:
        - If a `run_id` (`model_id`) is provided, it resolves the associated model artifact path.
        - If `model_name` and `model_version` are provided, it finds the corresponding run
          and resolves the full model artifact URI.

        Args:
            model_id (Optional[str]): MLflow run ID associated with the model.
            artifact_path (Optional[str]): Optional artifact subpath, e.g., "model".
            model_name (Optional[str]): Name of the registered model.
            model_version (Optional[str]): Version of the registered model.

        Returns:
            Dict[str, str]: Dictionary with resolved details:
                {
                    "model_uri": str,
                    "model_name": str,
                    "model_version": str,
                    "model_id": str
                }

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowModelError: If model artifacts or registry metadata cannot be resolved.
            CogflowValidationError: If neither run ID nor name/version are provided.

        Example:
            # 1. Resolve by run ID only
            >>> from cogflow import models
            >>> info = models.get_full_model_uri_from_run_or_registry(model_id="a1b2c3d4e5")
            >>> print(info["model_uri"])
            s3://mlflow-artifacts/1234/a1b2c3d4e5/artifacts/model

            # 2. Resolve by run ID and custom artifact path
            >>> info = models.get_full_model_uri_from_run_or_registry(
            ...     model_id="a1b2c3d4e5", artifact_path="custom_model"
            ... )
            >>> print(info["model_uri"])
            s3://mlflow-artifacts/1234/a1b2c3d4e5/artifacts/custom_model

            # 3. Resolve by registered model name and version
            >>> info = models.get_full_model_uri_from_run_or_registry(
            ...     model_name="sentiment-analyzer", model_version="3"
            ... )
            >>> print(info["model_uri"])
            s3://mlflow-artifacts/sentiment-analyzer/3/model

            # 4. Resolve by registered model name, version, and artifact path
            >>> info = models.get_full_model_uri_from_run_or_registry(
            ...     model_name="sentiment-analyzer", model_version="3", artifact_path="subdir/model"
            ... )
            >>> print(info["model_uri"])
            s3://mlflow-artifacts/sentiment-analyzer/3/model/subdir/model

        """
        if not (model_id or (model_name and model_version)):
            CogflowErrorHandler.log_and_raise(
                "Either `model_id` or both `model_name` and `model_version` must be provided.",
                raise_as=CogflowValidationError,
            )

        self._warn_if_unhealthy("resolving model URI")

        try:
            client = self.client

            # Normalize model_id to MLflow internal hex form
            if model_id:
                try:
                    model_id = common.uuid_to_hex(model_id)
                    logger.debug("Converted model_id to hex format: %s", model_id)
                except Exception as conv_err:
                    logger.warning("Failed to convert model_id to hex: %s", conv_err)

            # Resolve run_id from registry if model_name/version provided
            if not model_id and model_name and model_version:
                mv = client.get_model_version(
                    name=model_name, version=str(model_version)
                )
                model_id = mv.run_id

            run = self.mlflow.get_run(model_id)
            artifact_uri = run.info.artifact_uri

            if artifact_path:
                model_uri = f"{artifact_uri.rstrip('/')}/{artifact_path.lstrip('/')}"
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
                    raise CogflowModelError(
                        f"Multiple artifact directories found for run '{model_id}'. "
                        "Please specify `artifact_path` explicitly."
                    )
                elif model_files:
                    model_uri = f"{artifact_uri}/{model_files[0].path}"
                else:
                    raise CogflowModelError(
                        f"No valid model artifact found for run '{model_id}'. "
                        "Specify `artifact_path` explicitly."
                    )

            # Backfill name/version if not provided
            if not model_name or not model_version:
                results = self.search_model_versions(
                    filter_string=f"run_id = '{model_id}'"
                )
                if results:
                    model_name = results[0].name
                    model_version = results[0].version

            logger.info(
                "Resolved model URI for %s: %s",
                model_name or model_id,
                model_uri,
            )
            return {
                "model_uri": model_uri,
                "model_name": model_name,
                "model_version": model_version,
                "model_id": model_id,
            }
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Resolve model URI for {model_name or model_id}",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def detect_model_format(self, model_uri: str) -> str:
        """
        Detect the model format (flavor) from an MLflow model URI.

        Determines the model's saved format by inspecting its registered flavors
        (e.g., sklearn, python_function, pytorch, etc.).

        Args:
            model_uri (str): URI or path to the MLflow model.

        Returns:
            str: Detected model format.
                - "mlflow" for generic MLflow pyfunc models.
                - "sklearn" for scikit-learn models.
                - "unknown" if undetectable.

        Raises:
            CogflowModelError: If the model information cannot be retrieved.

        Example:
            >>> from cogflow import models
            >>> fmt = models.detect_model_format("models:/sentiment-analyzer/3")
            >>> print(fmt)
            sklearn
        """
        self._warn_if_unhealthy("detecting model format")

        try:
            model_info = self.mlflow.models.get_model_info(model_uri)
            flavors = model_info.flavors.keys()
            if "sklearn" in flavors:
                return "sklearn"
            if "python_function" in flavors:
                return "mlflow"
            return "unknown"

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Detect model format for '{model_uri}'",
                raise_as=CogflowModelError,
                re_raise=True,
            )

    def detect_model_type(self, model_uri: str) -> str:
        """
        Detect the model type (flavor) from an MLflow model URI.

        Inspects model metadata to determine which MLflow flavor
        was used when saving the model.

        Args:
            model_uri (str): Path or URI to the MLflow model.

        Returns:
            str: Detected model type.
                - "pyfunc" for MLflow’s generic Python function interface.
                - "sklearn" for scikit-learn models.
                - "unknown" otherwise.

        Raises:
            CogflowModelError: If the model info cannot be retrieved.

        Example:
            >>> from cogflow import models
            >>> mtype = models.detect_model_type("models:/sentiment-analyzer/3")
            >>> print(mtype)
            pyfunc
        """
        self._warn_if_unhealthy("detecting model type")

        try:
            model_info = self.mlflow.models.get_model_info(model_uri)
            flavors = model_info.flavors.keys()
            if "sklearn" in flavors:
                return "sklearn"
            if "python_function" in flavors:
                return "pyfunc"
            return "unknown"

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Detect model type for '{model_uri}'",
                raise_as=CogflowModelError,
                re_raise=True,
            )

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
        Start or resume an MLflow run.

        This method starts a new MLflow run or resumes an existing one.
        You can associate the run with a specific experiment and optionally set
        tags and a description. Supports nested runs for hierarchical tracking.

        Args:
            run_id (Optional[str]): The ID of the run to resume. If None, a new run is created.
            experiment_id (Optional[str]): The ID of the experiment to associate this run with.
            run_name (Optional[str]): A friendly name for the run.
            nested (bool): Whether to create this run as a nested run under the current run.
            tags (Optional[Dict[str, Any]]): Optional metadata tags.
            description (Optional[str]): Optional textual description for this run.

        Returns:
            mlflow.entities.Run: The MLflow run object representing the started or resumed run.

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If the run creation or resumption fails.

        Example:
            >>> from cogflow import models
            >>> run = models.start_run(run_name="nlp_training", tags={"stage": "training"})
            >>> print(run.info.run_id)
            6f71b1a2f9ab47f2b...
        """
        self._warn_if_unhealthy("starting MLflow run")

        try:
            logger.info(
                "Starting MLflow run: name=%s, experiment_id=%s",
                run_name,
                experiment_id,
            )
            run = self.mlflow.start_run(
                run_id=common.uuid_to_hex(run_id) if run_id else None,
                experiment_id=experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags,
                description=description,
            )
            logger.info("MLflow run started successfully (run_id=%s).", run.info.run_id)
            return run

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Start MLflow run (name={run_name})",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def end_run(self):
        """
        End the currently active MLflow run.

        This finalizes metrics, parameters, and artifacts logged under the current run.
        It should be called after training or evaluation completes.

        Returns:
            mlflow.entities.Run: The MLflow Run object corresponding to the ended run.

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If ending the run fails or no run is active.

        Example:
            >>> from cogflow import models
            >>> models.end_run()
        """
        self._warn_if_unhealthy("ending MLflow run")

        try:
            logger.info("Ending the current MLflow run...")
            run = self.mlflow.end_run()
            logger.info("MLflow run ended successfully.")
            return run

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="End MLflow run",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def set_experiment(
        self,
        experiment_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        """
        Set the active MLflow experiment.

        Determines which experiment subsequent runs will log to.
        You can specify either an experiment name or ID.

        Args:
            experiment_name (Optional[str]): The name of the experiment to set as active.
            experiment_id (Optional[str]): The ID of the experiment (if name not provided).

        Returns:
            None

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowExperimentError: If setting the experiment fails.

        Example:
            >>> from cogflow import models
            >>> models.set_experiment(experiment_name="customer_churn")
        """
        self._warn_if_unhealthy("setting experiment")

        try:
            logger.info(
                "Setting experiment: name=%s, id=%s", experiment_name, experiment_id
            )
            self.mlflow.set_experiment(
                experiment_name=experiment_name, experiment_id=experiment_id
            )
            logger.info("Active experiment set successfully.")
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Set experiment (name={experiment_name}, id={experiment_id})",
                raise_as=CogflowExperimentError,
                re_raise=True,
            )

    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a tag under the current MLflow run.

        Tags are key-value pairs that help annotate and organize runs.

        Args:
            key (str): The tag name.
            value (Any): The tag value (stringified if not already a string).

        Returns:
            None

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If tagging fails or no active run exists.

        Example:
            >>> from cogflow import models
            >>> models.set_tag("team", "mlops")
        """
        self._warn_if_unhealthy("setting tag")

        try:
            logger.debug("Setting tag %s=%s", key, value)
            self.mlflow.set_tag(key=key, value=value)
            logger.info("Tag %s set successfully.", key)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Set tag '{key}'",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def get_experiment_id_from_run(self, run_id: str) -> str:
        """
        Fetch the experiment ID associated with a given MLflow run ID.

        Args:
            run_id (str): The unique MLflow run identifier.

        Returns:
            str: The experiment ID associated with the run.

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If the run cannot be found or retrieval fails.

        Example:
            >>> from cogflow import models
            >>> exp_id = models.get_experiment_id_from_run("a1b2c3d4e5")
            >>> print(exp_id)
            12
        """
        self._warn_if_unhealthy("fetching experiment ID from run")

        try:
            run_id = common.uuid_to_hex(run_id)
            logger.debug("Converted run_id to hex format: %s", run_id)
            run = self.mlflow.get_run(run_id)
            exp_id = run.info.experiment_id
            logger.debug("Resolved experiment ID %s for run %s", exp_id, run_id)
            return exp_id

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Get experiment ID from run {run_id}",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def get_run(self, run_id: str):
        """
        Fetch an MLflow run by its ID.

        Args:
            run_id (str): The run identifier (UUID, with or without hyphens).

        Returns:
            mlflow.entities.Run: The MLflow Run object containing info, data
            (params, metrics, tags), and artifact URI.

        Raises:
            CogflowRunError: If the run cannot be found or retrieval fails.

        Example:
            >>> from cogflow import models
            >>> run = models.get_run("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
            >>> print(run.info.artifact_uri)
            >>> print(run.data.params)
        """
        self._warn_if_unhealthy("fetching run")

        try:
            run_id = common.uuid_to_hex(run_id)
            run = self.mlflow.get_run(run_id)
            logger.debug("Fetched run %s", run_id)
            return run
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Get run {run_id}",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def get_run_artifact_uri(self, run_id: str) -> str:
        """
        Get the S3 artifact URI for a specific run.

        Args:
            run_id (str): The run identifier (UUID, with or without hyphens).

        Returns:
            str: The artifact URI (e.g., "s3://mlflow/0/abc123/artifacts").

        Raises:
            CogflowRunError: If the run cannot be found or retrieval fails.

        Example:
            >>> from cogflow import models
            >>> uri = models.get_run_artifact_uri("a1b2c3d4e5f67890abcdef1234567890")
            >>> print(uri)
            s3://mlflow/0/a1b2c3d4e5f67890abcdef1234567890/artifacts
        """
        run = self.get_run(run_id)
        return run.info.artifact_uri

    def list_artifacts_grouped(self, run_id: str) -> Dict[str, List[str]]:
        """
        List artifacts for a run, grouped by directory.

        Args:
            run_id (str): The run identifier (UUID, with or without hyphens).

        Returns:
            Dict[str, List[str]]: A dictionary mapping directory names to lists
            of filenames. Root-level files use "" as the key.

        Raises:
            CogflowRunError: If the run cannot be found or retrieval fails.

        Example:
            >>> from cogflow import models
            >>> artifacts = models.list_artifacts_grouped("a1b2c3d4e5")
            >>> print(artifacts)
            {"": ["README.md"], "model": ["model.pkl", "config.json"]}
        """
        self._warn_if_unhealthy("listing artifacts")

        try:
            run_id = common.uuid_to_hex(run_id)
            artifacts = self.client.list_artifacts(run_id)
            grouped: Dict[str, List[str]] = {}

            for artifact in artifacts:
                if artifact.is_dir:
                    # List files inside the directory
                    sub_artifacts = self.client.list_artifacts(run_id, artifact.path)
                    grouped[artifact.path] = [
                        a.path.split("/")[-1]
                        for a in sub_artifacts
                        if not a.is_dir
                    ]
                else:
                    grouped.setdefault("", []).append(artifact.path)

            return grouped
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"List artifacts for run {run_id}",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str = "",
        run_view_type: Optional[int] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
        page_token: Optional[str] = None,
    ):
        """
        Search for MLflow runs based on specified criteria.

        Args:
            experiment_ids (List[str]): List of experiment IDs to search within.
            filter_string (str): Optional filter query (SQL-like syntax).
                Example: ``"metrics.rmse < 0.1"``.
            run_view_type (Optional[int]): Run visibility (active, deleted, or all).
            max_results (int): Maximum number of runs to return (default: 100).
            order_by (Optional[List[str]]): Sorting order, e.g., ["metrics.rmse DESC"].
            page_token (Optional[str]): Token for pagination.

        Returns:
            mlflow.entities.PagedList[Run]: A paginated list of MLflow run objects.

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If run search fails.

        Example:
            >>> from cogflow import models
            >>> runs = models.search_runs(
            ...     experiment_ids=["5"],
            ...     filter_string="metrics.accuracy > 0.9",
            ...     order_by=["metrics.accuracy DESC"]
            ... )
            >>> for r in runs:
            ...     print(r.info.run_id, r.data.metrics)
        """
        self._warn_if_unhealthy("searching runs")

        try:
            logger.info("Searching MLflow runs with filter: %s", filter_string)
            results = self.client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                run_view_type=run_view_type
                or self.mlflow.entities.ViewType.ACTIVE_ONLY,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )
            logger.info("Found %s matching runs.", len(results))
            return results

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Search runs (filter={filter_string})",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter to the active MLflow run.

        Parameters represent fixed configuration values for a run, such as
        hyperparameters or model architecture settings.

        Args:
            key (str): Parameter name.
            value (Any): Parameter value (converted to string if not already).

        Returns:
            None

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If logging fails or no active run exists.

        Example:
            >>> from cogflow import models
            >>> models.log_param("learning_rate", 0.001)
        """
        self._warn_if_unhealthy("logging parameter")

        try:
            logger.debug("Logging parameter: %s=%s", key, value)
            self.mlflow.log_param(key, value)
            logger.info("Parameter %s logged successfully.", key)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Log parameter '{key}'",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters in a single batch to the active MLflow run.

        Args:
            params (Dict[str, Any]): A dictionary of parameter name–value pairs.
                Example: ``{"lr": 0.01, "epochs": 100, "optimizer": "adam"}``

        Returns:
            None

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If logging fails or no run is active.

        Example:
            >>> from cogflow import models
            >>> models.log_params({"lr": 0.01, "batch_size": 32, "dropout": 0.2})
        """
        self._warn_if_unhealthy("logging multiple parameters")

        try:
            logger.debug("Logging multiple parameters: %s", params)
            self.mlflow.log_params(params)
            logger.info("%s parameters logged successfully.", len(params))
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="Log multiple parameters",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric to the current MLflow run.

        Metrics represent quantitative performance indicators (e.g., accuracy, loss).
        They can be logged across training steps or epochs.

        Args:
            key (str): Name of the metric (e.g., "accuracy").
            value (float): Value of the metric.
            step (Optional[int]): Step or iteration index. Defaults to 0 if not provided.

        Returns:
            None

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If logging fails or no active run exists.

        Example:
            >>> from cogflow import models
            >>> models.log_metric("accuracy", 0.94, step=10)
        """
        self._warn_if_unhealthy("logging metric")

        try:
            logger.debug("Logging metric: %s=%s, step=%s", key, value, step)
            self.mlflow.log_metric(key, value, step=step)
            logger.info("Metric %s logged successfully at step %s.", key, step)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Log metric '{key}'",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log multiple metrics at once for the current MLflow run.

        Args:
            metrics (Dict[str, float]): A dictionary mapping metric names to values.
                Example: ``{"mse": 2500.0, "rmse": 50.0}``
            step (Optional[int]): Step or iteration index for all metrics.

        Returns:
            None

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowRunError: If metric logging fails.

        Example:
            >>> from cogflow import models
            >>> models.log_metrics({"loss": 0.1, "val_loss": 0.08}, step=3)
        """
        self._warn_if_unhealthy("logging multiple metrics")

        try:
            logger.debug("Logging multiple metrics: %s, step=%s", metrics, step)
            self.mlflow.log_metrics(metrics, step=step)
            logger.info(
                "%s metrics logged successfully at step %s.", len(metrics), step
            )
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="Log multiple metrics",
                raise_as=CogflowRunError,
                re_raise=True,
            )

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Log a local file or directory as an artifact to the active MLflow run.

        Artifacts can be files like datasets, plots, or configuration files that
        are relevant to the experiment.

        Behavior:
          - If `run_id` is provided → logs the artifact(s) to that specific run
            (works even if the run has already finished).
          - If `run_id` is not provided → logs to the currently active run.
            If no run is active, a new run will automatically be created.

        Args:
            local_path (str): Path to the local file or directory.
            artifact_path (Optional[str]): Optional subdirectory within the run’s artifact URI.
            run_id (Optional[str]): The ID of the run to log the artifact to.

        Returns:
            None

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowArtifactError: If artifact upload fails.

        Example:
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
        self._warn_if_unhealthy("logging artifact")

        try:
            if run_id is not None:
                logger.debug("Logging artifact to run_id %s: %s", run_id, local_path)
                self.client.log_artifact(
                    run_id=common.uuid_to_hex(run_id),
                    local_path=local_path,
                    artifact_path=artifact_path,
                )
                logger.info("Artifact logged successfully to run %s.", run_id)
            else:
                logger.debug("Logging artifact: %s", local_path)
                self.mlflow.log_artifact(
                    local_path=local_path, artifact_path=artifact_path
                )
                logger.info("Artifact logged successfully.")
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Log artifact from {local_path}",
                raise_as=CogflowArtifactError,
                re_raise=True,
            )

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Log all files within a directory as artifacts to the current MLflow run.

        Args:
            local_dir (str): Path to a local directory containing files to upload.
            artifact_path (Optional[str]): Optional destination path in artifact storage.
            run_id (Optional[str]): The ID of the run to log the artifacts to.

        Returns:
            None

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowArtifactError: If artifact upload fails.

        Example:
            >>> from cogflow import models
            >>> models.log_artifacts("checkpoints", artifact_path="saved_models")
        """
        self._warn_if_unhealthy("logging multiple artifacts")

        try:
            if run_id is not None:
                logger.debug(
                    "Logging artifacts to run_id %s from directory: %s",
                    run_id,
                    local_dir,
                )
                self.client.log_artifacts(
                    run_id=common.uuid_to_hex(run_id),
                    local_dir=local_dir,
                    artifact_path=artifact_path,
                )
            else:
                logger.debug("Logging artifacts from directory: %s", local_dir)
                self.mlflow.log_artifacts(
                    local_dir=local_dir, artifact_path=artifact_path
                )
                logger.info("All artifacts logged successfully.")
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Log artifacts from directory {local_dir}",
                raise_as=CogflowArtifactError,
                re_raise=True,
            )

    def search_registered_models(
        self,
        filter_string: Optional[str] = None,
        max_results: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        page_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for registered models in the MLflow Model Registry.

        This method retrieves all registered models from the MLflow Tracking Server
        that match the given filter or sorting criteria. It is typically accessed via:

        Args:
            filter_string (Optional[str]): An optional filter string to narrow down
                the search results based on model attributes (e.g., "name ILIKE '%forecast%'").
            max_results (Optional[int]): Maximum number of results to return.
                Defaults to ``config.MAX_RESULTS`` if not specified.
            order_by (Optional[List[str]]): List of fields to order the results by
                (e.g., ["name ASC", "last_updated_timestamp DESC"]).
            page_token (Optional[str]): Token for pagination to retrieve subsequent pages.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
            metadata about a registered model such as its name, creation time, and latest versions.

        Raises:
            CogflowConnectionError: If the MLflow tracking server is unreachable.
            CogflowModelError: If an internal MLflow or API error occurs during the query.

        Examples:
            >>> from cogflow import models
            >>> results = models.search_registered_models(filter_string="name ILIKE '%forecast%'")
            >>> for model in results:
            ...     print(model["name"], model["last_updated_timestamp"])
            my_forecast_v1 1730191829
            my_forecast_v2 1730250042

        Notes:
            - This method depends on the MLflow tracking server configured in ``COGFLOW_MLFLOW_TRACKING_URI``.
            - Results are automatically paginated if the number of models exceeds ``max_results``.
        """
        # Warn user if MLflow may be unhealthy
        self._warn_if_unhealthy("searching registered models")
        max_results = max_results or config.MAX_RESULTS
        try:
            logger.debug("Searching registered models: filter=%s", filter_string)
            registered_models = self.client.search_registered_models(
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )
            results = [
                rm.to_dictionary() if hasattr(rm, "to_dictionary") else rm.__dict__
                for rm in registered_models
            ]
            logger.info("Found %s registered model(s).", len(results))
            return results
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e, context="Searching registered models", raise_as=CogflowModelError
            )

    def autolog(self) -> None:
        """
        Enable automatic logging of parameters, metrics, and models with MLflow.

        This method enables MLflow’s autologging feature, which automatically records
        parameters, metrics, model artifacts, and environment configurations
        for supported machine learning frameworks (e.g., scikit-learn, TensorFlow, PyTorch).

        Example:
            >>> manager = ModelManager()
            >>> manager.autolog()
            >>> # After enabling autolog, MLflow will automatically log model details.

        Returns:
            None

        Raises:
            Exception: If MLflow autologging setup fails.
        """
        if not self._ensure_health():
            logger.warning(
                "MLflow tracking server at %s may be unreachable. Autolog setup might fail.",
                config.MLFLOW_TRACKING_URI,
            )

        try:
            logger.info("Enabling MLflow autologging.")
            self.mlflow.autolog()
            logger.info("MLflow autologging successfully enabled.")
        except Exception as e:
            logger.error("Failed to enable MLflow autologging: %s", e)
            raise

    def set_tracking_uri(self, tracking_uri: str) -> None:
        """
        Set the MLflow tracking server URI.

        This method updates the MLflow tracking server URI, allowing CogFlow
        to communicate with a new or external MLflow instance dynamically.

        Example:
            >>> manager = ModelManager()
            >>> manager.set_tracking_uri("http://mlflow.my-domain.com:5000")

        Args:
            tracking_uri (str): The URI of the MLflow tracking server.
                Example: "http://mlflow.kubeflow.svc.cluster.local:5000".

        Returns:
            None

        Raises:
            Exception: If updating the tracking URI fails.
        """
        try:
            logger.info("Setting MLflow tracking URI: %s", tracking_uri)
            self.mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking URI updated successfully.")
        except Exception as e:
            logger.error("Failed to set MLflow tracking URI: %s", e)
            raise

    def get_artifact_uri(self, artifact_path: Optional[str] = None) -> str:
        """
        Retrieve the artifact URI of the current or specified MLflow run.

        This method returns the URI to the artifact directory for the current run,
        or for a given artifact path if specified.

        Args:
            artifact_path (Optional[str]): The relative path of the artifact within the run's
                artifact directory. If not provided, returns the base artifact URI for the current run.

        Returns:
            str: The artifact URI for the specified path or the current run.

        Raises:
            Exception: If retrieval fails or no active run exists.

        Example:
            >>> from cogflow import models
            >>> # Get the base artifact URI for the current run
            >>> uri = models.get_artifact_uri()
            >>> print(uri)
            s3://mlflow-artifacts/1234/abcd5678/artifacts

            >>> # Get the URI for a specific artifact path within the run
            >>> uri = models.get_artifact_uri(artifact_path="plots/roc_curve.png")
            >>> print(uri)
            s3://mlflow-artifacts/1234/abcd5678/artifacts/plots/roc_curve.png

        """
        if not self._ensure_health():
            logger.warning(
                "Tracking server might be unreachable; artifact URI may not resolve."
            )

        try:
            uri = self.mlflow.get_artifact_uri(artifact_path=artifact_path)
            logger.debug("Retrieved artifact URI: %s", uri)
            return uri
        except Exception as e:
            logger.error("Failed to get artifact URI: %s", e)
            raise

    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a new experiment in MLflow.

        Args:
            name (str): Name of the experiment to create.
            artifact_location (Optional[str]): Custom base artifact location for the experiment.
            tags (Optional[Dict[str, str]]): Key-value tags to assign to the experiment.

        Returns:
            str: The experiment ID of the newly created experiment.

        Raises:
            Exception: If experiment creation fails.
        Example:
            >>> from cogflow import models
            >>> exp_id = models.create_experiment(
            ...     name="my_new_experiment",
            ...     artifact_location="s3://mlflow-artifacts/experiments/my_new_experiment",
            ...     tags={"team": "mlops", "purpose": "ablation study"}
            ... )
            >>> print(exp_id)
            42
        """
        if not self._ensure_health():
            logger.warning(
                "Tracking server may be unreachable; experiment creation may fail."
            )

        try:
            logger.info("Creating new experiment: %s", name)
            exp_id = self.client.create_experiment(
                name=name, artifact_location=artifact_location, tags=tags
            )
            logger.info("Experiment %s created successfully with ID: %s", name, exp_id)
            return exp_id
        except Exception as e:
            logger.error("Failed to create experiment %s: %s", name, e)
            raise


# ---------------------------------------------------------------------
#  Create a singleton instance for the public interface
# ---------------------------------------------------------------------
_models = ModelManager()

# Exposed SDK-level methods (single source of truth)
for attr_name in dir(ModelManager):
    # Skip private methods and dunder methods
    if attr_name.startswith("_"):
        continue

    attr = getattr(ModelManager, attr_name)

    # Only export methods (callables) that belong to ModelManager
    if callable(attr):
        # Bind the method to the singleton instance
        globals()[attr_name] = getattr(_models, attr_name)
