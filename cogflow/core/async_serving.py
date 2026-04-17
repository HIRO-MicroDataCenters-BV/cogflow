"""
CogFlow - Async Serving Manager (Kubernetes-native)
----------------------------------------------------

Async version of ServingManager for use in async frameworks (FastAPI, etc.).
Uses kubernetes_asyncio instead of kubernetes for non-blocking I/O.

All public methods mirror ServingManager but are async.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from kubernetes_asyncio import client as async_client, config as async_config
from kubernetes_asyncio.client.exceptions import ApiException

from ..utils import common
from ..config import config as cog_config
from ..utils.exceptions import (
    CogflowErrorHandler,
    CogflowValidationError,
    CogflowModelError,
    CogflowDatasetError,
    CogflowConnectionError,
    CogflowServingError,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _get_serving_manager_class():
    """Lazy import to avoid circular dependency with serving.py."""
    from .serving import ServingManager
    return ServingManager

# ---------------------------------------------------------------------
# Global cache: Ensure async K8s config is only loaded once
# ---------------------------------------------------------------------
_ASYNC_K8S_CONFIG_LOADED = False


async def _ensure_async_k8s_config_loaded():
    """Load Kubernetes config for async client only once."""
    global _ASYNC_K8S_CONFIG_LOADED
    if _ASYNC_K8S_CONFIG_LOADED:
        return

    try:
        async_config.load_incluster_config()
    except async_config.config_exception.ConfigException:
        await async_config.load_kube_config()

    _ASYNC_K8S_CONFIG_LOADED = True


class AsyncServingManager:
    """
    Async version of ServingManager for non-blocking K8s operations.

    Shares static/pure methods with ServingManager (_process_isvc,
    _validate_canary, etc.) but uses kubernetes_asyncio for all API calls.
    """

    GROUP = "serving.kserve.io"
    VERSION = "v1beta1"
    PLURAL = "inferenceservices"

    @staticmethod
    def _validate_canary(*args, **kwargs):
        return _get_serving_manager_class()._validate_canary(*args, **kwargs)

    @staticmethod
    def _process_isvc(*args, **kwargs):
        return _get_serving_manager_class()._process_isvc(*args, **kwargs)

    @staticmethod
    def _get_model_helpers():
        return _get_serving_manager_class()._get_model_helpers()

    @staticmethod
    def _get_dataset_manager():
        return _get_serving_manager_class()._get_dataset_manager()

    def __init__(self):
        """Initialize async Kubernetes client."""
        self._api = None

    async def _get_api(self):
        """Lazy-initialize the async K8s API client."""
        if self._api is None:
            await _ensure_async_k8s_config_loaded()
            self._api = async_client.CustomObjectsApi()
        return self._api

    def _get_transformer_env(
        self,
        dataset_id: Optional[str],
        transformer_image: Optional[str],
        transformer_parameters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve transformer parameters (sync — no K8s I/O)."""
        # Reuse sync logic — this method does no K8s I/O
        if transformer_parameters:
            return transformer_parameters
        if not dataset_id:
            return {}
        try:
            dataset_mgr = self._get_dataset_manager()
            dataset_id_norm = common.normalize_uuid(dataset_id)
            dataset = dataset_mgr.get_dataset(dataset_id_norm)
            if dataset.get("data_source_type") == 20:
                if not transformer_image:
                    CogflowErrorHandler.log_and_raise(
                        "Dataset is Prometheus type but no transformer_image was provided.",
                        raise_as=CogflowValidationError,
                    )
                prom = dataset_mgr.get_prometheus_dataset(dataset_id_norm)
                return {
                    "PROMETHEUS_URL": prom.get("connection_type", {}).get("prometheus_url"),
                    "PROMETHEUS_METRICS": prom.get("metric_list", {}).get("METRIC_FEATURES"),
                }
            return {}
        except (CogflowValidationError, CogflowDatasetError):
            raise
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="Resolve transformer parameters from dataset",
                raise_as=CogflowDatasetError,
                re_raise=True,
            )

    # -----------------------------------------------------------------
    # CRUD Operations (async)
    # -----------------------------------------------------------------

    async def get_isvc(self, name: str, namespace: Optional[str] = None) -> dict:
        """Retrieve an InferenceService CRD by name (async)."""
        namespace = namespace or common.get_namespace()
        api = await self._get_api()
        try:
            return await api.get_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=self.PLURAL,
                name=name,
            )
        except ApiException as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Get InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

    async def delete_isvc(self, name: str, namespace: Optional[str] = None) -> bool:
        """Delete an InferenceService CRD (async)."""
        namespace = namespace or common.get_namespace()
        api = await self._get_api()
        try:
            await api.delete_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=self.PLURAL,
                name=name,
            )
            logger.info("InferenceService '%s' deleted from namespace '%s'.", name, namespace)
            return True
        except ApiException as e:
            if e.status == 404:
                logger.warning("InferenceService '%s' not found in '%s'.", name, namespace)
                return False
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Delete InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

    async def update_isvc(self, name: str, patch_body: dict, namespace: Optional[str] = None) -> dict:
        """Patch an existing InferenceService CRD (async)."""
        namespace = namespace or common.get_namespace()
        api = await self._get_api()
        try:
            return await api.patch_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=self.PLURAL,
                name=name,
                body=patch_body,
            )
        except ApiException as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Patch InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

    async def restart_isvc(self, name: str, namespace: Optional[str] = None) -> bool:
        """Restart ISVC by scaling to 0 then back to 1 (async)."""
        namespace = namespace or common.get_namespace()
        try:
            await self.update_isvc(name, {"spec": {"predictor": {"minReplicas": 0}}}, namespace)
            await asyncio.sleep(1)
            await self.update_isvc(name, {"spec": {"predictor": {"minReplicas": 1}}}, namespace)
            logger.info("InferenceService '%s' restart patches applied.", name)
            return True
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Restart InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowServingError,
                re_raise=True,
            )

    async def create_isvc(
        self,
        name: str,
        model_uri: str,
        namespace: Optional[str] = None,
        model_format: str = None,
        protocol_version: str = None,
        annot: dict = None,
        transformer_image: str = cog_config.TRANSFORMER_BASE_IMAGE,
        transformer_env: dict = None,
    ) -> dict:
        """Create a new InferenceService CRD (async)."""
        namespace = namespace or common.get_namespace()
        api = await self._get_api()

        # Build model spec
        model_spec: Dict[str, Any] = {"storageUri": model_uri}
        if model_format:
            model_spec["modelFormat"] = {"name": model_format}
        if protocol_version:
            model_spec["protocolVersion"] = protocol_version

        predictor_spec = {
            "serviceAccountName": "kserve-controller-s3",
            "minReplicas": 1,
            "model": model_spec,
        }

        spec: Dict[str, Any] = {"predictor": predictor_spec}

        # Transformer (top-level spec field, only when env is provided)
        if transformer_env:
            env_list = [
                {"name": k, "value": str(v)} for k, v in transformer_env.items()
            ]
            spec["transformer"] = {
                "containers": [
                    {
                        "name": f"{name}-transformer",
                        "image": transformer_image,
                        "env": env_list,
                    }
                ]
            }

        metadata = async_client.V1ObjectMeta(
            name=name,
            namespace=namespace,
            annotations=annot or {},
        )

        body = {
            "apiVersion": f"{self.GROUP}/{self.VERSION}",
            "kind": "InferenceService",
            "metadata": metadata.to_dict(),
            "spec": spec,
        }

        try:
            created = await api.create_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=self.PLURAL,
                body=body,
            )
            logger.info("InferenceService '%s' created in namespace '%s'.", name, namespace)
            return created
        except ApiException as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Create InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

    # -----------------------------------------------------------------
    # HIGH-LEVEL: DEPLOY MODEL (async)
    # -----------------------------------------------------------------

    async def deploy_model(
        self,
        model_id: str = None,
        isvc_name: Optional[str] = None,
        artifact_path: str = None,
        model_name: str = None,
        model_version: str = None,
        dataset_id: str = None,
        transformer_image: str = cog_config.TRANSFORMER_BASE_IMAGE,
        transformer_parameters: Dict[str, Any] = None,
        protocol_version: str = None,
        model_format: str = None,
        namespace: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        """Deploy a model as an InferenceService (async)."""
        namespace = namespace or common.get_namespace()

        try:
            detect_model_format, get_model_details = self._get_model_helpers()
            model_details = get_model_details(
                model_id=model_id,
                artifact_path=artifact_path,
                model_name=model_name,
                model_version=model_version,
            )
            model_uri = model_details["model_uri"]

            transformer_env = self._get_transformer_env(
                dataset_id, transformer_image, transformer_parameters
            )

            model_format = model_format or detect_model_format(model_uri=model_uri)

            if not isvc_name:
                isvc_name = f"model-{common.normalize_uuid(model_details['model_id'])}"

            annot = {
                "model_id": model_details["model_id"],
                "model_name": model_details["model_name"],
                "model_version": model_details["model_version"],
            }
            if dataset_id:
                annot["dataset_id"] = common.normalize_uuid(dataset_id)
            if model_type:
                annot["model_type"] = model_type

            logger.info(
                "Creating InferenceService '%s' for model_uri=%s in namespace=%s.",
                isvc_name, model_uri, namespace,
            )

            return await self.create_isvc(
                name=isvc_name,
                model_uri=model_uri,
                model_format=model_format,
                protocol_version=protocol_version,
                transformer_image=transformer_image,
                transformer_env=transformer_env,
                annot=annot,
                namespace=namespace,
            )
        except (CogflowValidationError, CogflowModelError, CogflowDatasetError):
            raise
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="Serve model via AsyncServingManager",
                raise_as=CogflowServingError,
                re_raise=True,
            )

    # -----------------------------------------------------------------
    # HIGH-LEVEL: UPDATE MODEL (async)
    # -----------------------------------------------------------------

    async def update_model(
        self,
        isvc_name: str,
        model_id: Optional[str] = None,
        artifact_path: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        dataset_id: Optional[str] = None,
        transformer_image: Optional[str] = cog_config.TRANSFORMER_BASE_IMAGE,
        transformer_parameters: Optional[dict] = None,
        protocol_version: Optional[str] = None,
        namespace: Optional[str] = None,
        model_format: Optional[str] = None,
        canary_traffic_percent: Optional[int] = None,
        model_type: Optional[str] = None,
    ) -> str:
        """Update an existing InferenceService (async)."""
        namespace = namespace or common.get_namespace()

        try:
            # Fetch existing ISVC. get_isvc() already wraps ApiException
            # (including 404) into CogflowConnectionError, so we re-raise here.
            isvc = await self.get_isvc(isvc_name, namespace)

            # Case 1: Traffic-only update
            if canary_traffic_percent is not None and not (
                model_id or model_name or model_version or artifact_path
            ):
                self._validate_canary(isvc, isvc_name, canary_traffic_percent)
                patch_body = {
                    "spec": {"predictor": {"canaryTrafficPercent": canary_traffic_percent}}
                }
                await self.update_isvc(isvc_name, patch_body, namespace)
                msg = f"Updated traffic to {canary_traffic_percent}% for '{isvc_name}'."
                logger.info("%s", msg)
                return msg

            # Case 2: Full model rollout/update
            detect_model_format, get_model_details = self._get_model_helpers()
            model_details = get_model_details(
                model_id=model_id,
                artifact_path=artifact_path,
                model_name=model_name,
                model_version=model_version,
            )

            transformer_env = self._get_transformer_env(
                dataset_id, transformer_image, transformer_parameters
            )

            model_uri = model_details["model_uri"]
            model_format = model_format or detect_model_format(model_uri=model_uri)

            annot = {
                "model_id": model_details["model_id"],
                "model_name": model_details["model_name"],
                "model_version": model_details["model_version"],
            }
            if dataset_id:
                annot["dataset_id"] = common.normalize_uuid(dataset_id)
            if model_type:
                annot["model_type"] = model_type

            # Build model patch
            model_patch: Dict[str, Any] = {}
            if model_uri:
                model_patch["storageUri"] = model_uri
            if model_format:
                model_patch["modelFormat"] = {"name": model_format}
            if protocol_version:
                model_patch["protocolVersion"] = protocol_version

            predictor_patch: Dict[str, Any] = {"model": model_patch}

            if canary_traffic_percent is not None:
                if not 0 <= canary_traffic_percent <= 100:
                    CogflowErrorHandler.log_and_raise(
                        f"Invalid canary_traffic_percent={canary_traffic_percent}.",
                        raise_as=CogflowValidationError,
                    )
                predictor_patch["canary"] = {"model": model_patch}
                predictor_patch["canaryTrafficPercent"] = canary_traffic_percent

            patch_body: Dict[str, Any] = {
                "metadata": {"annotations": annot},
                "spec": {"predictor": predictor_patch},
            }

            # Transformer update (top-level spec field, aligned with sync impl)
            if transformer_env:
                env_list = [
                    {"name": k, "value": str(v)} for k, v in transformer_env.items()
                ]
                patch_body["spec"]["transformer"] = {
                    "containers": [
                        {
                            "name": f"{isvc_name}-transformer",
                            "image": transformer_image or cog_config.TRANSFORMER_BASE_IMAGE,
                            "env": env_list,
                        }
                    ]
                }

            await self.update_isvc(isvc_name, patch_body, namespace)
            msg = f"Updated InferenceService '{isvc_name}' in namespace '{namespace}'."
            logger.info("%s", msg)
            return msg

        except (
            CogflowValidationError,
            CogflowModelError,
            CogflowDatasetError,
            CogflowServingError,
            CogflowConnectionError,
        ):
            raise
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Update model in '{isvc_name}'",
                raise_as=CogflowServingError,
                re_raise=True,
            )

    # -----------------------------------------------------------------
    # LIST / INSPECT (async)
    # -----------------------------------------------------------------

    async def list_models(
        self,
        namespace: Optional[str] = None,
        isvc_name: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """List served models / get a specific InferenceService (async)."""
        ns = namespace or common.get_namespace()
        api = await self._get_api()

        try:
            if isvc_name:
                isvc = await self.get_isvc(isvc_name, ns)
                return [self._process_isvc(isvc)] if isvc else None

            result = await api.list_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=ns,
                plural=self.PLURAL,
            )

            items = result.get("items", [])
            if not items:
                return []

            return [self._process_isvc(isvc) for isvc in items]

        except CogflowConnectionError:
            raise
        except ApiException as e:
            if e.status == 404:
                logger.info("No InferenceServices found in namespace '%s'.", ns)
                return None
            CogflowErrorHandler.handle_exception(
                e,
                context=f"List InferenceServices in namespace '{ns}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="list_models via AsyncServingManager",
                raise_as=CogflowServingError,
                re_raise=True,
            )


    # -----------------------------------------------------------------
    # LLM SERVING (async)
    # -----------------------------------------------------------------

    async def deploy_llm(
        self,
        *,
        storage_uri: str,
        isvc_name: str,
        served_model_name: str,
        namespace: Optional[str] = None,
        max_model_len: Optional[int] = None,
        dtype: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        trust_remote_code: bool = False,
        gpu_memory_utilization: Optional[float] = None,
        max_num_seqs: Optional[int] = None,
        resources: Optional[Dict[str, Dict[str, str]]] = None,
        tolerations: Optional[List[Dict[str, Any]]] = None,
        node_selector: Optional[Dict[str, str]] = None,
        min_replicas: int = 1,
        max_replicas: int = 1,
        hf_secret_name: Optional[str] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Create a KServe HF-runtime InferenceService (async).

        Spec-building is delegated to ``ServingManager._build_llm_predictor``
        so there is a single source of truth for the ISVC shape; this method
        adds the async CRD create and the async K8s client plumbing.
        """
        namespace = namespace or common.get_namespace()
        api = await self._get_api()

        sync_manager_cls = _get_serving_manager_class()
        predictor_spec = sync_manager_cls._build_llm_predictor(
            storage_uri=storage_uri,
            served_model_name=served_model_name,
            max_model_len=max_model_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            resources=resources,
            tolerations=tolerations,
            node_selector=node_selector,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            hf_secret_name=hf_secret_name,
        )

        metadata = async_client.V1ObjectMeta(
            name=isvc_name,
            namespace=namespace,
            annotations=annotations or {},
        )
        body = {
            "apiVersion": f"{self.GROUP}/{self.VERSION}",
            "kind": "InferenceService",
            "metadata": metadata.to_dict(),
            "spec": {"predictor": predictor_spec},
        }

        try:
            created = await api.create_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=self.PLURAL,
                body=body,
            )
            logger.info(
                "LLM InferenceService '%s' created in namespace '%s'.",
                isvc_name,
                namespace,
            )
            return created
        except ApiException as e:
            if e.status == 409:
                CogflowErrorHandler.handle_exception(
                    e,
                    context=(
                        f"LLM InferenceService '{isvc_name}' already exists in "
                        f"namespace '{namespace}'."
                    ),
                    raise_as=CogflowValidationError,
                    re_raise=True,
                )
            CogflowErrorHandler.handle_exception(
                e,
                context=(
                    f"Create LLM InferenceService '{isvc_name}' in namespace "
                    f"'{namespace}'"
                ),
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=(
                    f"Create LLM InferenceService '{isvc_name}' in namespace "
                    f"'{namespace}'"
                ),
                raise_as=CogflowServingError,
                re_raise=True,
            )


# Create singleton and expose async methods at module level
_async_serving = AsyncServingManager()

async_deploy_model = _async_serving.deploy_model
async_update_model = _async_serving.update_model
async_delete_isvc = _async_serving.delete_isvc
async_list_models = _async_serving.list_models
async_get_isvc = _async_serving.get_isvc
async_update_isvc = _async_serving.update_isvc
async_restart_isvc = _async_serving.restart_isvc
async_create_isvc = _async_serving.create_isvc
async_deploy_llm = _async_serving.deploy_llm
