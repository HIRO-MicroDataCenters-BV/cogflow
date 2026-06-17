"""
CogFlow - Serving Manager (Kubernetes-native)
---------------------------------------------

This module handles deployment and lifecycle management of model-serving
InferenceService (ISVC) CRDs in Kubernetes.

It depends on:
    - models.py       (lazy import inside methods)
    - datasets.py     (lazy import inside methods)

No circular imports occur because we load them lazily.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any

from kubernetes import client
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException

from ..config import config as cog_config
from ..utils import common
from ..utils.exceptions import (
    CogflowConnectionError,
    CogflowDatasetError,
    CogflowErrorHandler,
    CogflowModelError,
    CogflowServingError,
    CogflowValidationError,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Global cache: Ensure K8s config is only loaded once
# ---------------------------------------------------------------------
_K8S_CONFIG_LOADED = False


def _ensure_k8s_config_loaded():
    """Load Kubernetes config only once across all ServingManager instances."""
    if getattr(common, "_k8s_loaded_flag", False):
        return

    common.load_k8s_config()

    # mark flag on common module (internal, invisible)
    common._k8s_loaded_flag = True


# Sentinel values that mean "no flag emitted" for quantization /
# kv_cache_dtype runtime args. ``"none"`` and ``"auto"`` aren't real
# vLLM CLI values — they're recommender-side labels for the absence of
# a runtime cast — so emitting them would either no-op (best case) or
# break model load (vLLM rejects ``--quantization=auto``). The empty
# string is included because a stringly-typed config layer can
# substitute it for unset.
_LLM_ARG_SENTINELS = frozenset({"", "none", "auto"})


def _is_unset_sentinel(value: str | None) -> bool:
    """Return True when ``value`` should suppress the corresponding flag.

    Matches case-insensitively and ignores surrounding whitespace so the
    diff-clean guarantee holds when callers normalise to ``"AUTO"`` or
    ``" none "``. Real CLI values (e.g. ``"fp8"``, ``"fp8_e4m3"``) are
    preserved untouched and emitted verbatim.
    """
    if value is None:
        return True
    return value.strip().lower() in _LLM_ARG_SENTINELS


# =====================================================================
#   Serving Manager (Kubernetes-native)
# =====================================================================


class ServingManager:
    """Manages model-serving InferenceService (ISVC) CRDs in Kubernetes."""

    GROUP = "serving.kserve.io"
    VERSION = "v1beta1"
    PLURAL = "inferenceservices"

    def __init__(self):
        """Initialize the Kubernetes client if cluster config is available.

        This initialization is tolerant of missing kube config so importing the
        serving interface (for example, ``from cogflow import serving``) does not
        fail in environments without a cluster (CI, local dev, build agents).

        If config cannot be loaded, ``self._api`` remains ``None`` and the
        ``self.api`` property will retry on first access, raising
        ``CogflowConnectionError`` if k8s is still unavailable.
        """
        self._api = None
        try:
            _ensure_k8s_config_loaded()
            self._api = client.CustomObjectsApi()
        except (ConfigException, FileNotFoundError, OSError) as exc:
            logger.warning(
                "Kubernetes config could not be loaded: %s. "
                "ServingManager cluster operations will fail until config is available.",
                exc,
            )

    @property
    def api(self):
        """Lazily-validated Kubernetes CustomObjectsApi.

        Returns the cached client if already initialized; otherwise retries
        config load and raises ``CogflowConnectionError`` if it still fails.
        Routing every existing ``self.api.<call>`` site through this property
        means cluster operations get a clear, typed error instead of an
        ``AttributeError`` on ``None.<call>``.
        """
        if self._api is not None:
            return self._api
        try:
            _ensure_k8s_config_loaded()
            self._api = client.CustomObjectsApi()
        except (ConfigException, FileNotFoundError, OSError) as exc:
            raise CogflowConnectionError(f"Kubernetes config is not loaded: {exc}") from exc
        return self._api

    # -----------------------------------------------------------------
    # Lazy-load helpers (NO circular imports)
    # -----------------------------------------------------------------
    @staticmethod
    def _get_model_helpers():
        """Lazy import to avoid circular dependencies."""
        from cogflow import models

        return (
            models.detect_model_format,
            models.get_full_model_uri_from_run_or_registry,
        )

    @staticmethod
    def _get_dataset_manager():
        """Lazy import to avoid circular dependencies."""
        from cogflow.datasets import DatasetManager

        logger.debug("Lazy-loaded DatasetManager in ServingManager.")
        return DatasetManager()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _get_transformer_env(
        self,
        dataset_id: str | None,
        transformer_image: str | None,
        transformer_parameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Resolve transformer parameters:

        - If transformer_parameters explicitly provided -> return as-is.
        - If dataset_id is a Prometheus dataset -> derive PROMETHEUS_* params.
        """
        # Explicit transformer parameters → just return them
        if transformer_parameters:
            logger.debug(
                "Transformer parameters explicitly provided: %s",
                transformer_parameters,
            )
            return transformer_parameters

        if not dataset_id:
            return {}

        try:
            dataset_mgr = self._get_dataset_manager()
            dataset_id_norm = common.normalize_uuid(dataset_id)

            logger.info(
                "Resolving transformer parameters from dataset_id=%s (normalized=%s).",
                dataset_id,
                dataset_id_norm,
            )

            dataset = dataset_mgr.get_dataset(dataset_id_norm)

            # Prometheus dataset logic (data_source_type == 20)
            if dataset.get("data_source_type") == 20:
                if not transformer_image:
                    CogflowErrorHandler.log_and_raise(
                        ("Dataset is Prometheus type but no transformer_image was provided."),
                        raise_as=CogflowValidationError,
                    )

                prom = dataset_mgr.get_prometheus_dataset(dataset_id_norm)

                params = {
                    "PROMETHEUS_URL": prom.get("connection_type", {}).get("prometheus_url"),
                    "PROMETHEUS_METRICS": prom.get("metric_list", {}).get("METRIC_FEATURES"),
                }
                logger.info(
                    "Prometheus transformer parameters resolved for dataset_id=%s: %s",
                    dataset_id_norm,
                    params,
                )
                return params

            logger.debug(
                "Dataset_id=%s is not Prometheus type; no transformer parameters needed.",
                dataset_id_norm,
            )
            return {}

        except CogflowValidationError:
            raise
        except CogflowDatasetError:
            raise
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="Resolve transformer parameters from dataset",
                raise_as=CogflowDatasetError,
                re_raise=True,
            )

    @staticmethod
    def _validate_canary(isvc: dict, isvc_name: str, pct: int):
        """
        Validate canary_traffic_percent based on current ISVC state.

        Rules:
            - If canary DOES NOT exist (first rollout):
                  allowed = 1–100 (but not 0)
            - If canary EXISTS (promotion/update):
                  allowed = 0–100
        """
        if pct is None:
            return

        predictor = isvc.get("spec", {}).get("predictor", {})
        existing_canary_pct = predictor.get("canaryTrafficPercent")

        canary_exists = existing_canary_pct is not None

        if canary_exists:
            # Promotion or disable
            if not 0 <= pct <= 100:
                CogflowErrorHandler.log_and_raise(
                    (
                        f"Invalid canary_traffic_percent={pct}. "
                        f"For an existing canary in '{isvc_name}', "
                        "value must be between 0 and 100."
                    ),
                    raise_as=CogflowValidationError,
                )
            return

        # First-time rollout rules
        if not 1 <= pct <= 100:
            CogflowErrorHandler.log_and_raise(
                (f"Invalid canary_traffic_percent={pct}. Initial rollout for '{isvc_name}' must be between 1 and 99."),
                raise_as=CogflowValidationError,
            )
        return

    # -----------------------------------------------------------------
    # CRUD Operations
    # -----------------------------------------------------------------

    def get_isvc(self, name: str, namespace: str | None = None) -> dict:
        """
        Retrieve an InferenceService CRD by name and namespace.
        Args:
            name: InferenceService name.
            namespace: Namespace where the InferenceService is deployed.

        Returns:
            dict: InferenceService details.
        Raises:
            CogflowConnectionError: If there is a connection issue with Kubernetes API.
        Examples:
            >>> from cogflow import serving
            >>> isvc = serving.get_isvc("my-inferenceservice", "default")
            >>> print(isvc)

        """
        namespace = namespace or common.get_namespace()
        logger.info("Fetching InferenceService name=%s namespace=%s", name, namespace)
        try:
            return self.api.get_namespaced_custom_object(
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

    def delete_isvc(self, name: str, namespace: str | None = None) -> bool:
        """
        Delete an InferenceService CRD by name and namespace.
        Args:
            name: InferenceService name.
            namespace: Namespace where the InferenceService is deployed.
        Returns:
            bool: True if deleted successfully, False if not found.
        Raises:
            CogflowConnectionError: If there is a connection issue with Kubernetes API.
        Examples:
            >>> from cogflow import serving
            >>> success = serving.delete_isvc("my-inferenceservice", "default")
            >>> print(success)
        """
        namespace = namespace or common.get_namespace()
        logger.info("Deleting InferenceService name=%s namespace=%s", name, namespace)
        try:
            self.api.delete_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=self.PLURAL,
                name=name,
            )
            logger.info(
                "InferenceService '%s' deleted successfully from namespace '%s'.",
                name,
                namespace,
            )
            return True
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    "InferenceService '%s' not found in namespace '%s' during delete.",
                    name,
                    namespace,
                )
                return False
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Delete InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

    def update_isvc(
        self,
        name: str,
        patch_body: dict,
        namespace: str | None = None,
    ) -> dict:
        """
        Patch an existing InferenceService CRD.
        Args:
            name: InferenceService name.
            patch_body: Dictionary representing the patch to apply.
            namespace: Namespace where the InferenceService is deployed.
        Returns:
            dict: Updated InferenceService details.
        Raises:
            CogflowConnectionError: If there is a connection issue with Kubernetes API.
        Examples:
            >>> from cogflow import serving
            >>> patch = {"spec": {"predictor": {"minReplicas": 2}}}
            >>> updated_isvc = serving.update_isvc("my-inferenceservice", patch, "default")
            >>> print(updated_isvc)
        """
        namespace = namespace or common.get_namespace()
        logger.info(
            "Patching InferenceService name=%s namespace=%s keys=%s",
            name,
            namespace,
            list(patch_body.keys()),
        )
        try:
            return self.api.patch_namespaced_custom_object(
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

    def restart_isvc(self, name: str, namespace: str | None = None) -> bool:
        """
        Restart ISVC by scaling minReplicas to 0 then back to 1.
        Args:
            name: InferenceService name.
            namespace: Namespace where the InferenceService is deployed.
        Returns:
            bool: True if restart initiated successfully.
        Raises:
            CogflowServingError: If restart fails.
        Examples:
            >>> from cogflow import serving
            >>> success = serving.restart_isvc("my-inferenceservice", "default")
            >>> print(success)
        """
        namespace = namespace or common.get_namespace()
        logger.info(
            "Restarting InferenceService name=%s namespace=%s via scale down/up.",
            name,
            namespace,
        )
        try:
            # Scale to 0
            self.update_isvc(name, {"spec": {"predictor": {"minReplicas": 0}}}, namespace)
            time.sleep(1)
            # Scale back up
            self.update_isvc(name, {"spec": {"predictor": {"minReplicas": 1}}}, namespace)
            logger.info(
                "InferenceService '%s' restart patches applied successfully.",
                name,
            )
            return True
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Restart InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowServingError,
                re_raise=True,
            )

    # -----------------------------------------------------------------
    # CREATE SERVICE
    # -----------------------------------------------------------------

    def create_isvc(
        self,
        name: str,
        model_uri: str,
        namespace: str | None = None,
        model_format: str = None,
        protocol_version: str = None,
        annot: dict = None,
        transformer_image: str = cog_config.TRANSFORMER_BASE_IMAGE,
        transformer_env: dict = None,
    ) -> dict:
        """
        Create an InferenceService CRD.

        Keys with None are not sent in the spec to avoid invalid patches.
        Args:
            name: InferenceService name.
            model_uri: URI where the model artifacts are stored.
            namespace: Namespace where the InferenceService will be deployed.
            model_format: Model format (e.g., "tensorflow", "pytorch").
            protocol_version: Protocol version for the model server.
            annot: Annotations to attach to the InferenceService metadata.
            transformer_image: Container image for the transformer.
            transformer_env: Environment variables for the transformer container.
        Returns:
            dict: Created InferenceService details.
        Raises:
            CogflowConnectionError: If there is a connection issue with Kubernetes API.
        Examples:
            >>> from cogflow import serving
            >>> isvc = serving.create_isvc(
            ...     name="my-inferenceservice",
            ...     model_uri="s3://my-bucket/model/",
            ...     namespace="default",
            ...     model_format="tensorflow",
            ... )
            >>> print(isvc)
        """
        namespace = namespace or common.get_namespace()
        logger.info(
            "Creating InferenceService name=%s namespace=%s model_uri=%s",
            name,
            namespace,
            model_uri,
        )
        try:
            metadata = client.V1ObjectMeta(
                name=name,
                namespace=namespace,
                annotations=annot or {},
            )

            model_spec: dict[str, Any] = {"storageUri": model_uri}
            if model_format:
                model_spec["modelFormat"] = {"name": model_format}
            if protocol_version:
                model_spec["protocolVersion"] = protocol_version

            predictor_spec = {
                "serviceAccountName": "kserve-controller-s3",
                "minReplicas": 1,
                "model": model_spec,
            }

            spec: dict[str, Any] = {"predictor": predictor_spec}

            # Transformer
            if transformer_env:
                env_list = [{"name": k, "value": str(v)} for k, v in transformer_env.items()]
                spec["transformer"] = {
                    "containers": [
                        {
                            "name": f"{name}-transformer",
                            "image": transformer_image,
                            "env": env_list,
                        }
                    ]
                }

            body = {
                "apiVersion": f"{self.GROUP}/{self.VERSION}",
                "kind": "InferenceService",
                "metadata": metadata.to_dict(),
                "spec": spec,
            }

            created = self.api.create_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=self.PLURAL,
                body=body,
            )
            logger.info(
                "InferenceService '%s' created successfully in namespace '%s'.",
                name,
                namespace,
            )
            return created
        except ApiException as e:
            if e.status == 409:
                CogflowErrorHandler.handle_exception(
                    e,
                    context=f"InferenceService '{name}' not found in namespace '{namespace}'.",
                    raise_as=CogflowValidationError,
                    re_raise=True,
                )
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Create InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Create InferenceService '{name}' in namespace '{namespace}'",
                raise_as=CogflowServingError,
                re_raise=True,
            )

    # -----------------------------------------------------------------
    # LLM SERVING (KServe 0.15+ huggingface ClusterServingRuntime)
    # -----------------------------------------------------------------

    # Defaults match a small-to-mid 7B model on a single GPU. Callers can
    # override per-field via the `resources` arg.
    _LLM_DEFAULT_RESOURCES: dict[str, dict[str, str]] = {
        "requests": {"cpu": "4", "memory": "7Gi", "nvidia.com/gpu": "1"},
        "limits": {"cpu": "8", "memory": "8Gi", "nvidia.com/gpu": "1"},
    }

    @staticmethod
    def _merge_resources(
        override: dict[str, dict[str, str]] | None,
    ) -> dict[str, dict[str, str]]:
        """Per-field merge of caller overrides onto the default block."""
        merged = {
            "requests": dict(ServingManager._LLM_DEFAULT_RESOURCES["requests"]),
            "limits": dict(ServingManager._LLM_DEFAULT_RESOURCES["limits"]),
        }
        if override:
            for section in ("requests", "limits"):
                if override.get(section):
                    merged[section].update(override[section])
        return merged

    # Stricter DNS-1123 *label* check used for InferenceService names in
    # this serving flow (KServe/Knative require it — Kubernetes itself
    # accepts the looser DNS-1123 *subdomain* form for ``metadata.name``).
    # Surfaces errors here with a typed exception rather than letting the
    # K8s admission webhook return an opaque 422 later.
    _DNS1123_LABEL_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

    @staticmethod
    def _extract_hf_model_id(storage_uri: str) -> str | None:
        """Return the HF model id from an ``hf://`` URI, or ``None`` if
        the URI isn't an HF source.

        Raises ``CogflowValidationError`` when the URI is shaped like
        ``hf://`` but the extracted id is invalid — empty, contains
        whitespace, or carries an embedded scheme (``://``). The
        embedded-scheme check catches double-prefixed input like
        ``hf://hf://org/name`` where stripping only one ``hf://``
        would leave ``hf://org/name`` as the "id" and vLLM would
        then receive a bogus ``--model_id`` arg. Real HF ids are
        ``org/name`` or ``name`` — they never contain ``://``.
        """
        if not storage_uri.startswith("hf://"):
            return None
        hf_id = storage_uri[len("hf://") :].strip().strip("/")
        if not hf_id or any(ch.isspace() for ch in hf_id):
            raise CogflowValidationError(
                f"storage_uri={storage_uri!r} has an invalid HF model id; "
                f"expected 'hf://<org>/<model>' (or 'hf://<model>')"
            )
        if "://" in hf_id:
            raise CogflowValidationError(
                f"storage_uri={storage_uri!r} has an invalid HF model id "
                f"(extracted {hf_id!r} contains an embedded scheme); "
                f"expected 'hf://<org>/<model>' (or 'hf://<model>')"
            )
        return hf_id

    @staticmethod
    def _k8s_slugify(name: str) -> str:
        """Turn an arbitrary model name into a DNS-1123-compatible label.

        Lowercases, replaces any run of non-``[a-z0-9]`` with a single
        dash, strips leading/trailing dashes, and truncates to 63 chars
        (the DNS-1123 label limit).

        The result is not guaranteed to be DNS-1123-valid for adversarial
        inputs (e.g. an all-punctuation string collapses to an empty
        string); callers should validate via ``_DNS1123_LABEL_RE`` after.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        if len(slug) > 63:
            slug = slug[:63].rstrip("-")
        return slug

    @staticmethod
    def derive_llm_names(
        *,
        hf_model_id: str | None = None,
        served_model_name: str | None = None,
        isvc_name: str | None = None,
    ) -> tuple[str, str]:
        """Fill in ``(isvc_name, served_model_name)`` from ``hf_model_id``
        when not supplied, and validate the final ``isvc_name``.

        Rules:

        - ``served_model_name`` defaults to the part of ``hf_model_id``
          after the last ``/`` — ``'Qwen/Qwen2.5-Coder-7B-Instruct'``
          becomes ``'Qwen2.5-Coder-7B-Instruct'``. An ``hf_model_id`` with
          no slash is used as-is (HF supports bare org-less ids).
        - ``isvc_name`` defaults to a DNS-1123 slug of the resolved
          ``served_model_name`` (``Qwen2.5-Coder-7B-Instruct`` →
          ``qwen2-5-coder-7b-instruct``).

        Keeps the logic in cogflow so every caller (Cog-Engine today,
        direct SDK users tomorrow) gets the same defaults instead of
        reimplementing them.

        Raises
        ------
        CogflowValidationError
            If ``served_model_name`` can't be resolved (neither passed
            in nor derivable from a non-empty ``hf_model_id``), or if the
            final ``isvc_name`` is not a valid DNS-1123 label.
        """
        if not served_model_name:
            if not hf_model_id:
                raise CogflowValidationError(
                    "served_model_name is required when hf_model_id is not "
                    "provided (MLflow-backed LLM path must supply a name)"
                )
            slug_source = hf_model_id.strip().strip("/")
            # hf_model_id like 'Qwen/Qwen2.5-Coder-7B-Instruct': the
            # served model is the repo, not the org, so split once and
            # take the tail. A bare id with no slash is used directly.
            served_model_name = slug_source.rsplit("/", 1)[-1].strip()
            if not served_model_name:
                raise CogflowValidationError(f"could not derive served_model_name from hf_model_id={hf_model_id!r}")

        if not isvc_name:
            isvc_name = ServingManager._k8s_slugify(served_model_name)

        # DNS-1123 label: regex AND the 63-char length cap. Caller-supplied
        # isvc_name could be regex-valid but too long; ``_k8s_slugify``
        # already truncates, so this only rejects explicit inputs.
        if len(isvc_name) > 63:
            raise CogflowValidationError(
                f"isvc_name={isvc_name!r} exceeds the DNS-1123 label "
                f"length limit of 63 characters (got {len(isvc_name)})"
            )
        if not ServingManager._DNS1123_LABEL_RE.match(isvc_name):
            raise CogflowValidationError(
                f"isvc_name={isvc_name!r} is not a valid DNS-1123 label "
                f"(^[a-z0-9]([-a-z0-9]*[a-z0-9])?$); pass an explicit "
                f"isvc_name or a served_model_name that slugifies cleanly"
            )

        return isvc_name, served_model_name

    @staticmethod
    def _build_llm_args(
        served_model_name: str,
        *,
        max_model_len: int | None,
        dtype: str | None,
        tensor_parallel_size: int | None,
        trust_remote_code: bool,
        gpu_memory_utilization: float | None,
        max_num_seqs: int | None,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        hf_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        """Whitelisted runtime args passed into the HF runtime container.

        Order is stable so the emitted ISVC is diff-friendly across reruns.

        Naming note — the two conventions below are deliberate, not
        accidental. KServe's huggingface runtime uses ``parse_known_args``
        and consumes its own underscore-style flags first, then forwards
        the remainder to the vLLM backend parser (hyphen-style).

        - Underscore flags land in the KServe HF runtime parser
          (``kserve/python/huggingfaceserver``): ``--model_name``,
          ``--max_model_len``, ``--dtype``, ``--trust_remote_code``.
        - Hyphen flags fall through to vLLM's CLI
          (``--tensor-parallel-size``, ``--gpu-memory-utilization``,
          ``--max-num-seqs``, ``--quantization``, ``--kv-cache-dtype``).

        ``quantization`` and ``kv_cache_dtype`` are the runtime-cast knobs
        the recommender's fallback ladder relies on (fp8/AWQ weights and
        fp8 KV cache). Emitted only when set so unrelated callers keep
        emitting diff-clean ISVCs.
        """
        args: list[str] = [f"--model_name={served_model_name}"]
        if max_model_len is not None:
            args.append(f"--max_model_len={max_model_len}")
        if dtype is not None:
            args.append(f"--dtype={dtype}")
        if trust_remote_code:
            args.append("--trust_remote_code")
        if tensor_parallel_size is not None:
            args.append(f"--tensor-parallel-size={tensor_parallel_size}")
        if gpu_memory_utilization is not None:
            args.append(f"--gpu-memory-utilization={gpu_memory_utilization}")
        if max_num_seqs is not None:
            args.append(f"--max-num-seqs={max_num_seqs}")
        # ``"none"`` / ``"auto"`` (and the empty string) are recommender-
        # side sentinels for "vLLM should decide" — they aren't real
        # vLLM CLI values, so treat them like an unset kwarg and emit
        # nothing. Matched case-insensitively and after stripping
        # surrounding whitespace so a slightly noisy config layer
        # (``"AUTO"``, ``" none "``) still produces the same ISVC as an
        # unset field. Applied symmetrically to both flags so the
        # diff-clean guarantee holds regardless of which sentinel the
        # caller picks.
        normalized_quantization = quantization.strip() if quantization is not None else None
        if not _is_unset_sentinel(normalized_quantization):
            args.append(f"--quantization={normalized_quantization}")
        normalized_kv_cache_dtype = kv_cache_dtype.strip() if kv_cache_dtype is not None else None
        if not _is_unset_sentinel(normalized_kv_cache_dtype):
            args.append(f"--kv-cache-dtype={normalized_kv_cache_dtype}")
        # ``--hf-overrides`` is a vLLM hyphen-style flag taking a JSON
        # object that patches the loaded model's HF config. The platform
        # uses it to redirect the architecture to a runtime-registered
        # subclass (e.g. an adapter-aware ``NTK<Arch>ForCausalLM``)
        # without re-uploading the base weights. ``sort_keys`` keeps the
        # emitted JSON byte-stable so reruns produce a diff-clean ISVC.
        # vLLM requires a JSON *object*, so reject non-dicts and
        # non-serializable values up-front with a clear error rather than
        # emitting a bad flag or leaking a raw TypeError.
        if hf_overrides:
            if not isinstance(hf_overrides, dict):
                raise CogflowValidationError(
                    f"hf_overrides must be a JSON object (dict), got {type(hf_overrides).__name__}"
                )
            try:
                rendered = json.dumps(hf_overrides, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise CogflowValidationError(f"hf_overrides must be JSON-serializable: {exc}") from exc
            args.append(f"--hf-overrides={rendered}")
        return args

    @staticmethod
    def _build_llm_predictor(
        *,
        storage_uri: str,
        served_model_name: str,
        max_model_len: int | None,
        dtype: str | None,
        tensor_parallel_size: int | None,
        trust_remote_code: bool,
        gpu_memory_utilization: float | None,
        max_num_seqs: int | None,
        resources: dict[str, dict[str, str]] | None,
        tolerations: list[dict[str, Any]] | None,
        node_selector: dict[str, str] | None,
        min_replicas: int,
        max_replicas: int,
        hf_secret_name: str | None,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        hf_overrides: dict[str, Any] | None = None,
        controller_storage_uri: str | None = None,
    ) -> dict[str, Any]:
        # Replica bounds must be internally consistent before we hand the
        # ISVC to KServe — otherwise the CRD is rejected at admission (or
        # silently misbehaves if admission is permissive). Validate here
        # so both sync (deploy_llm) and async (async_deploy_llm) paths
        # surface a clear CogflowValidationError instead of leaking a
        # K8s API error upstream.
        if min_replicas < 0:
            raise CogflowValidationError(f"min_replicas must be >= 0, got {min_replicas}")
        if max_replicas < 1:
            raise CogflowValidationError(f"max_replicas must be >= 1, got {max_replicas}")
        if min_replicas > max_replicas:
            raise CogflowValidationError(f"min_replicas ({min_replicas}) must be <= max_replicas ({max_replicas})")

        # Source plumbing — HF Hub vs MLflow/MinIO routes through
        # different KServe code paths:
        #
        #  hf://<org>/<model>  →  pass as ``--model_id=<id>`` argv to the
        #                          HF runtime, which downloads it itself.
        #                          Do NOT set ``storageUri``: that would
        #                          route through KServe's storage-
        #                          initializer, which injects
        #                          ``POD_NAME``/``POD_NAMESPACE`` env
        #                          vars via ``valueFrom.fieldRef``. In
        #                          Serverless/Knative deployments those
        #                          env shapes are rejected by the
        #                          Knative admission webhook, so the
        #                          predictor never reconciles.
        #
        #  s3://mlflow/...      →  set ``storageUri``; KServe's storage-
        #                          initializer downloads to a local
        #                          path and passes ``--model_dir=...``
        #                          to the runtime.
        # ``_extract_hf_model_id`` validates the ``hf://`` URI shape and
        # returns ``None`` for non-HF sources. This is also called from
        # ``deploy_llm`` so direct callers of this helper (tests, future
        # SDK users) still get the same validation.
        hf_id = ServingManager._extract_hf_model_id(storage_uri)
        is_hf_source = hf_id is not None

        # A predictor has exactly one ``storageUri`` — one storage-initializer
        # download into ``/mnt/models``. The HF-base hybrid only works because
        # the base loads from the Hub via ``--model_id`` (the HF runtime's
        # ``get_model_id_or_path`` prefers it), leaving that single slot free
        # to stage the controller. With a non-HF base the base already claims
        # the slot, so a ``controller_storage_uri`` would be silently dropped.
        # Fail fast instead.
        if controller_storage_uri and not is_hf_source:
            raise CogflowValidationError(
                "controller_storage_uri is only supported with an hf:// base "
                f"(the base loads via --model_id, freeing the single storageUri "
                f"slot for the controller); got base storage_uri={storage_uri!r}."
            )
        runtime_args = ServingManager._build_llm_args(
            served_model_name,
            max_model_len=max_model_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            hf_overrides=hf_overrides,
        )
        if is_hf_source:
            # --model_id comes first for readability in the emitted YAML
            runtime_args = [f"--model_id={hf_id}", *runtime_args]

        model_block: dict[str, Any] = {
            "modelFormat": {"name": "huggingface"},
            "args": runtime_args,
            "resources": ServingManager._merge_resources(resources),
        }
        if not is_hf_source:
            model_block["storageUri"] = storage_uri
        elif controller_storage_uri:
            # Adapter-on-HF-base hybrid: the base weights load from the
            # HF Hub via ``--model_id`` (above), while a small side
            # artifact — e.g. an ntkmirror controller — is staged by
            # KServe's storage-initializer to the predictor's model
            # mount (``/mnt/models``). The runtime ignores the mount for
            # the base (``--model_id`` wins in the HF server's
            # ``get_model_id_or_path``), and the adapter-aware model
            # subclass reads its artifact from there. Only valid in
            # RawDeployment mode (the LLM default) — the storage-
            # initializer's fieldRef env is rejected by the Knative
            # webhook in Serverless mode.
            model_block["storageUri"] = controller_storage_uri

        if hf_secret_name:
            model_block["env"] = [
                {
                    "name": "HF_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": hf_secret_name, "key": "HF_TOKEN"}},
                }
            ]

        predictor: dict[str, Any] = {
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "model": model_block,
        }

        # S3-backed artifacts need the cluster-provisioned S3 SA. HF Hub
        # pulls do not — omitting the SA keeps the pod token-free. The
        # side controller artifact (``controller_storage_uri``) is also
        # S3-staged, so an HF base + S3 controller still needs the SA.
        if storage_uri.startswith("s3://") or (
            controller_storage_uri is not None and controller_storage_uri.startswith("s3://")
        ):
            predictor["serviceAccountName"] = "kserve-controller-s3"

        if tolerations:
            predictor["tolerations"] = tolerations
        if node_selector:
            predictor["nodeSelector"] = node_selector

        return predictor

    _DEPLOYMENT_MODE_ANNOTATION = "serving.kserve.io/deploymentMode"

    @staticmethod
    def _apply_raw_deployment_defaults(
        predictor: dict[str, Any],
        annotations: dict[str, str] | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Default LLM ISVCs to RawDeployment + Recreate strategy.

        LLM pods are large, GPU-pinned, and almost always run as a single
        replica. Knative/Serverless mode requires the new revision to be
        Ready before the old one is torn down — but on a one-GPU node the
        new pod can never schedule, so the rollout deadlocks. Defaulting
        to RawDeployment with ``strategy.type: Recreate`` makes the old
        pod terminate first, freeing the GPU for the new one. Callers can
        opt out by passing ``serving.kserve.io/deploymentMode`` themselves
        in ``annotations`` — caller wins.

        Returns the (possibly augmented) predictor and the merged
        annotations dict that should land on ``metadata.annotations``.
        """
        effective_annotations: dict[str, Any] = {
            ServingManager._DEPLOYMENT_MODE_ANNOTATION: "RawDeployment",
            **(annotations or {}),
        }
        # ``deploymentStrategy`` is rejected by the KServe webhook for
        # Serverless ISVCs, so only inject it when the resolved mode is
        # RawDeployment.
        if effective_annotations.get(ServingManager._DEPLOYMENT_MODE_ANNOTATION) == "RawDeployment":
            predictor = {**predictor, "deploymentStrategy": {"type": "Recreate"}}
        return predictor, effective_annotations

    def deploy_llm(
        self,
        *,
        storage_uri: str,
        isvc_name: str | None = None,
        served_model_name: str | None = None,
        namespace: str | None = None,
        # vLLM runtime args (whitelist)
        max_model_len: int | None = None,
        dtype: str | None = None,
        tensor_parallel_size: int | None = None,
        trust_remote_code: bool = False,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        hf_overrides: dict[str, Any] | None = None,
        controller_storage_uri: str | None = None,
        # scheduling / scaling
        resources: dict[str, dict[str, str]] | None = None,
        tolerations: list[dict[str, Any]] | None = None,
        node_selector: dict[str, str] | None = None,
        min_replicas: int = 1,
        max_replicas: int = 1,
        # auth
        hf_secret_name: str | None = None,
        annotations: dict[str, str] | None = None,
    ) -> dict:
        """Create a KServe InferenceService backed by the built-in
        ``huggingface`` ClusterServingRuntime (KServe 0.15+).

        ``hf_overrides`` patches the loaded model's HF config (emitted as
        the vLLM ``--hf-overrides`` flag) — used to redirect the
        architecture to a runtime-registered adapter-aware subclass.
        ``controller_storage_uri``, when set alongside an ``hf://`` base,
        stages a side artifact (e.g. an ntkmirror controller) to the
        predictor's model mount while the base still loads from the Hub.

        The ``storage_uri`` is used verbatim — pass ``hf://<org>/<model>``
        for a direct HF Hub pull or ``s3://mlflow/...`` for an MLflow-stored
        HF-format checkpoint. Callers (e.g. Cog-Engine) are responsible for
        resolving the source.

        ``isvc_name`` may be omitted whenever ``served_model_name`` is
        available; it is derived as a DNS-1123-safe slug of that name.
        For the ``hf://`` path, ``served_model_name`` may also be omitted:
        it defaults to the part after the last ``/`` of the HF model id,
        and ``isvc_name`` then derives from that. See
        :meth:`derive_llm_names`. The ``s3://`` / MLflow path still
        requires an explicit ``served_model_name`` (cogflow has no
        catalog visibility).
        """
        # Validate the ``hf://`` URI shape up-front so a malformed input
        # surfaces as "invalid HF model id" regardless of whether the
        # caller omitted names (otherwise the name-derivation step below
        # would raise "served_model_name is required" first and obscure
        # the real problem). ``_build_llm_predictor`` keeps the same
        # check as a defensive guard for direct helper callers.
        hf_model_id_hint = ServingManager._extract_hf_model_id(storage_uri)
        isvc_name, served_model_name = ServingManager.derive_llm_names(
            hf_model_id=hf_model_id_hint,
            served_model_name=served_model_name,
            isvc_name=isvc_name,
        )

        namespace = namespace or common.get_namespace()
        logger.info(
            "Creating LLM InferenceService name=%s namespace=%s storage_uri=%s",
            isvc_name,
            namespace,
            storage_uri,
        )
        # Replica / field validation happens before the try/except so a
        # bad input surfaces the typed CogflowValidationError unchanged,
        # not wrapped as a generic CogflowServingError.
        predictor_spec = ServingManager._build_llm_predictor(
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
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            hf_overrides=hf_overrides,
            controller_storage_uri=controller_storage_uri,
        )
        predictor_spec, effective_annotations = ServingManager._apply_raw_deployment_defaults(
            predictor_spec, annotations
        )

        try:
            metadata = client.V1ObjectMeta(
                name=isvc_name,
                namespace=namespace,
                annotations=effective_annotations,
            )
            body = {
                "apiVersion": f"{self.GROUP}/{self.VERSION}",
                "kind": "InferenceService",
                "metadata": metadata.to_dict(),
                "spec": {"predictor": predictor_spec},
            }

            created = self.api.create_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=self.PLURAL,
                body=body,
            )
            logger.info(
                "LLM InferenceService '%s' created successfully in namespace '%s'.",
                isvc_name,
                namespace,
            )
            return created
        except ApiException as e:
            if e.status == 409:
                CogflowErrorHandler.handle_exception(
                    e,
                    context=(f"LLM InferenceService '{isvc_name}' already exists in namespace '{namespace}'."),
                    raise_as=CogflowValidationError,
                    re_raise=True,
                )
            CogflowErrorHandler.handle_exception(
                e,
                context=(f"Create LLM InferenceService '{isvc_name}' in namespace '{namespace}'"),
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=(f"Create LLM InferenceService '{isvc_name}' in namespace '{namespace}'"),
                raise_as=CogflowServingError,
                re_raise=True,
            )

    def serve_llm(
        self,
        *,
        storage_uri: str | None = None,
        hf_model_id: str | None = None,
        isvc_name: str | None = None,
        served_model_name: str | None = None,
        namespace: str | None = None,
        # vLLM runtime args (whitelist)
        max_model_len: int | None = None,
        dtype: str | None = None,
        tensor_parallel_size: int | None = None,
        trust_remote_code: bool = False,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        hf_overrides: dict[str, Any] | None = None,
        controller_storage_uri: str | None = None,
        # scheduling / scaling
        resources: dict[str, dict[str, str]] | None = None,
        tolerations: list[dict[str, Any]] | None = None,
        node_selector: dict[str, str] | None = None,
        min_replicas: int = 1,
        max_replicas: int = 1,
        # auth
        hf_secret_name: str | None = None,
        annotations: dict[str, str] | None = None,
        # catalog
        user_id: str | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """High-level LLM serving: MLflow-backed catalog registration
        **plus** KServe InferenceService create, in one call.

        ``hf_overrides`` and ``controller_storage_uri`` carry through to
        :meth:`deploy_llm` — see its docstring; they enable serving a
        native adapter (e.g. an ntkmirror controller) on an HF base.

        This is the entry point notebook users should reach for — analogous
        to ``cogflow.log_model`` for classical artifacts. It:

        1. Resolves ``storage_uri`` (``hf://{hf_model_id}`` shorthand is
           accepted) and derives ``isvc_name`` / ``served_model_name`` via
           :meth:`ServingManager.derive_llm_names`.
        2. Opens an MLflow run, tags it with LLM metadata, and posts a
           catalog entry to the CogFlow backend (best-effort, like
           ``log_model``). The MLflow run_id becomes the catalog row's
           primary key — the uniform invariant every other registration
           path already respects, so ``get_run(model_id)`` works on LLM
           rows without any type-branching on the consumer side.
        3. Calls :meth:`deploy_llm` to create the KServe ISVC, injecting
           ``model_id=run_id`` / ``model_type=llm`` / ``hf_model_id``
           annotations on top of whatever the caller passed.

        For the thin, ISVC-only path (no catalog), call :meth:`deploy_llm`
        directly — ``serve_llm`` is strictly additive over it.

        Returns:
            Dict with ``run_id``, ``isvc_name``, ``served_model_name``,
            and ``isvc`` (the KServe custom-object create response). The
            ``run_id`` doubles as the CogFlow ``model_info.id`` for
            subsequent catalog lookups.

        Raises:
            CogflowValidationError: neither ``storage_uri`` nor
                ``hf_model_id`` supplied; malformed ``hf://`` URI;
                invalid derived name.
            CogflowModelError: MLflow run creation failed.
            CogflowServingError / CogflowConnectionError: ISVC create
                failed (same as :meth:`deploy_llm`).
        """
        # --- 1. Resolve storage_uri / hf_model_id into a consistent pair.
        # Accept either the shorthand or the full ``hf://`` URI; derive
        # the other end so downstream steps (tagging, annotations, deploy)
        # see a consistent view regardless of which the caller supplied.
        if storage_uri is None and hf_model_id is None:
            raise CogflowValidationError("serve_llm requires either hf_model_id or storage_uri")
        # Tolerate an ``hf_model_id`` that a caller accidentally prefixed
        # with ``hf://`` (single layer only — ``_extract_hf_model_id``
        # rejects deeper ``hf://hf://…`` nesting at the validator layer
        # below, so there's no need to strip a loop here).
        if hf_model_id is not None and hf_model_id.startswith("hf://"):
            hf_model_id = hf_model_id[len("hf://") :]
        # Reject inconsistent (s3://, file://, …) storage with an
        # ``hf_model_id`` — we'd catalog/tag HuggingFace while deploying
        # a non-HF artifact. MLflow-backed LLMs use ``storage_uri`` alone.
        if storage_uri is not None and not storage_uri.startswith("hf://") and hf_model_id is not None:
            raise CogflowValidationError(
                f"hf_model_id={hf_model_id!r} was supplied alongside a "
                f"non-HF storage_uri={storage_uri!r}; HuggingFace metadata "
                f"only applies to hf:// sources"
            )
        if storage_uri is None:
            # Route a bare ``hf_model_id`` through the same validator /
            # normalizer the ``hf://`` URI path uses so whitespace, stray
            # slashes, and empty input are rejected up-front — *before*
            # we open an MLflow run or POST to the catalog. Previously
            # an invalid id only surfaced deep inside deploy_llm and
            # left an orphan run + catalog entry behind.
            hf_model_id = ServingManager._extract_hf_model_id(f"hf://{hf_model_id}")
            storage_uri = f"hf://{hf_model_id}"
        elif storage_uri.startswith("hf://"):
            # When the caller supplies both, reject mismatches up-front —
            # otherwise we'd catalog/tag one model while deploying
            # another. When only ``storage_uri`` is set, back-derive
            # ``hf_model_id`` so downstream tags/annotations reflect the
            # actual deployed model. Normalize both sides so e.g.
            # ``"Org/Name/"`` compares equal to ``"Org/Name"``.
            extracted = ServingManager._extract_hf_model_id(storage_uri)
            if hf_model_id is None:
                hf_model_id = extracted
            else:
                normalized = ServingManager._extract_hf_model_id(f"hf://{hf_model_id}")
                if normalized != extracted:
                    raise CogflowValidationError(
                        f"hf_model_id={hf_model_id!r} does not match the id "
                        f"encoded in storage_uri={storage_uri!r} "
                        f"(extracted={extracted!r})"
                    )
                hf_model_id = normalized

        # --- 2. Derive names.
        isvc_name, served_model_name = ServingManager.derive_llm_names(
            hf_model_id=hf_model_id,
            served_model_name=served_model_name,
            isvc_name=isvc_name,
        )

        # --- 3. Catalog registration (MLflow run + best-effort POST).
        # Lazy import mirrors the existing pattern in ``_get_model_helpers``
        # / ``_get_dataset_manager`` to avoid a circular import at module
        # load time (serving imports models which imports serving for
        # the LLM serving runtime metadata). Uses the real submodule path
        # rather than ``cogflow.models`` (a _LazyLoader attribute that
        # isn't resolvable via ``from … import …``).
        from cogflow.core import models as cogflow_models

        run_id = cogflow_models.register_llm_catalog_entry(
            served_model_name=served_model_name,
            hf_model_id=hf_model_id,
            user_id=user_id,
            extra_tags=extra_tags,
        )

        # --- 4. Merge the run_id / model_type / hf_model_id annotations
        # onto whatever the caller supplied so consumers (Cog-Engine UI,
        # direct kubectl inspectors) see the catalog link on the ISVC.
        # Identity annotations are authoritative — overwrite any
        # caller-supplied values rather than defaulting — so the ISVC
        # metadata never drifts from the catalog entry. Unrelated
        # caller annotations (``my-custom=…``) still pass through.
        merged_annotations: dict[str, str] = dict(annotations or {})
        merged_annotations["model_type"] = "llm"
        merged_annotations["model_id"] = common.normalize_uuid(run_id)
        if hf_model_id:
            merged_annotations["hf_model_id"] = hf_model_id

        # --- 5. Actually create the ISVC.
        isvc_response = self.deploy_llm(
            storage_uri=storage_uri,
            isvc_name=isvc_name,
            served_model_name=served_model_name,
            namespace=namespace,
            max_model_len=max_model_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            hf_overrides=hf_overrides,
            controller_storage_uri=controller_storage_uri,
            resources=resources,
            tolerations=tolerations,
            node_selector=node_selector,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            hf_secret_name=hf_secret_name,
            annotations=merged_annotations,
        )

        return {
            "run_id": run_id,
            "isvc_name": isvc_name,
            "served_model_name": served_model_name,
            "isvc": isvc_response,
        }

    @staticmethod
    def _process_isvc(isvc: dict) -> dict[str, Any]:
        """
        Process a KServe InferenceService object and extract detailed
        model rollout and canary traffic information.

        Args:
            isvc (dict): Raw InferenceService object from Kubernetes API.

        Returns:
            dict: Processed information with rollout awareness.
        """
        metadata = isvc.get("metadata", {}) or {}
        annotations = metadata.get("annotations", {}) or {}
        status_dict = isvc.get("status", {}) or {}
        spec_dict = isvc.get("spec", {}) or {}

        # --- Identifiers ---
        isvc_name = metadata.get("name", "Unknown")
        model_name = annotations.get("model_name")
        model_id = annotations.get("model_id")
        model_version = annotations.get("model_version")
        dataset_id = annotations.get("dataset_id")
        model_type = annotations.get("model_type")
        creation_timestamp = metadata.get("creationTimestamp")

        # --- Base URLs ---
        served_model_url = (
            status_dict.get("address", {}).get("url")
            or status_dict.get("components", {}).get("predictor", {}).get("url")
            or status_dict.get("url")
            or status_dict.get("components", {}).get("transformer", {}).get("url")
        )

        # --- Status ---
        status = "not_ready"
        for cond in status_dict.get("conditions", []):
            if cond.get("type") == "Ready":
                if cond.get("status") == "True":
                    status = "ready"
                break

        # --- Components (predictor + transformer) ---
        components = status_dict.get("components", {}) or {}
        predictor = components.get("predictor", {}) or {}
        transformer = components.get("transformer", {}) or {}

        predictor_traffic = predictor.get("traffic", []) or []

        # --- Canary detection ---
        canary_spec = spec_dict.get("predictor", {}).get("canary")
        canary_traffic_percent = spec_dict.get("predictor", {}).get("canaryTrafficPercent")
        has_canary = canary_spec is not None or canary_traffic_percent is not None

        # --- Traffic computation ---
        traffic_entries = []

        def _extract_traffic_entries(source_traffic, component_name):
            entries = []
            for item in source_traffic or []:
                entries.append(
                    {
                        "revision": item.get("revisionName"),
                        "percent": item.get("percent", 0),
                        "tag": item.get("tag"),
                        "component": component_name,
                    }
                )
            return entries

        traffic_entries.extend(_extract_traffic_entries(predictor_traffic, "predictor"))
        total_traffic = sum(t["percent"] for t in traffic_entries if t["percent"])

        # --- Determine stable vs canary ---
        stable_revision = None
        canary_revision = None
        stable_traffic = None
        canary_traffic = None

        if has_canary:
            for t in traffic_entries:
                if t["percent"] and t["percent"] < 100:
                    if not stable_revision:
                        stable_revision = t["revision"]
                        stable_traffic = t["percent"]
                if t["percent"] and t["percent"] < 100 and t["tag"] == "canary":
                    canary_revision = t["revision"]
                    canary_traffic = t["percent"]

            # Fallback if only predictor used
            if not canary_revision and len(traffic_entries) == 2:
                canary_revision = traffic_entries[1]["revision"]
                canary_traffic = traffic_entries[1]["percent"]
                stable_revision = traffic_entries[0]["revision"]
                stable_traffic = traffic_entries[0]["percent"]
        else:
            # Single model – no canary
            if traffic_entries:
                stable_revision = traffic_entries[0].get("revision")
                stable_traffic = traffic_entries[0].get("percent", 100)

        # --- Age calculation ---
        if creation_timestamp:
            try:
                creation_time = datetime.strptime(creation_timestamp, "%Y-%m-%dT%H:%M:%SZ")
                age = str(datetime.utcnow() - creation_time).split(".", 1)[0]
            except Exception:
                age = "Unknown"
        else:
            age = "Unknown"

        # --- Compose final object ---
        model_info: dict[str, Any] = {
            "isvc_name": isvc_name,
            "served_model_url": served_model_url,
            "status": status,
            "model_id": model_id or None,
            "model_name": model_name or None,
            "model_version": model_version or None,
            "dataset_id": dataset_id or None,
            "model_type": model_type or None,
            "creation_timestamp": creation_timestamp,
            "age": age,
            "latest_ready_revision": predictor.get("latestReadyRevision") or transformer.get("latestReadyRevision"),
            "traffic_percentage": total_traffic or stable_traffic or 100,
            "has_canary": bool(has_canary),
            # NOTE: these two mirror your original logic
            "stable_revision": canary_revision,
            "canary_revision": stable_revision,
            "stable_traffic_percent": canary_traffic,
            "canary_traffic_percent": stable_traffic,
        }

        return model_info

    # -----------------------------------------------------------------
    # UPDATE SERVED MODEL (high-level)
    # -----------------------------------------------------------------

    def update_model(
        self,
        isvc_name: str,
        model_id: str | None = None,
        artifact_path: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        dataset_id: str | None = None,
        transformer_image: str | None = cog_config.TRANSFORMER_BASE_IMAGE,
        transformer_parameters: dict | None = None,
        protocol_version: str | None = None,
        namespace: str | None = None,
        model_format: str | None = None,
        canary_traffic_percent: int | None = None,
        model_type: str | None = None,
    ) -> str:
        """
        High-level update for an existing InferenceService.

        - If only canary_traffic_percent is provided (no new model):
            -> traffic-only update (promotion/disable).
        - Otherwise:
            -> full model rollout / canary rollout.
        Args:
            isvc_name: InferenceService name.
            model_id: Model identifier.
            artifact_path: Path to model artifacts.
            model_name: Model name.
            model_version: Model version.
            dataset_id: Dataset identifier for transformer parameters.
            transformer_image: Container image for the transformer.
            transformer_parameters: Environment variables for the transformer container.
            protocol_version: Protocol version for the model server.
            namespace: Namespace where the InferenceService is deployed.
            model_format: Model format (e.g., "tensorflow", "pytorch").
            canary_traffic_percent: Traffic percentage to route to canary model.
            model_type: Model type annotation (e.g., "llm", "lora"). Stored as
                an ISVC annotation under the key ``model_type``.
        Returns:
            str: Status message.
        Raises:
            CogflowServingError: If update fails.
        Examples:
            >>> from cogflow import serving
            >>> msg = serving.update_model(
            ...     isvc_name="my-inferenceservice",
            ...     model_id="123e4567-e89b-12d3-a456-426614174000",
            ...     canary_traffic_percent=20,
            ...     namespace="default",
            ... )
            >>> print(msg)
        """
        namespace = namespace or common.get_namespace()
        logger.info(
            "update_served_model called isvc_name=%s namespace=%s model_id=%s model_name=%s "
            "model_version=%s canary_traffic_percent=%s",
            isvc_name,
            namespace,
            model_id,
            model_name,
            model_version,
            canary_traffic_percent,
        )

        try:
            # Fetch existing ISVC
            try:
                isvc = self.get_isvc(isvc_name, namespace)
                logger.debug(
                    "Fetched existing InferenceService '%s' spec: %s",
                    isvc_name,
                    list(isvc.keys()),
                )
            except CogflowConnectionError as err:
                # Already wrapped
                raise err
            except ApiException as e:
                if e.status == 404:
                    CogflowErrorHandler.log_and_raise(
                        f"InferenceService '{isvc_name}' not found in namespace '{namespace}'.",
                        raise_as=CogflowServingError,
                    )
                raise

            # Case 1: Traffic-only update (promotion/disable)
            if canary_traffic_percent is not None and not (model_id or model_name or model_version or artifact_path):
                logger.info(
                    "Applying traffic-only update for InferenceService '%s' to %s%%.",
                    isvc_name,
                    canary_traffic_percent,
                )

                # Validate canary before applying changes
                self._validate_canary(isvc, isvc_name, canary_traffic_percent)

                patch_body = {"spec": {"predictor": {"canaryTrafficPercent": canary_traffic_percent}}}
                self.update_isvc(isvc_name, patch_body, namespace)
                msg = f"Updated traffic to {canary_traffic_percent}% for InferenceService '{isvc_name}'."
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

            transformer_env = self._get_transformer_env(dataset_id, transformer_image, transformer_parameters)

            model_uri = model_details["model_uri"]
            logger.info(
                "Resolved new model_uri=%s for InferenceService '%s'.",
                model_uri,
                isvc_name,
            )

            model_format = model_format or detect_model_format(model_uri=model_uri)

            # Build annotations
            annot = {
                "model_id": model_details["model_id"],
                "model_name": model_details["model_name"],
                "model_version": model_details["model_version"],
            }
            if dataset_id:
                annot["dataset_id"] = common.normalize_uuid(dataset_id)
            if model_type:
                annot["model_type"] = model_type

            # --- Model patch ---
            model_patch: dict[str, Any] = {}
            if model_uri:
                model_patch["storageUri"] = model_uri
            if model_format:
                model_patch["modelFormat"] = {"name": model_format}
            if protocol_version:
                model_patch["protocolVersion"] = protocol_version

            predictor_patch: dict[str, Any] = {"model": model_patch}

            # Canary rollout
            if canary_traffic_percent is not None:
                if not 0 <= canary_traffic_percent <= 100:
                    CogflowErrorHandler.log_and_raise(
                        f"Invalid canary_traffic_percent={canary_traffic_percent}. Must be between 0 and 100.",
                        raise_as=CogflowValidationError,
                    )
                predictor_patch["canary"] = {"model": model_patch}
                predictor_patch["canaryTrafficPercent"] = canary_traffic_percent

            patch_body: dict[str, Any] = {
                "metadata": {"annotations": annot},
                "spec": {"predictor": predictor_patch},
            }

            # Transformer
            if transformer_env:
                env_list = [{"name": k, "value": str(v)} for k, v in transformer_env.items()]
                patch_body["spec"]["transformer"] = {
                    "containers": [
                        {
                            "name": f"{isvc_name}-transformer",
                            "image": transformer_image,
                            "env": env_list,
                        }
                    ]
                }

            logger.info(
                "Patching InferenceService '%s' with new model spec and annotations.",
                isvc_name,
            )
            self.update_isvc(isvc_name, patch_body, namespace)

            msg = f"InferenceService '{isvc_name}' updated successfully."
            logger.info("%s", msg)
            return msg

        except (CogflowValidationError, CogflowModelError, CogflowDatasetError):
            # Already wrapped correctly
            raise
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context="Update served model via ServingManager",
                raise_as=CogflowServingError,
                re_raise=True,
            )

    # -----------------------------------------------------------------
    # SERVE MODEL (high-level)
    # -----------------------------------------------------------------

    def deploy_model(
        self,
        model_id: str = None,
        isvc_name: str | None = None,
        artifact_path: str = None,
        model_name: str = None,
        model_version: str = None,
        dataset_id: str = None,
        transformer_image: str = cog_config.TRANSFORMER_BASE_IMAGE,
        transformer_parameters: dict[str, Any] = None,
        protocol_version: str = None,
        model_format: str = None,
        namespace: str | None = None,
        model_type: str | None = None,
    ):
        """
        High-level entrypoint to serve a model:

        - Resolve model URI / metadata.
        - Resolve dataset-based transformer parameters (Prometheus).
        - Detect model format (if not provided).
        - Create the InferenceService.
        Args:
            model_id: Model identifier.
            isvc_name: InferenceService name.
            artifact_path: Path to model artifacts.
            model_name: Model name.
            model_version: Model version.
            dataset_id: Dataset identifier for transformer parameters.
            transformer_image: Container image for the transformer.
            transformer_parameters: Environment variables for the transformer container.
            protocol_version: Protocol version for the model server.
            model_format: Model format (e.g., "tensorflow", "pytorch").
            namespace: Namespace where the InferenceService will be deployed.
            model_type: Model type annotation (e.g., "llm", "lora"). Stored as
                an ISVC annotation under the key ``model_type`` and returned in
                ``list_models()`` responses.
        Returns:
            dict: Created InferenceService details.
        Raises:
            CogflowServingError: If serving fails.
        Examples:
            >>> from cogflow import serving
            >>> isvc = serving.deploy_model(
            ...     model_id="123e4567-e89b-12d3-a456-426614174000",
            ...     isvc_name="my-inferenceservice",
            ...     namespace="default",
            ... )
            >>> print(isvc)
        """
        namespace = namespace or common.get_namespace()
        logger.info(
            "serve_model called model_id=%s model_name=%s model_version=%s isvc_name=%s namespace=%s",
            model_id,
            model_name,
            model_version,
            isvc_name,
            namespace,
        )

        try:
            detect_model_format, get_model_details = self._get_model_helpers()
            model_details = get_model_details(
                model_id=model_id,
                artifact_path=artifact_path,
                model_name=model_name,
                model_version=model_version,
            )
            model_uri = model_details["model_uri"]

            transformer_env = self._get_transformer_env(dataset_id, transformer_image, transformer_parameters)

            model_format = model_format or detect_model_format(model_uri=model_uri)

            # Default ISVC name: model-<canonical-uuid>
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
                isvc_name,
                model_uri,
                namespace,
            )

            return self.create_isvc(
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
                context="Serve model via ServingManager",
                raise_as=CogflowServingError,
                re_raise=True,
            )

    # -----------------------------------------------------------------
    # LIST / INSPECT SERVED MODELS
    # -----------------------------------------------------------------

    def list_models(
        self,
        namespace: str | None = None,
        isvc_name: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Get served model(s) information from InferenceService CRDs.

        Args:
            namespace: Kubernetes namespace where the InferenceServices are deployed.
                       Defaults to the current namespace (common.get_namespace()).
            isvc_name: Optional name of a single InferenceService. If provided,
                       returns a list with a single model (or None if not found).
                       If omitted, returns all InferenceServices in the namespace.

        Returns:
            list[dict] or None:
                - If isvc_name is provided:
                     * list with a single model_info dict, or
                     * None if the service does not exist.
                - If isvc_name is None:
                     * list of model_info dicts (possibly empty).
                Each dict contains:
                    isvc_name, served_model_url, status, model_id, model_name,
                    model_version, dataset_id, creation_timestamp, age,
                    latest_ready_revision, traffic_percentage,
                    has_canary, stable_revision, canary_revision,
                    stable_traffic_percent, canary_traffic_percent
        Raises:
            CogflowConnectionError: If there is a connection issue with Kubernetes API.
            CogflowServingError: For other serving-related errors.
        Examples:
            >>> from cogflow import serving
            >>> # List all served models in the default namespace
            >>> models = serving.list_models()
            >>> for model in models:
            ...     print(model)
            ...
            >>> # Get a specific served model by InferenceService name
            >>> model = serving.list_models(isvc_name="my-inferenceservice")
            >>> print(model)
        """
        ns = namespace or common.get_namespace()
        logger.info("Fetching served models in namespace=%s (isvc_name=%s)", ns, isvc_name)

        def _get_single_isvc_info(isvc_obj: dict) -> list[dict[str, Any]] | None:
            if not isvc_obj:
                return None
            info = self._process_isvc(isvc_obj)
            return [info] if info else None

        def _get_all_isvc_info(resp: dict) -> list[dict[str, Any]]:
            if isinstance(resp, dict) and "items" in resp:
                isvc_list = resp["items"]
            else:
                isvc_list = []

            served_models: list[dict[str, Any]] = []
            for isvc in isvc_list:
                if isinstance(isvc, dict):
                    info = self._process_isvc(isvc)
                    if info:
                        served_models.append(info)

            served_models.sort(key=lambda x: x.get("creation_timestamp") or "", reverse=True)
            return served_models

        try:
            # Single service
            if isvc_name:
                try:
                    isvc = self.get_isvc(name=isvc_name, namespace=ns)
                except ApiException as e:
                    if getattr(e, "status", None) == 404:
                        logger.info(
                            "InferenceService '%s' not found in namespace '%s'.",
                            isvc_name,
                            ns,
                        )
                        return None
                    # Wrap any other API error
                    CogflowErrorHandler.handle_exception(
                        e,
                        context=(f"Get InferenceService '{isvc_name}' in namespace '{ns}' for get_served_models"),
                        raise_as=CogflowConnectionError,
                        re_raise=True,
                    )
                return _get_single_isvc_info(isvc)

            # All services
            logger.debug(
                "Listing all InferenceServices in namespace=%s for get_served_models",
                ns,
            )
            resp = self.api.list_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=ns,
                plural=self.PLURAL,
            )
            return _get_all_isvc_info(resp)

        except ApiException as e:
            if getattr(e, "status", None) == 404:
                logger.info("No InferenceServices found in namespace '%s' (404 returned).", ns)
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
                context="get_served_models via ServingManager",
                raise_as=CogflowServingError,
                re_raise=True,
            )


# Create a singleton instance for the public interface
_serving = ServingManager()

# Exposed SDK-level methods (single source of truth) — sync
for attr_name in dir(ServingManager):
    # Skip private methods and dunder methods
    if attr_name.startswith("_"):
        continue

    attr = getattr(ServingManager, attr_name)

    # Only export methods (callables) that belong to ServingManager
    if callable(attr):
        # Bind the method to the singleton instance
        globals()[attr_name] = getattr(_serving, attr_name)

# Expose async methods from AsyncServingManager so consumers reaching
# ``cogflow.serving`` (via the package-level _LazyLoader, which maps
# ``cogflow.serving`` → ``cogflow.core.serving``) get the full async
# surface alongside the sync one — without this block, ``from cogflow
# import serving as cogflow_serving; cogflow_serving.async_deploy_llm``
# raises AttributeError. Test
# ``test_serving_module_reexports_async_deploy_llm`` defends this
# contract by running an import in a fresh subprocess.
from .async_serving import (  # noqa: E402, F401
    async_create_isvc,
    async_delete_isvc,
    async_deploy_llm,
    async_deploy_model,
    async_get_isvc,
    async_list_models,
    async_restart_isvc,
    async_serve_llm,
    async_update_isvc,
    async_update_model,
)
