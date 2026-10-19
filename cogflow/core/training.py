"""
CogFlow - Training Manager (Kubeflow Training Operator)
-------------------------------------------------------

Create, inspect and delete Kubeflow Training Operator jobs — ``PyTorchJob``
and ``TFJob`` custom resources (``kubeflow.org/v1``) — from Python.

The manager mirrors :class:`cogflow.core.serving.ServingManager`: it talks
to the cluster through the synchronous ``kubernetes`` client, loads cluster
config lazily (in-cluster first, kubeconfig second) and raises
:class:`~cogflow.utils.exceptions.CogflowConnectionError` when no cluster
is reachable, so importing ``cogflow.training`` never fails on a laptop or
in CI.

Two things in :meth:`TrainingManager.build_job_spec` are load-bearing on a
Kubeflow cluster and are asserted by tests:

- The pod template carries the labels ``access-minio: "true"`` and
  ``mlflow-server-minio: "true"``. Kubeflow's ``PodDefault`` admission
  webhook matches those labels in every profile namespace and injects the
  MinIO credentials and MLflow URIs the trainer needs to log and register
  the model — no secret is ever copied by the caller.
- The pod template carries ``sidecar.istio.io/inject: "false"``. Profile
  namespaces have Istio injection enabled; with a sidecar the training
  container finishes but the pod never completes.

The Training Operator's validating webhook requires the main container to
be named ``pytorch`` (PyTorchJob) or ``tensorflow`` (TFJob); the builder
names it for you.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from kubernetes import client
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException

from ..utils import common
from ..utils.exceptions import (
    CogflowConnectionError,
    CogflowErrorHandler,
    CogflowTrainingError,
    CogflowValidationError,
)
from ..utils.logging import get_logger
from .serving import _ensure_k8s_config_loaded

logger = get_logger(__name__)

# Labels every cogflow-managed job carries. ``MANAGED_LABEL`` is the list
# selector; ``JOB_ID_LABEL`` ties the resource back to the caller's own id.
MANAGED_LABEL = "cogflow.hiro.ai/managed"
JOB_ID_LABEL = "cogflow.hiro.ai/job-id"
FRAMEWORK_LABEL = "cogflow.hiro.ai/framework"

# PodDefault selectors present in every Kubeflow profile namespace.
POD_DEFAULT_LABELS: dict[str, str] = {
    "access-minio": "true",
    "mlflow-server-minio": "true",
}
ISTIO_SIDECAR_ANNOTATION = "sidecar.istio.io/inject"

# Labels the Training Operator stamps on the pods it creates.
OPERATOR_JOB_NAME_LABEL = "training.kubeflow.org/job-name"
OPERATOR_REPLICA_TYPE_LABEL = "training.kubeflow.org/replica-type"

# Job-state vocabulary returned by :func:`parse_job_status`. Order is the
# precedence used when several conditions are ``True`` at once.
STATE_PRECEDENCE: tuple[str, ...] = ("Succeeded", "Failed", "Running", "Restarting", "Created")
TERMINAL_STATES: frozenset[str] = frozenset({"succeeded", "failed"})

GPU_RESOURCE = "nvidia.com/gpu"

# Kubernetes DNS-1035 label (the operator derives Service names from the job name).
DNS1035_NAME_RE = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")


@dataclass(frozen=True)
class FrameworkSpec:
    """Per-framework names the Training Operator cares about."""

    kind: str
    plural: str
    container: str
    primary: str
    replica_specs_key: str


FRAMEWORKS: dict[str, FrameworkSpec] = {
    "pytorch": FrameworkSpec(
        kind="PyTorchJob",
        plural="pytorchjobs",
        container="pytorch",
        primary="Master",
        replica_specs_key="pytorchReplicaSpecs",
    ),
    "tensorflow": FrameworkSpec(
        kind="TFJob",
        plural="tfjobs",
        container="tensorflow",
        primary="Chief",
        replica_specs_key="tfReplicaSpecs",
    ),
}


def framework_spec(framework: str) -> FrameworkSpec:
    """Resolve ``framework`` (case-insensitive) to its :class:`FrameworkSpec`.

    Raises:
        CogflowValidationError: for an unknown framework.
    """
    key = (framework or "").strip().lower()
    spec = FRAMEWORKS.get(key)
    if spec is None:
        CogflowErrorHandler.log_and_raise(
            f"Unsupported training framework {framework!r}. Supported: {sorted(FRAMEWORKS)}.",
            raise_as=CogflowValidationError,
        )
    return spec  # type: ignore[return-value]


def parse_job_status(job: dict[str, Any] | None) -> dict[str, Any]:
    """Reduce a PyTorchJob/TFJob resource to one state plus context.

    Pure function — safe to unit-test without a cluster.

    The operator appends a condition per lifecycle step and may leave
    several with ``status: "True"`` (``Created`` and ``Running`` both stay
    true while the job runs). The returned ``state`` is the highest entry
    of :data:`STATE_PRECEDENCE` whose condition is true, lower-cased;
    ``"unknown"`` when there is no true condition yet.

    Returns:
        dict with keys ``state``, ``reason``, ``message``, ``start_time``,
        ``completion_time``, ``last_transition_time``, ``replica_statuses``.
    """
    status = (job or {}).get("status") or {}
    conditions = status.get("conditions") or []
    true_by_type: dict[str, dict[str, Any]] = {}
    for cond in conditions:
        if str(cond.get("status", "")).lower() == "true" and cond.get("type"):
            true_by_type[cond["type"]] = cond

    chosen: dict[str, Any] | None = None
    for state in STATE_PRECEDENCE:
        if state in true_by_type:
            chosen = true_by_type[state]
            break

    return {
        "state": chosen["type"].lower() if chosen else "unknown",
        "reason": chosen.get("reason") if chosen else None,
        "message": chosen.get("message") if chosen else None,
        "start_time": status.get("startTime"),
        "completion_time": status.get("completionTime"),
        "last_transition_time": chosen.get("lastTransitionTime") if chosen else None,
        "replica_statuses": status.get("replicaStatuses") or {},
    }


def _env_list(env: dict[str, Any] | None) -> list[dict[str, str]]:
    return [{"name": str(k), "value": str(v)} for k, v in sorted((env or {}).items())]


def _resources(cpu: str | None, memory: str | None, gpus: int) -> dict[str, dict[str, str]]:
    requests: dict[str, str] = {}
    limits: dict[str, str] = {}
    if cpu:
        requests["cpu"] = str(cpu)
    if memory:
        requests["memory"] = str(memory)
        limits["memory"] = str(memory)
    if gpus > 0:
        limits[GPU_RESOURCE] = str(gpus)
    out: dict[str, dict[str, str]] = {}
    if requests:
        out["requests"] = requests
    if limits:
        out["limits"] = limits
    return out


class TrainingManager:
    """Manage Kubeflow Training Operator jobs (PyTorchJob / TFJob)."""

    GROUP = "kubeflow.org"
    VERSION = "v1"

    def __init__(self):
        """Initialise the Kubernetes clients if cluster config is available.

        Tolerant of a missing kube config so ``from cogflow import training``
        works without a cluster; cluster calls then raise
        :class:`CogflowConnectionError` through the ``api`` property.
        """
        self._api = None
        self._core_api = None
        try:
            _ensure_k8s_config_loaded()
            self._api = client.CustomObjectsApi()
            self._core_api = client.CoreV1Api()
        except (ConfigException, FileNotFoundError, OSError) as exc:
            logger.warning(
                "Kubernetes config could not be loaded: %s. "
                "TrainingManager cluster operations will fail until config is available.",
                exc,
            )

    @property
    def api(self):
        """Lazily-validated ``CustomObjectsApi``."""
        if self._api is not None:
            return self._api
        try:
            _ensure_k8s_config_loaded()
            self._api = client.CustomObjectsApi()
        except (ConfigException, FileNotFoundError, OSError) as exc:
            raise CogflowConnectionError(f"Kubernetes config is not loaded: {exc}") from exc
        return self._api

    @property
    def core_api(self):
        """Lazily-validated ``CoreV1Api`` (pods and logs)."""
        if self._core_api is not None:
            return self._core_api
        try:
            _ensure_k8s_config_loaded()
            self._core_api = client.CoreV1Api()
        except (ConfigException, FileNotFoundError, OSError) as exc:
            raise CogflowConnectionError(f"Kubernetes config is not loaded: {exc}") from exc
        return self._core_api

    # -----------------------------------------------------------------
    # Spec building
    # -----------------------------------------------------------------

    def build_job_spec(
        self,
        *,
        name: str,
        framework: str,
        image: str,
        command: list[str] | None = None,
        args: list[str] | None = None,
        env: dict[str, Any] | None = None,
        workers: int = 1,
        gpus: int = 0,
        cpu: str | None = None,
        memory: str | None = None,
        tolerations: list[dict[str, Any]] | None = None,
        node_selector: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        job_id: str | None = None,
        ttl_seconds_after_finished: int | None = 86400,
        clean_pod_policy: str = "None",
        backoff_limit: int = 0,
        image_pull_policy: str | None = None,
        service_account_name: str | None = None,
    ) -> dict[str, Any]:
        """Build a ``PyTorchJob`` or ``TFJob`` manifest as a plain dict.

        Args:
            name: Resource name (DNS-1035: lowercase alphanumerics and ``-``,
                at most 63 characters, starting with a letter).
            framework: ``"pytorch"`` or ``"tensorflow"``.
            image: Container image for every replica.
            command / args: Container entrypoint and arguments.
            env: Extra environment variables (MinIO/MLflow values arrive
                via PodDefaults — do not pass credentials here).
            workers: Total replica count. ``1`` means the primary replica
                only; ``n > 1`` adds a ``Worker`` block with ``n - 1``
                replicas beside the ``Master``/``Chief``.
            gpus: ``nvidia.com/gpu`` limit per replica; ``0`` adds no GPU
                request.
            cpu / memory: Optional resource requests (memory is also the
                limit).
            tolerations / node_selector: Pod scheduling hints (e.g. the
                taint on a GPU node).
            labels / annotations: Extra metadata merged onto the job *and*
                its pod template.
            job_id: Caller-side identifier stamped as a label.
            ttl_seconds_after_finished: Operator garbage-collects the job
                (and its pods) this long after it finishes; ``None`` keeps
                it forever.
            clean_pod_policy: ``"None"`` (keep pods for logs), ``"Running"``
                or ``"All"``.
            backoff_limit: Restart budget before the job is marked failed.

        Returns:
            dict: the manifest, ready for :meth:`create_job`.

        Raises:
            CogflowValidationError: on an unknown framework or invalid
                ``workers``/``gpus``/``name``.
        """
        spec = framework_spec(framework)
        if not name or len(name) > 63 or not DNS1035_NAME_RE.fullmatch(name):
            CogflowErrorHandler.log_and_raise(
                f"Invalid job name {name!r}: must match DNS-1035 (lowercase alphanumerics and '-', "
                "starting with a letter, ending with an alphanumeric) and be at most 63 characters.",
                raise_as=CogflowValidationError,
            )
        if workers < 1:
            CogflowErrorHandler.log_and_raise(
                f"workers must be >= 1 (got {workers}).",
                raise_as=CogflowValidationError,
            )
        if gpus < 0:
            CogflowErrorHandler.log_and_raise(
                f"gpus must be >= 0 (got {gpus}).",
                raise_as=CogflowValidationError,
            )

        job_labels: dict[str, str] = {
            MANAGED_LABEL: "true",
            FRAMEWORK_LABEL: spec.container,
        }
        if job_id:
            job_labels[JOB_ID_LABEL] = str(job_id)
        job_labels.update(labels or {})

        pod_labels = dict(POD_DEFAULT_LABELS)
        pod_labels.update(job_labels)
        pod_annotations = {ISTIO_SIDECAR_ANNOTATION: "false"}
        pod_annotations.update(annotations or {})

        container: dict[str, Any] = {
            "name": spec.container,
            "image": image,
        }
        if command:
            container["command"] = list(command)
        if args:
            container["args"] = list(args)
        env_list = _env_list(env)
        if env_list:
            container["env"] = env_list
        resources = _resources(cpu, memory, gpus)
        if resources:
            container["resources"] = resources
        if image_pull_policy:
            container["imagePullPolicy"] = image_pull_policy

        pod_spec: dict[str, Any] = {"containers": [container]}
        if tolerations:
            pod_spec["tolerations"] = copy.deepcopy(tolerations)
        if node_selector:
            pod_spec["nodeSelector"] = dict(node_selector)
        if service_account_name:
            pod_spec["serviceAccountName"] = service_account_name

        template = {
            "metadata": {"labels": pod_labels, "annotations": pod_annotations},
            "spec": pod_spec,
        }

        def _replica(count: int) -> dict[str, Any]:
            return {
                "replicas": count,
                "restartPolicy": "OnFailure",
                "template": copy.deepcopy(template),
            }

        replica_specs: dict[str, Any] = {spec.primary: _replica(1)}
        if workers > 1:
            replica_specs["Worker"] = _replica(workers - 1)

        run_policy: dict[str, Any] = {
            "cleanPodPolicy": clean_pod_policy,
            "backoffLimit": backoff_limit,
        }
        if ttl_seconds_after_finished is not None:
            run_policy["ttlSecondsAfterFinished"] = int(ttl_seconds_after_finished)

        manifest = {
            "apiVersion": f"{self.GROUP}/{self.VERSION}",
            "kind": spec.kind,
            "metadata": {"name": name, "labels": job_labels, "annotations": dict(annotations or {})},
            "spec": {
                "runPolicy": run_policy,
                spec.replica_specs_key: replica_specs,
            },
        }
        return manifest

    # -----------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------

    @staticmethod
    def _framework_of(manifest: dict[str, Any]) -> FrameworkSpec:
        kind = (manifest.get("kind") or "").strip()
        for spec in FRAMEWORKS.values():
            if spec.kind == kind:
                return spec
        CogflowErrorHandler.log_and_raise(
            f"Manifest kind {kind!r} is not a supported training job (PyTorchJob, TFJob).",
            raise_as=CogflowValidationError,
        )
        raise AssertionError("unreachable")  # pragma: no cover

    def create_job(self, manifest: dict[str, Any], namespace: str | None = None) -> dict[str, Any]:
        """Create the job described by ``manifest`` (from :meth:`build_job_spec`).

        Returns:
            dict: the created resource as returned by the API server.

        Raises:
            CogflowValidationError: if the manifest kind is not a training job.
            CogflowTrainingError: if the API server rejects the resource
                (409 already exists, 422 webhook validation, ...).
            CogflowConnectionError: on any other Kubernetes API failure.
        """
        spec = self._framework_of(manifest)
        namespace = namespace or common.get_namespace()
        name = manifest.get("metadata", {}).get("name")
        logger.info("Creating %s name=%s namespace=%s", spec.kind, name, namespace)
        try:
            return self.api.create_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=spec.plural,
                body=manifest,
            )
        except ApiException as e:
            if e.status in (400, 409, 422):
                CogflowErrorHandler.log_and_raise(
                    f"{spec.kind} '{name}' was rejected by the API server ({e.status}): {e.reason} {e.body or ''}".strip(),
                    raise_as=CogflowTrainingError,
                    cause=e,
                )
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Create {spec.kind} '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
            raise  # pragma: no cover - handle_exception re-raised already

    def get_job(self, name: str, framework: str, namespace: str | None = None) -> dict[str, Any] | None:
        """Fetch a job by name.

        Returns:
            The resource dict, or ``None`` when it does not exist (deleted,
            or garbage-collected after ``ttlSecondsAfterFinished``).
        """
        spec = framework_spec(framework)
        namespace = namespace or common.get_namespace()
        try:
            return self.api.get_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=spec.plural,
                name=name,
            )
        except ApiException as e:
            if e.status == 404:
                logger.info("%s '%s' not found in namespace '%s'.", spec.kind, name, namespace)
                return None
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Get {spec.kind} '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
            raise  # pragma: no cover - handle_exception re-raised already

    def get_job_status(self, name: str, framework: str, namespace: str | None = None) -> dict[str, Any]:
        """:func:`parse_job_status` of :meth:`get_job`; ``state == "missing"`` when the resource is gone."""
        job = self.get_job(name, framework, namespace)
        if job is None:
            return {
                "state": "missing",
                "reason": "NotFound",
                "message": "job resource no longer exists",
                "start_time": None,
                "completion_time": None,
                "last_transition_time": None,
                "replica_statuses": {},
            }
        return parse_job_status(job)

    def list_jobs(
        self,
        framework: str,
        namespace: str | None = None,
        label_selector: str | None = f"{MANAGED_LABEL}=true",
    ) -> list[dict[str, Any]]:
        """List jobs of one framework in a namespace (cogflow-managed ones by default)."""
        spec = framework_spec(framework)
        namespace = namespace or common.get_namespace()
        kwargs: dict[str, Any] = {
            "group": self.GROUP,
            "version": self.VERSION,
            "namespace": namespace,
            "plural": spec.plural,
        }
        if label_selector:
            kwargs["label_selector"] = label_selector
        try:
            result = self.api.list_namespaced_custom_object(**kwargs)
            return list(result.get("items") or [])
        except ApiException as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"List {spec.kind}s in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
            raise  # pragma: no cover - handle_exception re-raised already

    def delete_job(self, name: str, framework: str, namespace: str | None = None) -> bool:
        """Delete a job (and, through the operator, its pods).

        Returns:
            ``True`` when deleted, ``False`` when it did not exist.
        """
        spec = framework_spec(framework)
        namespace = namespace or common.get_namespace()
        logger.info("Deleting %s name=%s namespace=%s", spec.kind, name, namespace)
        try:
            self.api.delete_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural=spec.plural,
                name=name,
            )
            return True
        except ApiException as e:
            if e.status == 404:
                logger.warning("%s '%s' not found in namespace '%s' during delete.", spec.kind, name, namespace)
                return False
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Delete {spec.kind} '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
            raise  # pragma: no cover - handle_exception re-raised already

    # -----------------------------------------------------------------
    # Pods and logs
    # -----------------------------------------------------------------

    def list_job_pods(
        self,
        name: str,
        framework: str,
        namespace: str | None = None,
        replica_type: str | None = None,
    ) -> list[Any]:
        """Pods the operator created for ``name`` (optionally one replica type)."""
        spec = framework_spec(framework)
        namespace = namespace or common.get_namespace()
        replica = (replica_type or spec.primary).lower()
        selector = f"{OPERATOR_JOB_NAME_LABEL}={name},{OPERATOR_REPLICA_TYPE_LABEL}={replica}"
        try:
            pods = self.core_api.list_namespaced_pod(namespace=namespace, label_selector=selector)
            return list(pods.items or [])
        except ApiException as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"List pods for {spec.kind} '{name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
            raise  # pragma: no cover - handle_exception re-raised already

    def get_job_logs(
        self,
        name: str,
        framework: str,
        namespace: str | None = None,
        replica_type: str | None = None,
        tail_lines: int | None = 500,
    ) -> list[str]:
        """Log lines of the job's primary replica (or ``replica_type``).

        Returns:
            list[str]: non-empty log lines; ``[]`` when no pod exists yet.

        Raises:
            CogflowTrainingError: when the pod exists but its log cannot be
                read (container still creating, image pull failure, ...).
            CogflowConnectionError: on other Kubernetes API failures.
        """
        spec = framework_spec(framework)
        namespace = namespace or common.get_namespace()
        pods = self.list_job_pods(name, framework, namespace, replica_type)
        if not pods:
            logger.info("No %s pods yet for job '%s' in namespace '%s'.", spec.kind, name, namespace)
            return []
        # Index 0 is the lowest ordinal (``<name>-master-0``); the operator
        # names pods deterministically so the sort is stable.
        pod = sorted(pods, key=lambda p: p.metadata.name)[0]
        kwargs: dict[str, Any] = {
            "name": pod.metadata.name,
            "namespace": namespace,
            "container": spec.container,
        }
        if tail_lines is not None:
            kwargs["tail_lines"] = int(tail_lines)
        try:
            raw = self.core_api.read_namespaced_pod_log(**kwargs)
        except ApiException as e:
            if e.status in (400, 404):
                CogflowErrorHandler.log_and_raise(
                    f"Logs for pod '{pod.metadata.name}' are not available yet ({e.status}): {e.reason}",
                    raise_as=CogflowTrainingError,
                    cause=e,
                )
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Read logs of pod '{pod.metadata.name}' in namespace '{namespace}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )
            raise  # pragma: no cover - handle_exception re-raised already
        return [line.rstrip() for line in (raw or "").split("\n") if line.strip()]


# Singleton + module-level exports, same shape as ``cogflow.core.serving``.
_training = TrainingManager()

for attr_name in dir(TrainingManager):
    if attr_name.startswith("_"):
        continue
    attr = getattr(TrainingManager, attr_name)
    if callable(attr):
        globals()[attr_name] = getattr(_training, attr_name)
