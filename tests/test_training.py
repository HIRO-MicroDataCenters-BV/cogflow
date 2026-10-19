"""
Unit tests for cogflow.core.training.TrainingManager.

All cluster access is faked: CustomObjectsApi (jobs), CoreV1Api (pods and
logs), the k8s config loader and namespace resolution.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

from cogflow.utils import common
from cogflow.utils import exceptions as exc_mod

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


class FakeCustomObjectsApi:
    """Records calls; behaviour is driven by per-test attributes."""

    def __init__(self):
        self.calls = []
        self.get_response = None
        self.get_error: ApiException | None = None
        self.create_error: ApiException | None = None
        self.delete_error: ApiException | None = None
        self.list_items: list[dict] = []

    def create_namespaced_custom_object(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.create_error is not None:
            raise self.create_error
        return {"metadata": {"name": kwargs["body"]["metadata"]["name"]}}

    def get_namespaced_custom_object(self, **kwargs):
        self.calls.append(("get", kwargs))
        if self.get_error is not None:
            raise self.get_error
        return self.get_response or {"metadata": {"name": kwargs["name"]}, "status": {}}

    def list_namespaced_custom_object(self, **kwargs):
        self.calls.append(("list", kwargs))
        return {"items": self.list_items}

    def delete_namespaced_custom_object(self, **kwargs):
        self.calls.append(("delete", kwargs))
        if self.delete_error is not None:
            raise self.delete_error
        return {}


class FakeCoreV1Api:
    def __init__(self):
        self.calls = []
        self.pods: list[SimpleNamespace] = []
        self.log_text = "line one\n\nline two\n"
        self.log_error: ApiException | None = None

    def list_namespaced_pod(self, **kwargs):
        self.calls.append(("list_pods", kwargs))
        return SimpleNamespace(items=self.pods)

    def read_namespaced_pod_log(self, **kwargs):
        self.calls.append(("read_log", kwargs))
        if self.log_error is not None:
            raise self.log_error
        return self.log_text


@pytest.fixture
def training_env(monkeypatch):
    if hasattr(common, "_k8s_loaded_flag"):
        monkeypatch.delattr(common, "_k8s_loaded_flag", raising=False)
    monkeypatch.setattr(common, "load_k8s_config", lambda: None, raising=True)
    monkeypatch.setattr(common, "get_namespace", lambda: "test-namespace", raising=True)

    import cogflow.core.training as training_module

    fake_api = FakeCustomObjectsApi()
    fake_core = FakeCoreV1Api()
    monkeypatch.setattr(training_module.client, "CustomObjectsApi", lambda: fake_api, raising=True)
    monkeypatch.setattr(training_module.client, "CoreV1Api", lambda: fake_core, raising=True)
    return training_module, fake_api, fake_core


@pytest.fixture
def training(training_env):
    module, _, _ = training_env
    return module.TrainingManager()


def _pod(name: str) -> SimpleNamespace:
    return SimpleNamespace(metadata=SimpleNamespace(name=name))


# ---------------------------------------------------------------------
# build_job_spec
# ---------------------------------------------------------------------


def test_pytorch_spec_shape(training):
    spec = training.build_job_spec(
        name="tj-abc12345",
        framework="pytorch",
        image="hiroregistry/cogflow:latest-beta-torch",
        command=["python", "-m", "cogflow.trainers.pytorch.tabular"],
        args=["--task", "classification"],
        env={"API_PATH": "http://cog-api-dev.kubeflow/apidev", "COG_TRAINING_JOB_ID": "abc"},
        job_id="abc",
    )
    assert spec["apiVersion"] == "kubeflow.org/v1"
    assert spec["kind"] == "PyTorchJob"
    assert spec["metadata"]["name"] == "tj-abc12345"
    assert spec["metadata"]["labels"]["cogflow.hiro.ai/managed"] == "true"
    assert spec["metadata"]["labels"]["cogflow.hiro.ai/job-id"] == "abc"

    replicas = spec["spec"]["pytorchReplicaSpecs"]
    assert set(replicas) == {"Master"}, "workers=1 must not emit a Worker block"
    master = replicas["Master"]
    assert master["replicas"] == 1
    assert master["restartPolicy"] == "OnFailure"

    template = master["template"]
    # PodDefault selectors: this is how the pod gets MinIO creds + MLflow URIs.
    assert template["metadata"]["labels"]["access-minio"] == "true"
    assert template["metadata"]["labels"]["mlflow-server-minio"] == "true"
    # Istio sidecar must be off or the pod never completes.
    assert template["metadata"]["annotations"]["sidecar.istio.io/inject"] == "false"

    container = template["spec"]["containers"][0]
    assert container["name"] == "pytorch", "operator webhook requires the container name"
    assert container["image"] == "hiroregistry/cogflow:latest-beta-torch"
    assert container["command"] == ["python", "-m", "cogflow.trainers.pytorch.tabular"]
    assert container["args"] == ["--task", "classification"]
    assert {"name": "API_PATH", "value": "http://cog-api-dev.kubeflow/apidev"} in container["env"]
    assert "resources" not in container, "gpus=0 must not request a GPU"

    run_policy = spec["spec"]["runPolicy"]
    assert run_policy["cleanPodPolicy"] == "None"
    assert run_policy["backoffLimit"] == 0
    assert run_policy["ttlSecondsAfterFinished"] == 86400


def test_tensorflow_spec_uses_chief_and_tensorflow_container(training):
    spec = training.build_job_spec(name="tj-tf", framework="tensorflow", image="img", workers=3)
    assert spec["kind"] == "TFJob"
    replicas = spec["spec"]["tfReplicaSpecs"]
    assert set(replicas) == {"Chief", "Worker"}
    assert replicas["Chief"]["replicas"] == 1
    assert replicas["Worker"]["replicas"] == 2
    for block in replicas.values():
        assert block["template"]["spec"]["containers"][0]["name"] == "tensorflow"


def test_workers_add_worker_block_for_pytorch(training):
    spec = training.build_job_spec(name="tj-w", framework="pytorch", image="img", workers=4)
    replicas = spec["spec"]["pytorchReplicaSpecs"]
    assert replicas["Master"]["replicas"] == 1
    assert replicas["Worker"]["replicas"] == 3
    # Worker template is a deep copy, not the same object.
    assert replicas["Worker"]["template"] is not replicas["Master"]["template"]
    assert replicas["Worker"]["template"] == replicas["Master"]["template"]


def test_gpu_resources_tolerations_and_selector(training):
    tol = [{"key": "storage-type", "operator": "Equal", "value": "local", "effect": "NoSchedule"}]
    spec = training.build_job_spec(
        name="tj-gpu",
        framework="pytorch",
        image="img",
        gpus=2,
        cpu="2",
        memory="8Gi",
        tolerations=tol,
        node_selector={"gpu": "l40"},
    )
    pod = spec["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert container["resources"]["limits"]["memory"] == "8Gi"
    assert container["resources"]["requests"] == {"cpu": "2", "memory": "8Gi"}
    assert pod["tolerations"] == tol
    assert pod["tolerations"] is not tol
    assert pod["nodeSelector"] == {"gpu": "l40"}


def test_ttl_none_omits_key_and_custom_policy(training):
    spec = training.build_job_spec(
        name="tj-x", framework="pytorch", image="img", ttl_seconds_after_finished=None, clean_pod_policy="All"
    )
    assert "ttlSecondsAfterFinished" not in spec["spec"]["runPolicy"]
    assert spec["spec"]["runPolicy"]["cleanPodPolicy"] == "All"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"framework": "mxnet"},
        {"workers": 0},
        {"gpus": -1},
        {"name": "Upper-Case"},
        {"name": "x" * 64},
        {"name": ""},
    ],
)
def test_build_job_spec_validation(training, kwargs):
    base = {"name": "tj-ok", "framework": "pytorch", "image": "img"}
    base.update(kwargs)
    with pytest.raises(exc_mod.CogflowValidationError):
        training.build_job_spec(**base)


def test_extra_labels_and_annotations_land_on_job_and_pod(training):
    spec = training.build_job_spec(
        name="tj-l", framework="pytorch", image="img", labels={"team": "ml"}, annotations={"note": "x"}
    )
    assert spec["metadata"]["labels"]["team"] == "ml"
    assert spec["metadata"]["annotations"] == {"note": "x"}
    template = spec["spec"]["pytorchReplicaSpecs"]["Master"]["template"]
    assert template["metadata"]["labels"]["team"] == "ml"
    assert template["metadata"]["annotations"]["note"] == "x"
    assert template["metadata"]["annotations"]["sidecar.istio.io/inject"] == "false"


# ---------------------------------------------------------------------
# parse_job_status
# ---------------------------------------------------------------------


def _job_with(conditions, **status):
    return {"status": {"conditions": conditions, **status}}


def test_parse_status_prefers_terminal_states(training_env):
    module, _, _ = training_env
    job = _job_with(
        [
            {"type": "Created", "status": "True"},
            {"type": "Running", "status": "True", "message": "running"},
            {"type": "Succeeded", "status": "True", "reason": "JobSucceeded", "lastTransitionTime": "t3"},
        ],
        startTime="t1",
        completionTime="t3",
        replicaStatuses={"Master": {"succeeded": 1}},
    )
    parsed = module.parse_job_status(job)
    assert parsed["state"] == "succeeded"
    assert parsed["reason"] == "JobSucceeded"
    assert parsed["start_time"] == "t1"
    assert parsed["completion_time"] == "t3"
    assert parsed["last_transition_time"] == "t3"
    assert parsed["replica_statuses"] == {"Master": {"succeeded": 1}}


def test_parse_status_failed_beats_running(training_env):
    module, _, _ = training_env
    job = _job_with(
        [
            {"type": "Running", "status": "True"},
            {"type": "Failed", "status": "True", "message": "boom"},
        ]
    )
    parsed = module.parse_job_status(job)
    assert parsed["state"] == "failed"
    assert parsed["message"] == "boom"


def test_parse_status_running_when_created_and_running_true(training_env):
    module, _, _ = training_env
    job = _job_with([{"type": "Created", "status": "True"}, {"type": "Running", "status": "True"}])
    assert module.parse_job_status(job)["state"] == "running"


def test_parse_status_ignores_false_conditions(training_env):
    module, _, _ = training_env
    job = _job_with([{"type": "Created", "status": "True"}, {"type": "Running", "status": "False"}])
    assert module.parse_job_status(job)["state"] == "created"


def test_parse_status_unknown_without_conditions(training_env):
    module, _, _ = training_env
    assert module.parse_job_status({})["state"] == "unknown"
    assert module.parse_job_status(None)["state"] == "unknown"


# ---------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------


def test_create_job_targets_right_plural_and_namespace(training, training_env):
    _, fake_api, _ = training_env
    spec = training.build_job_spec(name="tj-c", framework="tensorflow", image="img")
    training.create_job(spec, namespace="user-ns")
    kind, kwargs = fake_api.calls[-1]
    assert kind == "create"
    assert kwargs["plural"] == "tfjobs"
    assert kwargs["group"] == "kubeflow.org"
    assert kwargs["version"] == "v1"
    assert kwargs["namespace"] == "user-ns"
    assert kwargs["body"] is spec


def test_create_job_defaults_namespace(training, training_env):
    _, fake_api, _ = training_env
    spec = training.build_job_spec(name="tj-c", framework="pytorch", image="img")
    training.create_job(spec)
    assert fake_api.calls[-1][1]["namespace"] == "test-namespace"
    assert fake_api.calls[-1][1]["plural"] == "pytorchjobs"


def test_create_job_rejects_non_training_kind(training):
    with pytest.raises(exc_mod.CogflowValidationError):
        training.create_job({"kind": "InferenceService", "metadata": {"name": "x"}})


def test_create_job_conflict_is_training_error(training, training_env):
    _, fake_api, _ = training_env
    fake_api.create_error = ApiException(status=409, reason="AlreadyExists")
    spec = training.build_job_spec(name="tj-dup", framework="pytorch", image="img")
    with pytest.raises(exc_mod.CogflowTrainingError):
        training.create_job(spec)


def test_create_job_server_error_is_connection_error(training, training_env):
    _, fake_api, _ = training_env
    fake_api.create_error = ApiException(status=503, reason="Unavailable")
    spec = training.build_job_spec(name="tj-503", framework="pytorch", image="img")
    with pytest.raises(exc_mod.CogflowConnectionError):
        training.create_job(spec)


def test_get_job_returns_none_on_404(training, training_env):
    _, fake_api, _ = training_env
    fake_api.get_error = ApiException(status=404, reason="Not Found")
    assert training.get_job("tj-gone", "pytorch", "ns") is None
    assert training.get_job_status("tj-gone", "pytorch", "ns")["state"] == "missing"


def test_get_job_status_parses_resource(training, training_env):
    _, fake_api, _ = training_env
    fake_api.get_response = _job_with([{"type": "Running", "status": "True"}])
    status = training.get_job_status("tj-run", "tensorflow", "ns")
    assert status["state"] == "running"
    assert fake_api.calls[-1][1]["plural"] == "tfjobs"


def test_list_jobs_uses_managed_selector(training, training_env):
    _, fake_api, _ = training_env
    fake_api.list_items = [{"metadata": {"name": "a"}}]
    items = training.list_jobs("pytorch", namespace="ns")
    assert items == [{"metadata": {"name": "a"}}]
    assert fake_api.calls[-1][1]["label_selector"] == "cogflow.hiro.ai/managed=true"


def test_delete_job_true_then_false_on_404(training, training_env):
    _, fake_api, _ = training_env
    assert training.delete_job("tj-d", "pytorch", "ns") is True
    fake_api.delete_error = ApiException(status=404, reason="Not Found")
    assert training.delete_job("tj-d", "pytorch", "ns") is False


# ---------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------


def test_get_job_logs_selects_primary_pod_and_container(training, training_env):
    _, _, fake_core = training_env
    fake_core.pods = [_pod("tj-l-master-0")]
    lines = training.get_job_logs("tj-l", "pytorch", namespace="ns", tail_lines=50)
    assert lines == ["line one", "line two"]
    list_call = fake_core.calls[0][1]
    assert list_call["namespace"] == "ns"
    assert (
        list_call["label_selector"] == "training.kubeflow.org/job-name=tj-l,training.kubeflow.org/replica-type=master"
    )
    read_call = fake_core.calls[1][1]
    assert read_call == {"name": "tj-l-master-0", "namespace": "ns", "container": "pytorch", "tail_lines": 50}


def test_get_job_logs_tensorflow_defaults_to_chief(training, training_env):
    _, _, fake_core = training_env
    fake_core.pods = [_pod("tj-t-chief-0")]
    training.get_job_logs("tj-t", "tensorflow", namespace="ns")
    assert fake_core.calls[0][1]["label_selector"].endswith("replica-type=chief")
    assert fake_core.calls[1][1]["container"] == "tensorflow"


def test_get_job_logs_explicit_replica_and_lowest_ordinal(training, training_env):
    _, _, fake_core = training_env
    fake_core.pods = [_pod("tj-w-worker-1"), _pod("tj-w-worker-0")]
    training.get_job_logs("tj-w", "pytorch", namespace="ns", replica_type="Worker")
    assert fake_core.calls[0][1]["label_selector"].endswith("replica-type=worker")
    assert fake_core.calls[1][1]["name"] == "tj-w-worker-0"


def test_get_job_logs_no_pods_returns_empty(training, training_env):
    _, _, fake_core = training_env
    assert training.get_job_logs("tj-none", "pytorch", namespace="ns") == []
    assert len(fake_core.calls) == 1


def test_get_job_logs_not_ready_is_training_error(training, training_env):
    _, _, fake_core = training_env
    fake_core.pods = [_pod("tj-p-master-0")]
    fake_core.log_error = ApiException(status=400, reason="ContainerCreating")
    with pytest.raises(exc_mod.CogflowTrainingError):
        training.get_job_logs("tj-p", "pytorch", namespace="ns")


# ---------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------


def test_module_exports_bound_methods(training_env):
    module, _, _ = training_env
    for name in (
        "build_job_spec",
        "create_job",
        "get_job",
        "get_job_status",
        "list_jobs",
        "delete_job",
        "get_job_logs",
    ):
        assert callable(getattr(module, name)), name
    assert callable(module.parse_job_status)


def test_lazy_package_lists_training(training_env):
    """``cogflow.training`` must be reachable through the package's lazy loader.

    Read from the package source: under pytest the repo root's empty
    ``__init__.py`` shadows ``import cogflow`` with the repo directory.
    """
    import pathlib

    module, _, _ = training_env
    package_init = pathlib.Path(module.__file__).resolve().parents[1] / "__init__.py"
    assert '"training",' in package_init.read_text()
