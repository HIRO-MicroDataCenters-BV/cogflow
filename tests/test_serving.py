"""
Unit tests for cogflow.core.serving.ServingManager.

All external dependencies are mocked:
- Kubernetes CustomObjectsApi
- K8s config loader (common.load_k8s_config)
- Namespace resolution (common.get_namespace)
- Model helper functions (via _get_model_helpers)
- Transformer env resolution (via _get_transformer_env)
"""

import pytest
from cogflow.utils import common

# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------


@pytest.fixture
def serving_module(monkeypatch):
    """
    Import cogflow.core.serving with heavy dependencies patched so
    we never talk to a real Kubernetes cluster or real config.
    """
    # Reset any custom flag we might have added on common in other tests
    if hasattr(common, "_k8s_loaded_flag"):
        del common._k8s_loaded_flag

    # Track how many times config is loaded
    load_calls = {"count": 0}

    def fake_load_k8s_config():
        load_calls["count"] += 1

    def fake_get_namespace():
        return "test-namespace"

    # Patch common BEFORE importing serving module
    monkeypatch.setattr(common, "load_k8s_config", fake_load_k8s_config, raising=True)
    monkeypatch.setattr(common, "get_namespace", fake_get_namespace, raising=True)

    # Fake Kubernetes API client
    class FakeCustomObjectsApi:
        """A fake CustomObjectsApi that records calls made to it."""

        def __init__(self):
            self.calls = []

        def get_namespaced_custom_object(self, **kwargs):
            """Simulate retrieval of a custom object."""
            self.calls.append(("get", kwargs))
            return {
                "metadata": {
                    "name": kwargs["name"],
                    "annotations": {},
                    "creationTimestamp": "2024-01-01T00:00:00Z",
                },
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "components": {"predictor": {"traffic": []}},
                },
                "spec": {"predictor": {}},
            }

        def delete_namespaced_custom_object(self, **kwargs):
            """Simulate deletion of a custom object."""
            self.calls.append(("delete", kwargs))
            return {}

        def patch_namespaced_custom_object(self, **kwargs):
            """Simulate patching a custom object."""
            self.calls.append(("patch", kwargs))
            return {"metadata": {"name": kwargs["name"]}}

        def list_namespaced_custom_object(self, **kwargs):
            """Simulate listing custom objects."""
            self.calls.append(("list", kwargs))
            return {
                "items": [
                    {
                        "metadata": {
                            "name": "isvc-a",
                            "annotations": {"model_name": "m1"},
                            "creationTimestamp": "2024-01-01T00:00:00Z",
                        },
                        "status": {
                            "conditions": [{"type": "Ready", "status": "True"}],
                            "components": {"predictor": {"traffic": []}},
                        },
                        "spec": {"predictor": {}},
                    }
                ]
            }

        def create_namespaced_custom_object(self, **kwargs):
            """Simulate creation of a custom object."""
            self.calls.append(("create", kwargs))
            return {
                "metadata": {
                    "name": kwargs["body"]["metadata"]["name"],
                }
            }

    fake_api = FakeCustomObjectsApi()

    # Import serving module and patch its client.CustomObjectsApi
    import cogflow.core.serving as serving_module

    monkeypatch.setattr(
        serving_module.client,
        "CustomObjectsApi",
        lambda: fake_api,
        raising=True,
    )

    # Ensure its internal "_k8s_config_loaded" doesn't block our fake load
    monkeypatch.setattr(serving_module, "_k8s_config_loaded", False, raising=False)

    return serving_module, fake_api, load_calls


@pytest.fixture
def serving(serving_module):
    """
    Returns a fresh ServingManager instance for tests that need it.
    """
    serving_module_obj, _, _ = serving_module
    return serving_module_obj.ServingManager()


# ---------------------------------------------------------------------
# BASIC INIT / CONFIG BEHAVIOR
# ---------------------------------------------------------------------


def test_init_loads_k8s_config_once(monkeypatch):
    """
    Verify ServingManager loads Kubernetes config exactly once, even if
    instantiated multiple times.
    """
    load_calls = {"count": 0}

    def mock_load():
        load_calls["count"] += 1

    # Patch common helpers
    monkeypatch.setattr(common, "load_k8s_config", mock_load, raising=True)
    monkeypatch.setattr(common, "get_namespace", lambda: "ns", raising=True)

    # Reset the loaded flag set by previous serving module imports
    if hasattr(common, "_k8s_loaded_flag"):
        monkeypatch.delattr(common, "_k8s_loaded_flag", raising=False)

    # Import serving module after patching
    import cogflow.core.serving as serving_module

    # Reset internal flag so ServingManager will call load_k8s_config again
    monkeypatch.setattr(serving_module, "_k8s_config_loaded", False, raising=False)

    # Avoid real K8s client
    class DummyApi:
        """Dummy CustomObjectsApi that does nothing."""

        pass

    monkeypatch.setattr(
        serving_module.client,
        "CustomObjectsApi",
        lambda: DummyApi(),  # pylint: disable=unnecessary-lambda
        raising=True,
    )

    serving_module.ServingManager()
    serving_module.ServingManager()

    assert load_calls["count"] == 1
    assert serving_module._k8s_config_loaded is False


# ---------------------------------------------------------------------
# CRUD OPERATIONS
# ---------------------------------------------------------------------


def test_get_isvc(serving, serving_module):
    """
    get_isvc should call the underlying CustomObjectsApi.get_namespaced_custom_object.
    """
    _, fake_api, _ = serving_module

    result = serving.get_isvc("model-a")
    assert result["metadata"]["name"] == "model-a"
    assert fake_api.calls[-1][0] == "get"


def test_delete_isvc(serving, serving_module):
    """
    delete_isvc should call delete_namespaced_custom_object and return True.
    """
    _, fake_api, _ = serving_module

    ok = serving.delete_isvc("model-x")
    assert ok is True
    assert fake_api.calls[-1][0] == "delete"


def test_update_isvc(serving, serving_module):
    """
    update_isvc should patch the InferenceService spec.
    """
    _, fake_api, _ = serving_module

    patch = {"spec": {"predictor": {"minReplicas": 3}}}
    result = serving.update_isvc("m1", patch)

    assert result["metadata"]["name"] == "m1"
    assert fake_api.calls[-1][0] == "patch"


def test_restart_isvc(serving, serving_module):
    """
    restart_isvc should call patch twice (scale to 0 then back to 1).
    """
    _, fake_api, _ = serving_module

    serving.restart_isvc("m2")

    patch_calls = [c for c in fake_api.calls if c[0] == "patch"]
    assert len(patch_calls) == 2


# ---------------------------------------------------------------------
# MODEL DEPLOYMENT
# ---------------------------------------------------------------------


def test_deploy_model_resolves_model_and_calls_create_isvc(
    monkeypatch, serving, serving_module
):
    """
    deploy_model should:
    - resolve model URI via model helper
    - compute transformer env
    - call create_isvc (which calls CustomObjectsApi.create_namespaced_custom_object)
      with the expected values.
    """
    serving_module_obj, fake_api, _ = serving_module

    # Patch instance method _get_model_helpers on this ServingManager instance
    def fake_detect_model_format(model_uri=None, **_):
        return "onnx"

    def fake_get_model_details(**_):
        return {
            "model_uri": "s3://bucket/model",
            "model_name": "n1",
            "model_version": "v1",
            "model_id": "11111111-1111-1111-1111-111111111111",
        }

    monkeypatch.setattr(
        serving,
        "_get_model_helpers",
        lambda: (fake_detect_model_format, fake_get_model_details),
        raising=True,
    )

    # Patch transformer env resolution on the instance
    monkeypatch.setattr(
        serving,
        "_get_transformer_env",
        lambda *args, **kwargs: {"A": "1"},
        raising=True,
    )

    # Call deploy_model
    serving.deploy_model(
        model_id="11111111-1111-1111-1111-111111111111",
        isvc_name="myisvc",
        namespace="test-namespace",
    )

    # Find create call
    create_calls = [c for c in fake_api.calls if c[0] == "create"]
    assert len(create_calls) == 1
    _, kwargs = create_calls[0]

    body = kwargs["body"]
    assert body["metadata"]["name"] == "myisvc"
    assert body["spec"]["predictor"]["model"]["storageUri"] == "s3://bucket/model"
    assert body["spec"]["predictor"]["model"]["modelFormat"]["name"] == "onnx"

    # Check transformer env is wired through
    transformer = body["spec"]["transformer"]
    env_list = transformer["containers"][0]["env"]
    assert {"name": "A", "value": "1"} in env_list


def test_deploy_model_with_model_type(monkeypatch, serving, serving_module):
    """
    deploy_model should set model_type as an ISVC annotation when provided.
    """
    _, fake_api, _ = serving_module

    def fake_get_model_details(**_):
        return {
            "model_uri": "s3://bucket/model",
            "model_name": "n1",
            "model_version": "v1",
            "model_id": "11111111-1111-1111-1111-111111111111",
        }

    monkeypatch.setattr(
        serving,
        "_get_model_helpers",
        lambda: (lambda **_: "onnx", fake_get_model_details),
        raising=True,
    )
    monkeypatch.setattr(
        serving, "_get_transformer_env", lambda *args, **kwargs: {}, raising=True
    )

    serving.deploy_model(
        model_id="11111111-1111-1111-1111-111111111111",
        isvc_name="myisvc",
        namespace="test-namespace",
        model_type="llm",
    )

    create_calls = [c for c in fake_api.calls if c[0] == "create"]
    annotations = create_calls[0][1]["body"]["metadata"]["annotations"]
    assert annotations["model_type"] == "llm"


def test_deploy_model_without_model_type(monkeypatch, serving, serving_module):
    """
    deploy_model should NOT set model_type annotation when not provided.
    """
    _, fake_api, _ = serving_module

    def fake_get_model_details(**_):
        return {
            "model_uri": "s3://bucket/model",
            "model_name": "n1",
            "model_version": "v1",
            "model_id": "11111111-1111-1111-1111-111111111111",
        }

    monkeypatch.setattr(
        serving,
        "_get_model_helpers",
        lambda: (lambda **_: "onnx", fake_get_model_details),
        raising=True,
    )
    monkeypatch.setattr(
        serving, "_get_transformer_env", lambda *args, **kwargs: {}, raising=True
    )

    serving.deploy_model(model_id="11111111-1111-1111-1111-111111111111")

    create_calls = [c for c in fake_api.calls if c[0] == "create"]
    annotations = create_calls[0][1]["body"]["metadata"]["annotations"]
    assert "model_type" not in annotations


# ---------------------------------------------------------------------
# LIST MODELS
# ---------------------------------------------------------------------


def test_list_models_all(serving, serving_module):
    """
    list_models with no isvc_name should list all services in namespace.
    """
    _, fake_api, _ = serving_module

    models = serving.list_models()
    assert len(models) == 1
    assert models[0]["model_name"] == "m1"
    assert fake_api.calls[-1][0] == "list"


def test_list_models_single(serving, serving_module):
    """
    list_models with isvc_name should fetch and wrap a single model.
    """
    _, fake_api, _ = serving_module

    models = serving.list_models(isvc_name="isvc-a")
    assert len(models) == 1
    assert models[0]["isvc_name"] == "isvc-a"
    assert fake_api.calls[-1][0] == "get"


# ---------------------------------------------------------------------
# _process_isvc
# ---------------------------------------------------------------------


def test_process_isvc(serving_module):
    """
    _process_isvc should extract rollout and traffic info correctly.
    """
    module_obj, _, _ = serving_module

    data = {
        "metadata": {
            "name": "x",
            "annotations": {},
            "creationTimestamp": "2024-01-01T00:00:00Z",
        },
        "status": {
            "conditions": [{"type": "Ready", "status": "True"}],
            "components": {
                "predictor": {
                    "traffic": [{"revisionName": "r1", "percent": 100}],
                    "latestReadyRevision": "r1",
                }
            },
        },
        "spec": {"predictor": {}},
    }

    result = module_obj.ServingManager._process_isvc(data)
    assert result["isvc_name"] == "x"
    assert result["status"] == "ready"
    assert result["traffic_percentage"] == 100
    assert result["latest_ready_revision"] == "r1"


# ---------------------------------------------------------------------
# Regression: missing kube config must not break import
# ---------------------------------------------------------------------


def test_servingmanager_init_tolerates_missing_kube_config(monkeypatch):
    """Importing serving + constructing ServingManager must not raise when
    kube config is unavailable. First cluster access must raise
    CogflowConnectionError (not AttributeError)."""
    import importlib

    from kubernetes.config.config_exception import ConfigException

    from cogflow.utils import common
    from cogflow.utils.exceptions import CogflowConnectionError

    if hasattr(common, "_k8s_loaded_flag"):
        del common._k8s_loaded_flag

    def raising_load():
        raise ConfigException("Invalid kube-config file. No configuration found.")

    monkeypatch.setattr(common, "load_k8s_config", raising_load, raising=True)

    # Reload re-runs the module body, including `_serving = ServingManager()`.
    # Pre-fix this raised; with the fix it must succeed.
    import cogflow.core.serving as sm

    importlib.reload(sm)

    assert sm._serving is not None
    assert sm._serving._api is None

    # Accessing .api retries the load and surfaces the typed connection error.
    with pytest.raises(CogflowConnectionError):
        _ = sm._serving.api


# ---------------------------------------------------------------------
# LLM SERVING (deploy_llm)
# ---------------------------------------------------------------------


def _deploy_llm_create_call(fake_api):
    """Helper: extract the body of the single create call from the fake API."""
    creates = [c for c in fake_api.calls if c[0] == "create"]
    assert len(creates) == 1, f"expected 1 create call, got {len(creates)}"
    return creates[0][1]["body"]


def test_deploy_llm_hf_uri_emits_expected_spec(serving, serving_module):
    """hf:// URI → no serviceAccountName; modelFormat=huggingface; tolerations + args pass through."""
    _, fake_api, _ = serving_module

    serving.deploy_llm(
        storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
        isvc_name="qwen25-coder",
        served_model_name="qwen25-coder",
        max_model_len=4096,
        tolerations=[
            {"key": "storage-type", "operator": "Equal",
             "value": "local", "effect": "NoSchedule"}
        ],
        annotations={"model_type": "llm", "hf_model_id": "Qwen/Qwen2.5-Coder-7B-Instruct"},
    )

    body = _deploy_llm_create_call(fake_api)
    predictor = body["spec"]["predictor"]
    model = predictor["model"]

    assert body["metadata"]["name"] == "qwen25-coder"
    assert body["metadata"]["annotations"]["model_type"] == "llm"
    assert "serviceAccountName" not in predictor  # HF pull — no S3 SA
    assert model["modelFormat"] == {"name": "huggingface"}
    # HF source: pass the id via --model_id (runtime downloads from HF
    # Hub). storageUri would route through KServe's storage-initializer,
    # which injects fieldRef env vars Knative rejects.
    assert "storageUri" not in model
    assert "--model_id=Qwen/Qwen2.5-Coder-7B-Instruct" in model["args"]
    assert "--model_name=qwen25-coder" in model["args"]
    assert "--max_model_len=4096" in model["args"]
    assert predictor["tolerations"][0]["key"] == "storage-type"


def test_deploy_llm_rejects_empty_hf_id(serving, serving_module):
    """'hf://' with no model id should fail fast, not emit --model_id=."""
    from cogflow.utils.exceptions import CogflowValidationError

    for bad in ("hf://", "hf:///", "hf://   ", "hf:// / / "):
        with pytest.raises(CogflowValidationError, match="invalid HF model id"):
            serving.deploy_llm(
                storage_uri=bad,
                isvc_name="bad",
                served_model_name="bad",
            )


def test_deploy_llm_hf_uri_trims_decorative_slashes(serving, serving_module):
    """Leading/trailing slashes on the HF id are stripped before emit."""
    _, fake_api, _ = serving_module
    serving.deploy_llm(
        storage_uri="hf:///Qwen/Qwen2.5-Coder-7B-Instruct/",
        isvc_name="q",
        served_model_name="q",
    )
    args = _deploy_llm_create_call(fake_api)["spec"]["predictor"]["model"]["args"]
    assert "--model_id=Qwen/Qwen2.5-Coder-7B-Instruct" in args


def test_deploy_llm_s3_uri_sets_service_account(serving, serving_module):
    """s3:// URI → uses storageUri + kserve-controller-s3 SA."""
    _, fake_api, _ = serving_module

    serving.deploy_llm(
        storage_uri="s3://mlflow/0/abc/artifacts/model",
        isvc_name="mlf-llm",
        served_model_name="mlf-llm",
    )

    body = _deploy_llm_create_call(fake_api)
    predictor = body["spec"]["predictor"]
    model = predictor["model"]
    assert predictor["serviceAccountName"] == "kserve-controller-s3"
    assert model["storageUri"] == "s3://mlflow/0/abc/artifacts/model"
    # s3 path does NOT emit --model_id; the storage-initializer passes
    # --model_dir to the runtime instead.
    assert not any(a.startswith("--model_id=") for a in model["args"])


def test_deploy_llm_default_resources(serving, serving_module):
    """No resources override → defaults: 4/7Gi/1gpu requests, 8/8Gi/1gpu limits."""
    _, fake_api, _ = serving_module

    serving.deploy_llm(
        storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
        isvc_name="q",
        served_model_name="q",
    )

    res = _deploy_llm_create_call(fake_api)["spec"]["predictor"]["model"]["resources"]
    assert res["requests"] == {"cpu": "4", "memory": "7Gi", "nvidia.com/gpu": "1"}
    assert res["limits"] == {"cpu": "8", "memory": "8Gi", "nvidia.com/gpu": "1"}


def test_deploy_llm_resource_override_is_per_field_merge(serving, serving_module):
    """Only the overridden keys change; other defaults survive."""
    _, fake_api, _ = serving_module

    serving.deploy_llm(
        storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
        isvc_name="q",
        served_model_name="q",
        resources={
            "requests": {"memory": "14Gi", "nvidia.com/gpu": "2"},
            "limits": {"nvidia.com/gpu": "2"},
        },
    )

    res = _deploy_llm_create_call(fake_api)["spec"]["predictor"]["model"]["resources"]
    assert res["requests"]["cpu"] == "4"              # default preserved
    assert res["requests"]["memory"] == "14Gi"        # overridden
    assert res["requests"]["nvidia.com/gpu"] == "2"   # overridden
    assert res["limits"]["cpu"] == "8"                # default preserved
    assert res["limits"]["memory"] == "8Gi"           # default preserved
    assert res["limits"]["nvidia.com/gpu"] == "2"     # overridden


def test_deploy_llm_rejects_inverted_replica_range(serving, serving_module):
    """min_replicas > max_replicas should surface as CogflowValidationError
    before hitting the K8s API (where it would produce an admission error
    with less context).
    """
    from cogflow.utils.exceptions import CogflowValidationError

    with pytest.raises(CogflowValidationError, match="min_replicas"):
        serving.deploy_llm(
            storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
            isvc_name="bad",
            served_model_name="bad",
            min_replicas=3,
            max_replicas=1,
        )


def test_deploy_llm_rejects_negative_min_replicas(serving, serving_module):
    from cogflow.utils.exceptions import CogflowValidationError

    with pytest.raises(CogflowValidationError, match="min_replicas"):
        serving.deploy_llm(
            storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
            isvc_name="bad",
            served_model_name="bad",
            min_replicas=-1,
        )


def test_deploy_llm_rejects_zero_max_replicas(serving, serving_module):
    from cogflow.utils.exceptions import CogflowValidationError

    with pytest.raises(CogflowValidationError, match="max_replicas"):
        serving.deploy_llm(
            storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
            isvc_name="bad",
            served_model_name="bad",
            max_replicas=0,
        )


def test_deploy_llm_node_selector_and_replicas(serving, serving_module):
    _, fake_api, _ = serving_module

    serving.deploy_llm(
        storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
        isvc_name="q",
        served_model_name="q",
        node_selector={"gpu": "a100"},
        min_replicas=2,
        max_replicas=4,
    )

    predictor = _deploy_llm_create_call(fake_api)["spec"]["predictor"]
    assert predictor["nodeSelector"] == {"gpu": "a100"}
    assert predictor["minReplicas"] == 2
    assert predictor["maxReplicas"] == 4


def test_deploy_llm_with_hf_secret_adds_env(serving, serving_module):
    _, fake_api, _ = serving_module

    serving.deploy_llm(
        storage_uri="hf://meta-llama/Llama-3.1-8B-Instruct",
        isvc_name="llama",
        served_model_name="llama",
        hf_secret_name="cog-llm-token-llama",
    )

    env = _deploy_llm_create_call(fake_api)["spec"]["predictor"]["model"]["env"]
    assert env == [
        {
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {"name": "cog-llm-token-llama", "key": "HF_TOKEN"}
            },
        }
    ]


def test_deploy_llm_whitelisted_args_order_and_flags(serving, serving_module):
    """Args appear in stable order; trust_remote_code is a bare flag."""
    _, fake_api, _ = serving_module

    serving.deploy_llm(
        storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
        isvc_name="q",
        served_model_name="q",
        max_model_len=4096,
        dtype="bfloat16",
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_num_seqs=32,
    )

    args = _deploy_llm_create_call(fake_api)["spec"]["predictor"]["model"]["args"]
    # HF source prepends --model_id= so the runtime knows which model to
    # pull. Underscored flags (model_name, max_model_len, dtype,
    # trust_remote_code) land in KServe's HF runtime parser; hyphenated
    # flags (tensor-parallel-size, gpu-memory-utilization, max-num-seqs)
    # fall through to vLLM via parse_known_args. See _build_llm_args
    # docstring for the split rationale.
    assert args == [
        "--model_id=Qwen/Qwen2.5-Coder-7B-Instruct",
        "--model_name=q",
        "--max_model_len=4096",
        "--dtype=bfloat16",
        "--trust_remote_code",
        "--tensor-parallel-size=2",
        "--gpu-memory-utilization=0.9",
        "--max-num-seqs=32",
    ]


# ---------------------------------------------------------------------
# LLM name derivation (derive_llm_names + deploy_llm with omitted names)
# ---------------------------------------------------------------------


def test_derive_llm_names_uses_slug_after_slash(serving_module):
    """hf_model_id='Org/Name' → served_model_name='Name',
    isvc_name=<DNS-1123 slug of Name>."""
    module_obj, _, _ = serving_module
    isvc, smn = module_obj.ServingManager.derive_llm_names(
        hf_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    assert smn == "Qwen2.5-Coder-7B-Instruct"
    assert isvc == "qwen2-5-coder-7b-instruct"


def test_derive_llm_names_bare_hf_id_no_slash(serving_module):
    """HF supports bare ids without an org segment — use as-is."""
    module_obj, _, _ = serving_module
    isvc, smn = module_obj.ServingManager.derive_llm_names(
        hf_model_id="gpt2",
    )
    assert smn == "gpt2"
    assert isvc == "gpt2"


def test_derive_llm_names_respects_explicit_inputs(serving_module):
    """Caller-supplied names beat HF-id derivation."""
    module_obj, _, _ = serving_module
    isvc, smn = module_obj.ServingManager.derive_llm_names(
        hf_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        served_model_name="my-model",
        isvc_name="my-isvc",
    )
    assert smn == "my-model"
    assert isvc == "my-isvc"


def test_derive_llm_names_mlflow_without_name_raises(serving_module):
    """No hf_model_id + no served_model_name → typed error."""
    from cogflow.utils.exceptions import CogflowValidationError

    module_obj, _, _ = serving_module
    with pytest.raises(CogflowValidationError, match="served_model_name is required"):
        module_obj.ServingManager.derive_llm_names()


def test_derive_llm_names_rejects_non_dns1123_isvc(serving_module):
    """Explicit isvc_name must match the DNS-1123 label pattern."""
    from cogflow.utils.exceptions import CogflowValidationError

    module_obj, _, _ = serving_module
    with pytest.raises(CogflowValidationError, match="DNS-1123"):
        module_obj.ServingManager.derive_llm_names(
            served_model_name="ok",
            isvc_name="Not_A_Valid_Name",
        )


def test_deploy_llm_derives_both_names_from_hf_uri(serving, serving_module):
    """Omit both name args → cogflow derives them from the hf:// URI."""
    _, fake_api, _ = serving_module

    serving.deploy_llm(storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct")

    body = _deploy_llm_create_call(fake_api)
    assert body["metadata"]["name"] == "qwen2-5-coder-7b-instruct"
    args = body["spec"]["predictor"]["model"]["args"]
    assert "--model_name=Qwen2.5-Coder-7B-Instruct" in args
    assert "--model_id=Qwen/Qwen2.5-Coder-7B-Instruct" in args


def test_deploy_llm_derives_isvc_from_served_model_name(serving, serving_module):
    """Only served_model_name supplied → isvc_name defaults to its slug."""
    _, fake_api, _ = serving_module

    serving.deploy_llm(
        storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
        served_model_name="My.Custom_Name",
    )

    body = _deploy_llm_create_call(fake_api)
    assert body["metadata"]["name"] == "my-custom-name"


def test_deploy_llm_malformed_hf_uri_raises_hf_error_not_name_error(serving):
    """Malformed ``hf://`` URI with names omitted must surface the
    specific "invalid HF model id" error from URI validation, not
    the generic "served_model_name is required" from name derivation.

    Regression: earlier the derivation step ran first and shadowed
    the URI error when both the URI was bad and the names were None.
    """
    from cogflow.utils.exceptions import CogflowValidationError

    for bad in ("hf://", "hf:///", "hf://   "):
        with pytest.raises(CogflowValidationError, match="invalid HF model id"):
            serving.deploy_llm(storage_uri=bad)


def test_deploy_llm_s3_without_served_model_name_raises(serving):
    """MLflow/s3 path can't derive a name — caller must supply one."""
    from cogflow.utils.exceptions import CogflowValidationError

    with pytest.raises(CogflowValidationError, match="served_model_name is required"):
        serving.deploy_llm(storage_uri="s3://mlflow/0/abc/artifacts/model")


def test_serving_module_reexports_async_deploy_llm():
    """cogflow.serving must expose async_deploy_llm alongside the other
    async_* helpers. Consumers (e.g. Cog-Engine) reach it via
    ``from cogflow import serving as cogflow_serving`` through the
    ``cogflow.__init__`` _LazyLoader.

    We check this in a fresh subprocess because pytest-mock's conftest
    eagerly imports subpackages of ``cogflow`` before the test runs,
    which defeats the _LazyLoader (``sys.modules['cogflow']`` ends up
    as the plain package module, not the _LazyLoader stub). That's a
    pre-existing cogflow-wide quirk, not a regression of this change —
    the subprocess guarantees the only import machinery in play is the
    one consumers actually see.
    """
    import subprocess
    import sys

    script = (
        "from cogflow import serving as cogflow_serving; "
        "assert hasattr(cogflow_serving, 'async_deploy_llm'); "
        "assert callable(cogflow_serving.async_deploy_llm)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "Timed out while verifying public API "
            "`cogflow.serving.async_deploy_llm` in a subprocess.\n"
            f"stdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        )
    assert result.returncode == 0, (
        f"Public API `cogflow.serving.async_deploy_llm` does not resolve.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
