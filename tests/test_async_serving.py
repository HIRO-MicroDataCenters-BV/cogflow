"""
Unit tests for cogflow.core.async_serving.AsyncServingManager.

All Kubernetes I/O is mocked via kubernetes_asyncio so tests
do not require a real cluster.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cogflow.utils import common
import cogflow.core.async_serving as async_serving_module


# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------


@pytest.fixture
def fake_async_api():
    """A fake async CustomObjectsApi that records calls and returns canned data."""
    api = MagicMock()
    api.calls = []

    async def get_namespaced_custom_object(**kwargs):
        api.calls.append(("get", kwargs))
        return {
            "metadata": {
                "name": kwargs["name"],
                "annotations": {"model_name": "m1", "model_type": "llm"},
                "creationTimestamp": "2024-01-01T00:00:00Z",
            },
            "status": {
                "conditions": [{"type": "Ready", "status": "True"}],
                "components": {"predictor": {"traffic": []}},
            },
            "spec": {"predictor": {}},
        }

    async def delete_namespaced_custom_object(**kwargs):
        api.calls.append(("delete", kwargs))
        return {}

    async def patch_namespaced_custom_object(**kwargs):
        api.calls.append(("patch", kwargs))
        return {"metadata": {"name": kwargs["name"]}}

    async def list_namespaced_custom_object(**kwargs):
        api.calls.append(("list", kwargs))
        return {
            "items": [
                {
                    "metadata": {
                        "name": "isvc-a",
                        "annotations": {"model_name": "m1", "model_type": "llm"},
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

    async def create_namespaced_custom_object(**kwargs):
        api.calls.append(("create", kwargs))
        return {"metadata": {"name": kwargs["body"]["metadata"]["name"]}}

    api.get_namespaced_custom_object = get_namespaced_custom_object
    api.delete_namespaced_custom_object = delete_namespaced_custom_object
    api.patch_namespaced_custom_object = patch_namespaced_custom_object
    api.list_namespaced_custom_object = list_namespaced_custom_object
    api.create_namespaced_custom_object = create_namespaced_custom_object
    return api


@pytest.fixture
def async_serving(monkeypatch, fake_async_api):
    """Returns a fresh AsyncServingManager instance with the api pre-injected."""
    # Prevent real K8s config loading when the sync serving singleton
    # is instantiated indirectly via _get_serving_manager_class().
    monkeypatch.setattr(common, "load_k8s_config", lambda: None, raising=True)
    monkeypatch.setattr(common, "get_namespace", lambda: "test-ns", raising=True)
    # Pre-mark the sync flag so the singleton doesn't try to load config
    common._k8s_loaded_flag = True
    monkeypatch.setattr(
        async_serving_module, "_ASYNC_K8S_CONFIG_LOADED", True, raising=True
    )
    mgr = async_serving_module.AsyncServingManager()
    mgr._api = fake_async_api  # Bypass _get_api()
    return mgr


# ---------------------------------------------------------------------
# CRUD OPERATIONS
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_get_isvc(async_serving, fake_async_api):
    """async get_isvc should call get_namespaced_custom_object."""
    result = await async_serving.get_isvc("model-a")
    assert result["metadata"]["name"] == "model-a"
    assert fake_async_api.calls[-1][0] == "get"


@pytest.mark.asyncio
async def test_async_delete_isvc(async_serving, fake_async_api):
    """async delete_isvc should return True on success."""
    ok = await async_serving.delete_isvc("model-x")
    assert ok is True
    assert fake_async_api.calls[-1][0] == "delete"


@pytest.mark.asyncio
async def test_async_update_isvc(async_serving, fake_async_api):
    """async update_isvc should patch the InferenceService spec."""
    patch_body = {"spec": {"predictor": {"minReplicas": 3}}}
    result = await async_serving.update_isvc("m1", patch_body)
    assert result["metadata"]["name"] == "m1"
    assert fake_async_api.calls[-1][0] == "patch"


@pytest.mark.asyncio
async def test_async_restart_isvc(async_serving, fake_async_api):
    """async restart_isvc should call patch twice (scale to 0 then back)."""
    await async_serving.restart_isvc("m2")
    patch_calls = [c for c in fake_async_api.calls if c[0] == "patch"]
    assert len(patch_calls) == 2


# ---------------------------------------------------------------------
# DEPLOY MODEL with model_type
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_deploy_model_with_model_type(
    monkeypatch, async_serving, fake_async_api
):
    """
    async deploy_model should set model_type as an ISVC annotation
    when provided.
    """

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
        async_serving,
        "_get_model_helpers",
        lambda: (fake_detect_model_format, fake_get_model_details),
        raising=True,
    )
    monkeypatch.setattr(
        async_serving,
        "_get_transformer_env",
        lambda *args, **kwargs: {},
        raising=True,
    )

    await async_serving.deploy_model(
        model_id="11111111-1111-1111-1111-111111111111",
        isvc_name="myisvc",
        namespace="test-ns",
        model_type="llm",
    )

    create_calls = [c for c in fake_async_api.calls if c[0] == "create"]
    assert len(create_calls) == 1
    body = create_calls[0][1]["body"]

    assert body["metadata"]["name"] == "myisvc"
    annotations = body["metadata"]["annotations"]
    assert annotations["model_type"] == "llm"
    assert annotations["model_name"] == "n1"
    assert body["spec"]["predictor"]["model"]["storageUri"] == "s3://bucket/model"


@pytest.mark.asyncio
async def test_async_deploy_model_without_model_type(
    monkeypatch, async_serving, fake_async_api
):
    """
    async deploy_model should NOT set model_type annotation when not provided.
    """

    def fake_get_model_details(**_):
        return {
            "model_uri": "s3://bucket/model",
            "model_name": "n1",
            "model_version": "v1",
            "model_id": "11111111-1111-1111-1111-111111111111",
        }

    monkeypatch.setattr(
        async_serving,
        "_get_model_helpers",
        lambda: (lambda **_: "onnx", fake_get_model_details),
        raising=True,
    )
    monkeypatch.setattr(
        async_serving, "_get_transformer_env", lambda *args, **kwargs: {}, raising=True
    )

    await async_serving.deploy_model(model_id="11111111-1111-1111-1111-111111111111")

    create_calls = [c for c in fake_async_api.calls if c[0] == "create"]
    annotations = create_calls[0][1]["body"]["metadata"]["annotations"]
    assert "model_type" not in annotations


# ---------------------------------------------------------------------
# LIST MODELS
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_list_models_all(async_serving, fake_async_api):
    """async list_models with no isvc_name should list all in namespace."""
    models = await async_serving.list_models()
    assert len(models) == 1
    assert models[0]["model_name"] == "m1"
    assert models[0]["model_type"] == "llm"
    assert fake_async_api.calls[-1][0] == "list"


@pytest.mark.asyncio
async def test_async_list_models_single(async_serving, fake_async_api):
    """async list_models with isvc_name should return a single wrapped model."""
    models = await async_serving.list_models(isvc_name="isvc-a")
    assert len(models) == 1
    assert models[0]["isvc_name"] == "isvc-a"
    assert models[0]["model_type"] == "llm"


# ---------------------------------------------------------------------
# MODULE-LEVEL BINDINGS
# ---------------------------------------------------------------------


def test_async_module_exports():
    """Verify module-level async functions are exposed."""
    assert callable(async_serving_module.async_deploy_model)
    assert callable(async_serving_module.async_update_model)
    assert callable(async_serving_module.async_delete_isvc)
    assert callable(async_serving_module.async_list_models)
    assert callable(async_serving_module.async_get_isvc)
    assert callable(async_serving_module.async_update_isvc)
    assert callable(async_serving_module.async_restart_isvc)
    assert callable(async_serving_module.async_create_isvc)
    assert callable(async_serving_module.async_deploy_llm)


# ---------------------------------------------------------------------
# LLM SERVING (async deploy_llm)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_deploy_llm_emits_expected_spec(async_serving, fake_async_api):
    """The async path delegates spec-building to the sync helper, so the
    emitted ISVC must match what deploy_llm in serving.py produces."""
    await async_serving.deploy_llm(
        storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
        isvc_name="qwen25-coder",
        served_model_name="qwen25-coder",
        max_model_len=4096,
        tolerations=[
            {"key": "storage-type", "operator": "Equal",
             "value": "local", "effect": "NoSchedule"}
        ],
        annotations={"model_type": "llm"},
    )

    creates = [c for c in fake_async_api.calls if c[0] == "create"]
    assert len(creates) == 1
    body = creates[0][1]["body"]
    predictor = body["spec"]["predictor"]
    model = predictor["model"]

    assert body["metadata"]["name"] == "qwen25-coder"
    assert body["metadata"]["annotations"]["model_type"] == "llm"
    assert "serviceAccountName" not in predictor
    assert model["modelFormat"] == {"name": "huggingface"}
    # HF source: --model_id arg instead of storageUri (see sync test).
    assert "storageUri" not in model
    assert "--model_id=Qwen/Qwen2.5-Coder-7B-Instruct" in model["args"]
    assert "--model_name=qwen25-coder" in model["args"]
    assert "--max_model_len=4096" in model["args"]
    assert predictor["tolerations"][0]["key"] == "storage-type"
    # Defaults applied
    assert model["resources"]["requests"]["nvidia.com/gpu"] == "1"
