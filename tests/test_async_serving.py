"""
Unit tests for cogflow.core.async_serving.AsyncServingManager.

All Kubernetes I/O is mocked via kubernetes_asyncio so tests
do not require a real cluster.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import cogflow.core.async_serving as async_serving_module
from cogflow.utils import common

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
    monkeypatch.setattr(async_serving_module, "_ASYNC_K8S_CONFIG_LOADED", True, raising=True)
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
async def test_async_deploy_model_with_model_type(monkeypatch, async_serving, fake_async_api):
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
async def test_async_deploy_model_without_model_type(monkeypatch, async_serving, fake_async_api):
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
    monkeypatch.setattr(async_serving, "_get_transformer_env", lambda *args, **kwargs: {}, raising=True)

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
    assert callable(async_serving_module.async_serve_llm)


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
        tolerations=[{"key": "storage-type", "operator": "Equal", "value": "local", "effect": "NoSchedule"}],
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


@pytest.mark.asyncio
async def test_async_serve_llm_registers_and_deploys(async_serving, fake_async_api, monkeypatch):
    """async_serve_llm should register the catalog entry (async, via
    async_register_llm_catalog_entry — sync version would deadlock the
    event loop when caller and /models/log share a worker) and then
    create the KServe ISVC async. Annotations on the resulting ISVC
    must include model_id=run_id — the uniform invariant that keeps
    GET /models/{id} working for LLM rows without any type branching.
    """
    import cogflow.core.models as core_models_module

    fake_run_id = "deadbeef0000deadbeef0000deadbeef"
    register_mock = AsyncMock(return_value=fake_run_id)
    monkeypatch.setattr(
        core_models_module,
        "async_register_llm_catalog_entry",
        register_mock,
        raising=True,
    )

    result = await async_serving.serve_llm(hf_model_id="Qwen/Qwen2.5-Coder-7B-Instruct")

    register_mock.assert_called_once()
    assert result["run_id"] == fake_run_id
    assert result["served_model_name"] == "Qwen2.5-Coder-7B-Instruct"
    assert result["isvc_name"] == "qwen2-5-coder-7b-instruct"

    creates = [c for c in fake_async_api.calls if c[0] == "create"]
    assert len(creates) == 1
    body = creates[0][1]["body"]
    annotations = body["metadata"]["annotations"]
    assert annotations["model_id"] == common.normalize_uuid(fake_run_id)
    assert annotations["model_type"] == "llm"
    assert annotations["hf_model_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"


@pytest.mark.asyncio
async def test_async_register_finetuned_catalog_entry_posts_adapter_row(monkeypatch):
    """async_register_finetuned_catalog_entry POSTs the adapter row via
    the async HTTP client (the sync POST would deadlock when caller and
    /models/log share a worker). Shares the validation + payload shape
    with the sync variant."""
    import cogflow.core.models as core_models_module

    run_id = "abcdef000000abcdef000000abcdef00"
    base_id = "22222222222222222222222222222222"
    post_mock = AsyncMock()
    monkeypatch.setattr(core_models_module.network, "make_async_post_request", post_mock, raising=True)
    monkeypatch.setattr(core_models_module.common, "get_current_user", lambda: "u@example.com", raising=True)

    returned = await core_models_module.async_register_finetuned_catalog_entry(
        run_id=run_id,
        served_model_name="my-lora",
        adapter_type="lora",
        base_model_id=base_id,
        base_model_hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        register_date_ms=1_700_000_000_000,
    )

    assert returned == run_id
    post_mock.assert_awaited_once()
    payload = post_mock.call_args.kwargs["data"]
    assert payload["model_id"] == common.normalize_uuid(run_id)
    assert payload["type"] == "lora"
    assert payload["base_model_id"] == common.normalize_uuid(base_id)
    assert payload["hf_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------
# LLM SERVING ENGINE (async parity with the sync path)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_deploy_llm_airllm_spec_parity(async_serving, fake_async_api):
    """The async path delegates to the sync spec builder, so engine='airllm'
    must emit the identical modelFormat / args / volume shape."""
    await async_serving.deploy_llm(
        storage_uri="hf://NousResearch/Meta-Llama-3-8B-Instruct",
        engine="airllm",
        max_model_len=4096,
        cache_pvc_name="air-cache",
    )

    creates = [c for c in fake_async_api.calls if c[0] == "create"]
    assert len(creates) == 1
    predictor = creates[0][1]["body"]["spec"]["predictor"]
    assert predictor["model"]["modelFormat"] == {"name": "airllm"}
    assert "--model_id=NousResearch/Meta-Llama-3-8B-Instruct" in predictor["model"]["args"]
    assert not any(a.startswith("--tensor-parallel-size") for a in predictor["model"]["args"])
    assert predictor["volumes"] == [
        {"name": "airllm-cache", "persistentVolumeClaim": {"claimName": "air-cache"}}
    ]


@pytest.mark.asyncio
async def test_async_deploy_llm_airllm_rejects_vllm_only_kwargs(async_serving, fake_async_api):
    from cogflow.utils.exceptions import CogflowValidationError

    with pytest.raises(CogflowValidationError, match="tensor_parallel_size"):
        await async_serving.deploy_llm(
            storage_uri="hf://Qwen/Qwen2.5-Coder-7B-Instruct",
            engine="airllm",
            tensor_parallel_size=2,
        )


@pytest.mark.asyncio
async def test_async_serve_llm_records_engine(async_serving, fake_async_api, monkeypatch):
    """async_serve_llm stamps llm_engine on catalog tags + ISVC annotations,
    matching the sync path."""
    import cogflow.core.models as core_models_module

    fake_run_id = "deadbeef1111deadbeef1111deadbeef"
    register_mock = AsyncMock(return_value=fake_run_id)
    monkeypatch.setattr(
        core_models_module,
        "async_register_llm_catalog_entry",
        register_mock,
        raising=True,
    )

    await async_serving.serve_llm(
        hf_model_id="Qwen/Qwen2.5-Coder-7B-Instruct", engine="airllm"
    )

    assert register_mock.call_args.kwargs["extra_tags"]["llm_engine"] == "airllm"
    creates = [c for c in fake_async_api.calls if c[0] == "create"]
    annotations = creates[0][1]["body"]["metadata"]["annotations"]
    assert annotations["llm_engine"] == "airllm"


@pytest.mark.asyncio
async def test_async_serve_llm_invalid_engine_fails_before_catalog(async_serving, fake_async_api, monkeypatch):
    import cogflow.core.models as core_models_module
    from cogflow.utils.exceptions import CogflowValidationError

    register_mock = AsyncMock(return_value="deadbeef" * 4)
    monkeypatch.setattr(
        core_models_module,
        "async_register_llm_catalog_entry",
        register_mock,
        raising=True,
    )

    with pytest.raises(CogflowValidationError, match="is not supported"):
        await async_serving.serve_llm(hf_model_id="Qwen/Qwen2.5-Coder-7B-Instruct", engine="tgi")

    register_mock.assert_not_called()
