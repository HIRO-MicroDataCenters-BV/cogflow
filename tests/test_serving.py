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
