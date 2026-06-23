from types import SimpleNamespace

import pytest

from cogflow.core.pipelines import orchestration
from cogflow.utils.exceptions import (
    CogflowConnectionError,
    CogflowPipelineError,
)

# ============================================================
# _safe_kfp_call
# ============================================================


def test_safe_kfp_call_success(mocker):
    fn = mocker.Mock(return_value="ok")

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_k8s_models",
        return_value=(RuntimeError, None),
    )

    result = orchestration._safe_kfp_call(fn, "ctx")
    assert result == "ok"


def test_safe_kfp_call_api_exception(mocker):
    def bad():
        raise RuntimeError("api-fail")

    # Treat RuntimeError as ApiException
    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_k8s_models",
        return_value=(RuntimeError, None),
    )

    with pytest.raises(CogflowConnectionError):
        orchestration._safe_kfp_call(bad, "ctx")


def test_safe_kfp_call_generic_exception(mocker):
    def bad():
        raise ValueError("boom")

    # ApiException is NOT ValueError
    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_k8s_models",
        return_value=(RuntimeError, None),
    )

    with pytest.raises(CogflowPipelineError):
        orchestration._safe_kfp_call(bad, "ctx")


# ============================================================
# client()
# ============================================================


def test_client_internal_cluster(mocker):
    fake_client = mocker.Mock()

    class FakeClient:
        def __init__(self, **kwargs):
            pass

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_kfp",
        return_value=(SimpleNamespace(Client=FakeClient), None),
    )

    mocker.patch.object(FakeClient, "__call__", return_value=fake_client)

    result = orchestration.client()
    assert result is not None


def test_client_external_with_cookies(mocker):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def _load_config(self):
            return SimpleNamespace(verify_ssl=True)

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_kfp",
        return_value=(SimpleNamespace(Client=FakeClient), None),
    )

    result = orchestration.client(
        api_url="http://example",
        session_cookies="cookie",
        namespace="ns",
    )

    assert result is not None


# ============================================================
# pipeline decorator
# ============================================================


def test_pipeline_decorator(mocker):
    fake_pipeline = mocker.Mock()

    dsl = SimpleNamespace(pipeline=lambda **_: fake_pipeline)

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_kfp",
        return_value=(None, dsl),
    )

    decorator = orchestration.pipeline(name="p", description="d")
    assert decorator == fake_pipeline


# ============================================================
# create_component_from_func
# ============================================================


def test_create_component_from_func(mocker):
    fake_op = mocker.Mock()
    fake_op.add_env_variable = mocker.Mock()

    fake_comp = mocker.Mock(return_value=fake_op)
    fake_comp.component_spec = SimpleNamespace(name="x")

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_kfp_components",
        return_value=SimpleNamespace(create_component_from_func=lambda **_: fake_comp),
    )

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_config",
        return_value=SimpleNamespace(COMP_BASE_IMAGE="img"),
    )

    mocker.patch(
        "cogflow.core.pipelines.orchestration._inject_env_into_container_op",
        return_value=fake_op,
    )

    wrapped = orchestration.create_component_from_func(lambda x: x)
    result = wrapped(1)

    assert result == fake_op


# ============================================================
# load_component_* helpers
# ============================================================


def test_load_component_from_text(mocker):
    fake = mocker.Mock()

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_kfp_components",
        return_value=SimpleNamespace(load_component_from_text=lambda _: fake),
    )

    result = orchestration.load_component_from_text("yaml")
    assert result == fake


def test_load_component_from_file(mocker):
    fake = mocker.Mock()

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_kfp_components",
        return_value=SimpleNamespace(load_component_from_file=lambda _: fake),
    )

    result = orchestration.load_component_from_file("x.yaml")
    assert result == fake


def test_load_component_from_url(mocker):
    fake = mocker.Mock()

    mocker.patch(
        "cogflow.core.pipelines.orchestration._load_kfp_components",
        return_value=SimpleNamespace(load_component_from_url=lambda _: fake),
    )

    result = orchestration.load_component_from_url("http://x")
    assert result == fake


# ============================================================
# create_run_from_pipeline_func
# ============================================================


def test_create_run_from_pipeline_func(mocker):
    fake_client = mocker.Mock()
    fake_client.create_run_from_pipeline_func.return_value = "run"

    mocker.patch(
        "cogflow.core.pipelines.orchestration.client",
        return_value=fake_client,
    )

    result = orchestration.create_run_from_pipeline_func(lambda: None)
    assert result == "run"


# ============================================================
# delete_pipeline
# ============================================================


def test_delete_pipeline(mocker):
    fake_client = mocker.Mock()
    fake_client.delete_pipeline.return_value = True

    mocker.patch(
        "cogflow.core.pipelines.orchestration.client",
        return_value=fake_client,
    )

    result = orchestration.delete_pipeline("pid")
    assert result is True


# ============================================================
# _valid_param_names
# ============================================================


def test_valid_param_names():
    def fn(a, b=1, *, c=2, **kw):
        pass

    sig = orchestration.inspect.signature(fn)
    names = orchestration._valid_param_names(sig)

    assert names == ["a", "b", "c"]


# ============================================================
# _create_service_component / _delete_service_component
# ============================================================


class _MockApiException(Exception):
    """Lightweight stand-in for kubernetes.client.exceptions.ApiException."""

    def __init__(self, status):
        self.status = status


def _patch_k8s(mocker, *, api_mock):
    """Patch kubernetes imports used by the self-contained KFP components."""
    mocker.patch("kubernetes.config.load_incluster_config")
    mocker.patch("kubernetes.client.CoreV1Api", return_value=api_mock)
    mocker.patch("kubernetes.client.V1Service")
    mocker.patch("kubernetes.client.V1ObjectMeta")
    mocker.patch("kubernetes.client.V1ServiceSpec")
    mocker.patch("kubernetes.client.V1ServicePort")
    mocker.patch("kubernetes.client.exceptions.ApiException", _MockApiException)


def test_create_service_component_success(mocker):
    mock_api = mocker.Mock()
    _patch_k8s(mocker, api_mock=mock_api)

    result = orchestration._create_service_component("flserver-abc")

    assert result == "flserver-abc"
    mock_api.create_namespaced_service.assert_called_once()


def test_create_service_component_409_swallowed(mocker):
    mock_api = mocker.Mock()
    mock_api.create_namespaced_service.side_effect = _MockApiException(409)
    _patch_k8s(mocker, api_mock=mock_api)

    result = orchestration._create_service_component("flserver-abc")

    assert result == "flserver-abc"


def test_create_service_component_non_409_raises(mocker):
    mock_api = mocker.Mock()
    mock_api.create_namespaced_service.side_effect = _MockApiException(403)
    _patch_k8s(mocker, api_mock=mock_api)

    with pytest.raises(_MockApiException):
        orchestration._create_service_component("flserver-abc")


def test_delete_service_component_success(mocker):
    mock_api = mocker.Mock()
    _patch_k8s(mocker, api_mock=mock_api)

    result = orchestration._delete_service_component("flserver-abc")

    assert result == "flserver-abc"
    mock_api.delete_namespaced_service.assert_called_once()


def test_delete_service_component_404_swallowed(mocker):
    mock_api = mocker.Mock()
    mock_api.delete_namespaced_service.side_effect = _MockApiException(404)
    _patch_k8s(mocker, api_mock=mock_api)

    result = orchestration._delete_service_component("flserver-abc")

    assert result == "flserver-abc"


def test_delete_service_component_non_404_raises(mocker):
    mock_api = mocker.Mock()
    mock_api.delete_namespaced_service.side_effect = _MockApiException(403)
    _patch_k8s(mocker, api_mock=mock_api)

    with pytest.raises(_MockApiException):
        orchestration._delete_service_component("flserver-abc")
