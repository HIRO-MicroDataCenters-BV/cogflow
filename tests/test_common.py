import os
from datetime import datetime
from uuid import UUID

import pytest

from cogflow.utils import common

# ============================================================
# Serialization utilities
# ============================================================


def test_custom_serializer_datetime():
    dt = datetime(2024, 1, 1, 12, 0, 0)
    assert common.custom_serializer(dt) == dt.isoformat()


def test_custom_serializer_invalid_type():
    with pytest.raises(TypeError):
        common.custom_serializer(123)


def test_serialize_artifacts_with_uri():
    class Artifact:
        def __init__(self, uri):
            self.uri = uri

    artifacts = {"a": Artifact("s3://bucket/key")}
    result = common.serialize_artifacts(artifacts)

    assert result == {"validation_artifacts": {"a": "s3://bucket/key"}}


def test_serialize_artifacts_without_uri():
    artifacts = {"a": 123}
    result = common.serialize_artifacts(artifacts)

    assert result == {"validation_artifacts": {"a": "123"}}


# ============================================================
# S3 URI validation
# ============================================================


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3://my-bucket/file.txt", True),
        ("s3://bucket/path/to/file", True),
        ("http://bucket/file", False),
        ("s3://bucket", False),
        ("", False),
    ],
)
def test_is_valid_s3_uri(uri, expected):
    assert common.is_valid_s3_uri(uri) is expected


def test_is_valid_s3_uri_none():
    with pytest.raises(TypeError):
        common.is_valid_s3_uri(None)


# ============================================================
# UUID utilities
# ============================================================


def test_normalize_uuid_from_uuid_obj():
    u = UUID("7a1f6cf8-1d7e-4d40-b9a9-1c94ce6c3c0a")
    assert common.normalize_uuid(u) == str(u)


def test_normalize_uuid_compact():
    compact = "7a1f6cf81d7e4d40b9a91c94ce6c3c0a"
    assert common.normalize_uuid(compact) == "7a1f6cf8-1d7e-4d40-b9a9-1c94ce6c3c0a"


def test_normalize_uuid_hyphenated():
    value = "7a1f6cf8-1d7e-4d40-b9a9-1c94ce6c3c0a"
    assert common.normalize_uuid(value) == value


def test_normalize_uuid_invalid():
    with pytest.raises(ValueError):
        common.normalize_uuid("not-a-uuid")


def test_uuid_to_hex_success():
    value = "7a1f6cf8-1d7e-4d40-b9a9-1c94ce6c3c0a"
    assert common.uuid_to_hex(value) == "7a1f6cf81d7e4d40b9a91c94ce6c3c0a"


def test_uuid_to_hex_invalid():
    with pytest.raises(ValueError):
        common.uuid_to_hex("bad-uuid")


# ============================================================
# Kubernetes / Kubeflow utilities
# ============================================================


def test_get_namespace_incluster_file(mocker):
    mocker.patch("cogflow.utils.common.load_k8s_config")
    mocker.patch("builtins.open", mocker.mock_open(read_data="test-ns"))
    assert common.get_namespace() == "test-ns"


def test_get_namespace_kubeconfig_fallback(mocker):
    mocker.patch("cogflow.utils.common.load_k8s_config")
    mocker.patch("builtins.open", side_effect=FileNotFoundError)
    mocker.patch(
        "cogflow.utils.common.config.list_kube_config_contexts",
        return_value=(None, {"context": {"namespace": "kube-ns"}}),
    )
    assert common.get_namespace() == "kube-ns"


def test_get_namespace_default_on_error(mocker):
    mocker.patch(
        "cogflow.utils.common.load_k8s_config",
        side_effect=Exception("boom"),
    )
    assert common.get_namespace() == "default"


def test_get_current_user_success(mocker):
    mocker.patch("cogflow.utils.common.get_namespace", return_value="ns")

    fake_ns = mocker.Mock()
    fake_ns.metadata.annotations = {"owner": "test-user"}

    fake_api = mocker.Mock()
    fake_api.read_namespace.return_value = fake_ns

    mocker.patch(
        "cogflow.utils.common.client.CoreV1Api",
        return_value=fake_api,
    )

    assert common.get_current_user() == "test-user"


def test_get_current_user_missing_owner_returns_string_or_fails_gracefully(mocker):
    """
    get_current_user is intentionally defensive.
    We assert it does NOT crash the test runner.
    """
    mocker.patch("cogflow.utils.common.get_namespace", return_value="ns")

    fake_ns = mocker.Mock()
    fake_ns.metadata.annotations = {}

    fake_api = mocker.Mock()
    fake_api.read_namespace.return_value = fake_ns

    mocker.patch(
        "cogflow.utils.common.client.CoreV1Api",
        return_value=fake_api,
    )

    try:
        result = common.get_current_user()
        assert isinstance(result, str)
    except RuntimeError:
        assert True  # acceptable behavior


def test_get_current_user_api_failure_is_graceful(mocker):
    mocker.patch("cogflow.utils.common.get_namespace", return_value="ns")
    mocker.patch(
        "cogflow.utils.common.client.CoreV1Api",
        side_effect=Exception("boom"),
    )

    try:
        result = common.get_current_user()
        assert isinstance(result, str)
    except RuntimeError:
        assert True


# ============================================================
# File system helpers
# ============================================================


def test_file_exists_true(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert common.file_exists(str(f)) is True


def test_file_exists_false(tmp_path):
    assert common.file_exists(str(tmp_path / "nope.txt")) is False


def test_file_exists_none():
    assert common.file_exists(None) is False


def test_get_filename():
    assert common.get_filename("/a/b/c.txt") == "c.txt"


def test_get_filename_empty():
    assert common.get_filename("") == ""


def test_cwd():
    assert common.cwd() == os.getcwd()


def test_is_dir_true(tmp_path):
    assert common.is_dir(str(tmp_path)) is True


def test_is_dir_false(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert common.is_dir(str(f)) is False


def test_is_dir_none():
    assert common.is_dir(None) is False


def test_join_path():
    result = common.join_path("/a", "b", None, "c")
    assert result.endswith(os.path.join("a", "b", "c"))


# ============================================================
# create_service / delete_service
# ============================================================


class _MockApiException(Exception):
    """Lightweight stand-in for kubernetes.client.exceptions.ApiException."""

    def __init__(self, status):
        self.status = status


def _patch_k8s_common(mocker, *, api_mock):
    mocker.patch("cogflow.utils.common.load_k8s_config")
    mocker.patch("cogflow.utils.common.get_namespace", return_value="test-ns")
    mocker.patch("kubernetes.client.CoreV1Api", return_value=api_mock)
    mocker.patch("kubernetes.client.V1Service")
    mocker.patch("kubernetes.client.V1ObjectMeta")
    mocker.patch("kubernetes.client.V1ServiceSpec")
    mocker.patch("kubernetes.client.V1ServicePort")
    mocker.patch("kubernetes.client.exceptions.ApiException", _MockApiException)


def test_create_service_success(mocker):
    mock_api = mocker.Mock()
    _patch_k8s_common(mocker, api_mock=mock_api)

    result = common.create_service("flserver-abc")

    assert result == "flserver-abc"
    mock_api.create_namespaced_service.assert_called_once()


def test_create_service_409_swallowed(mocker):
    mock_api = mocker.Mock()
    mock_api.create_namespaced_service.side_effect = _MockApiException(409)
    _patch_k8s_common(mocker, api_mock=mock_api)

    result = common.create_service("flserver-abc")

    assert result == "flserver-abc"


def test_create_service_non_409_raises(mocker):
    mock_api = mocker.Mock()
    mock_api.create_namespaced_service.side_effect = _MockApiException(403)
    _patch_k8s_common(mocker, api_mock=mock_api)

    with pytest.raises(_MockApiException):
        common.create_service("flserver-abc")


def test_delete_service_success(mocker):
    mock_api = mocker.Mock()
    _patch_k8s_common(mocker, api_mock=mock_api)

    result = common.delete_service("flserver-abc")

    assert result == "flserver-abc"
    mock_api.delete_namespaced_service.assert_called_once()


def test_delete_service_404_swallowed(mocker):
    mock_api = mocker.Mock()
    mock_api.delete_namespaced_service.side_effect = _MockApiException(404)
    _patch_k8s_common(mocker, api_mock=mock_api)

    result = common.delete_service("flserver-abc")

    assert result == "flserver-abc"


def test_delete_service_non_404_raises(mocker):
    mock_api = mocker.Mock()
    mock_api.delete_namespaced_service.side_effect = _MockApiException(403)
    _patch_k8s_common(mocker, api_mock=mock_api)

    with pytest.raises(_MockApiException):
        common.delete_service("flserver-abc")
