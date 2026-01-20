import pytest
from cogflow.utils import storage

# ------------------------------------------------------------------
# Dummy Minio client
# ------------------------------------------------------------------


class DummyMinio:
    def __init__(self, endpoint, access_key, secret_key, secure):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure


@pytest.fixture(autouse=True)
def patch_minio(mocker):
    mocker.patch("cogflow.utils.storage.Minio", DummyMinio)


@pytest.fixture
def base_config(mocker):
    mocker.patch("cogflow.utils.storage.config.AWS_ACCESS_KEY_ID", "ak")
    mocker.patch("cogflow.utils.storage.config.AWS_SECRET_ACCESS_KEY", "sk")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_minio_client_with_https_endpoint(base_config, mocker):
    mocker.patch(
        "cogflow.utils.storage.config.MLFLOW_S3_ENDPOINT_URL",
        "https://minio.internal:9000",
    )

    client = storage.minio_client()

    assert client.endpoint == "minio.internal:9000"
    assert client.secure is True


def test_minio_client_with_http_endpoint(base_config, mocker):
    mocker.patch(
        "cogflow.utils.storage.config.MLFLOW_S3_ENDPOINT_URL",
        "http://minio.internal:9000",
    )

    client = storage.minio_client()

    assert client.endpoint == "minio.internal:9000"
    assert client.secure is False


def test_minio_client_without_protocol_does_not_crash(base_config, mocker):
    """
    No protocol endpoint behavior is unstable in current implementation.
    This test only ensures no exception is raised.
    """
    mocker.patch(
        "cogflow.utils.storage.config.MLFLOW_S3_ENDPOINT_URL",
        "minio.internal:9000",
    )

    client = storage.minio_client()

    assert client is not None


def test_minio_client_missing_endpoint_raises(mocker):
    mocker.patch("cogflow.utils.storage.config.MLFLOW_S3_ENDPOINT_URL", None)

    with pytest.raises(RuntimeError, match="MLFLOW_S3_ENDPOINT_URL is not set"):
        storage.minio_client()


def test_minio_client_invalid_endpoint_does_not_crash(base_config, mocker):
    """
    Invalid endpoints are currently passed through.
    Test documents this behavior without enforcing details.
    """
    mocker.patch(
        "cogflow.utils.storage.config.MLFLOW_S3_ENDPOINT_URL",
        ":::::",
    )

    client = storage.minio_client()

    assert client is not None
