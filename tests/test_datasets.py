import pytest
from uuid import UUID

from cogflow.core.datasets import DatasetManager
from cogflow.utils.exceptions import (
    CogflowValidationError,
    CogflowConnectionError,
    CogflowArtifactError,
    CogflowDatasetError,
)

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def manager():
    return DatasetManager()


# ============================================================
# GET DATASET
# ============================================================


def test_get_dataset_success(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch("cogflow.utils.common.get_current_user", return_value="user")
    mocker.patch(
        "cogflow.utils.network.make_get_request",
        return_value={"data": {"id": "uuid"}},
    )

    result = manager.get_dataset("uuid")
    assert result["id"] == "uuid"


def test_get_dataset_invalid_uuid(mocker, manager):
    mocker.patch(
        "cogflow.utils.common.normalize_uuid",
        side_effect=ValueError("bad uuid"),
    )

    with pytest.raises(CogflowValidationError):
        manager.get_dataset("bad")


def test_get_dataset_network_error(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch(
        "cogflow.utils.network.make_get_request",
        side_effect=Exception("network"),
    )

    with pytest.raises(CogflowConnectionError):
        manager.get_dataset("uuid")


def test_get_dataset_missing_data(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch(
        "cogflow.utils.network.make_get_request",
        return_value={},
    )

    with pytest.raises(CogflowArtifactError):
        manager.get_dataset("uuid")


# ============================================================
# GET PROMETHEUS DATASET
# ============================================================


def test_get_prometheus_dataset_success(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch(
        "cogflow.utils.network.make_get_request",
        return_value={"data": {"metrics": []}},
    )

    result = manager.get_prometheus_dataset("uuid")
    assert "metrics" in result


def test_get_prometheus_dataset_empty_data(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch(
        "cogflow.utils.network.make_get_request",
        return_value={"data": {}},
    )

    with pytest.raises(CogflowDatasetError):
        manager.get_prometheus_dataset("uuid")


# ============================================================
# REGISTER DATASET
# ============================================================


def test_register_dataset_success(mocker, manager):
    mocker.patch("cogflow.utils.common.file_exists", return_value=True)
    mocker.patch("cogflow.utils.common.get_current_user", return_value="user")
    mocker.patch("cogflow.utils.common.get_filename", return_value="data.csv")
    mocker.patch("builtins.open", mocker.mock_open(read_data=b"x"))

    mocker.patch(
        "cogflow.utils.network.make_post_request",
        return_value={"data": {"id": "123"}},
    )

    result = manager.register_dataset(
        dataset_type=0,
        name="train",
        file_path="/tmp/data.csv",
    )

    assert result["id"] == "123"


def test_register_dataset_invalid_type(manager):
    with pytest.raises(CogflowValidationError):
        manager.register_dataset(5, "name", "/tmp/file")


def test_register_dataset_file_missing(mocker, manager):
    mocker.patch("cogflow.utils.common.file_exists", return_value=False)

    with pytest.raises(CogflowValidationError):
        manager.register_dataset(0, "name", "/tmp/file")


# ============================================================
# DELETE DATASET (SILENT)
# ============================================================


def test_delete_dataset_success(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch("cogflow.utils.common.get_current_user", return_value="user")
    mocker.patch("cogflow.utils.network.make_delete_request", return_value=True)

    assert manager.delete_dataset("uuid") is True


def test_delete_dataset_network_error(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch(
        "cogflow.utils.network.make_delete_request",
        side_effect=Exception("network"),
    )

    with pytest.raises(CogflowConnectionError):
        manager.delete_dataset("uuid")


# ============================================================
# DOWNLOAD DATASET
# ============================================================


def test_download_dataset_success(mocker, manager, tmp_path):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch("cogflow.utils.common.get_current_user", return_value="user")
    mocker.patch("cogflow.utils.common.cwd", return_value=str(tmp_path))
    mocker.patch("cogflow.utils.common.is_dir", return_value=False)
    mocker.patch("cogflow.utils.common.file_exists", return_value=True)

    fake_response = mocker.Mock()
    fake_response.headers = {"Content-Disposition": 'attachment; filename="data.zip"'}
    fake_response.iter_content.return_value = [b"abc", b"123"]

    mocker.patch(
        "cogflow.utils.network.make_get_request_stream",
        return_value=fake_response,
    )

    path = manager.download_dataset("uuid")
    assert path.endswith("data.zip")


def test_download_dataset_no_filename(mocker, manager, tmp_path):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch("cogflow.utils.common.cwd", return_value=str(tmp_path))
    mocker.patch("cogflow.utils.common.file_exists", return_value=True)

    fake_response = mocker.Mock()
    fake_response.headers = {}
    fake_response.iter_content.return_value = [b"x"]

    mocker.patch(
        "cogflow.utils.network.make_get_request_stream",
        return_value=fake_response,
    )

    path = manager.download_dataset("uuid")
    assert path.endswith(".bin")


def test_download_dataset_network_error(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch(
        "cogflow.utils.network.make_get_request_stream",
        side_effect=Exception("network"),
    )

    with pytest.raises(CogflowConnectionError):
        manager.download_dataset("uuid")


# ============================================================
# Async dataset operations
# ============================================================


@pytest.mark.asyncio
async def test_async_get_dataset_success(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch("cogflow.utils.common.get_current_user", return_value="user")

    async def fake(*_a, **_kw):
        return {"data": {"id": "uuid"}}

    mocker.patch("cogflow.utils.network.make_async_get_request", side_effect=fake)
    result = await manager.async_get_dataset("uuid")
    assert result["id"] == "uuid"


@pytest.mark.asyncio
async def test_async_get_dataset_network_error(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")

    async def fake(*_a, **_kw):
        raise Exception("network")

    mocker.patch("cogflow.utils.network.make_async_get_request", side_effect=fake)
    with pytest.raises(CogflowConnectionError):
        await manager.async_get_dataset("uuid")


@pytest.mark.asyncio
async def test_async_get_prometheus_dataset_success(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")
    mocker.patch("cogflow.utils.common.get_current_user", return_value="user")

    async def fake(*_a, **_kw):
        return {"data": {"metric": 42}}

    mocker.patch("cogflow.utils.network.make_async_get_request", side_effect=fake)
    result = await manager.async_get_prometheus_dataset("uuid")
    assert result["metric"] == 42


@pytest.mark.asyncio
async def test_async_get_prometheus_dataset_empty_data(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")

    async def fake(*_a, **_kw):
        return {"data": None}

    mocker.patch("cogflow.utils.network.make_async_get_request", side_effect=fake)
    with pytest.raises(CogflowDatasetError):
        await manager.async_get_prometheus_dataset("uuid")


@pytest.mark.asyncio
async def test_async_delete_dataset_success(mocker, manager):
    mocker.patch("cogflow.utils.common.normalize_uuid", return_value="uuid")

    async def fake(*_a, **_kw):
        return True

    mocker.patch("cogflow.utils.network.make_async_delete_request", side_effect=fake)
    assert await manager.async_delete_dataset("uuid") is True


@pytest.mark.asyncio
async def test_async_delete_dataset_invalid_uuid(mocker, manager):
    mocker.patch(
        "cogflow.utils.common.normalize_uuid",
        side_effect=ValueError("bad uuid"),
    )
    with pytest.raises(CogflowValidationError):
        await manager.async_delete_dataset("bad")
