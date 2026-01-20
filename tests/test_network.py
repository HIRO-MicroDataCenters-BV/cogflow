import pytest
import requests
from tenacity import RetryError

from cogflow.utils import network

# ============================================================
# Helpers
# ============================================================


class FakeResponse:
    def __init__(
        self,
        ok=True,
        status_code=200,
        json_data=None,
        text="",
        headers=None,
    ):
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} error")

    def iter_content(self, chunk_size=1024):
        yield b"data"


# ============================================================
# POST
# ============================================================


def test_make_post_request_success_with_json(mocker):
    response = FakeResponse(ok=True, json_data={"a": 1})

    mocker.patch(
        "cogflow.utils.network.requests.post",
        return_value=response,
    )

    result = network.make_post_request(
        url="http://x",
        data={"x": 1},
    )
    assert result == {"a": 1}


def test_make_post_request_success_with_files(mocker):
    response = FakeResponse(ok=True, json_data={"ok": True})

    mocker.patch(
        "cogflow.utils.network.requests.post",
        return_value=response,
    )

    result = network.make_post_request(
        url="http://x",
        files={"f": b"1"},
    )
    assert result["ok"] is True


def test_make_post_request_failure(mocker):
    response = FakeResponse(ok=False, status_code=400, text="bad")

    mocker.patch(
        "cogflow.utils.network.requests.post",
        return_value=response,
    )

    with pytest.raises(RetryError):
        network.make_post_request(url="http://x")


# ============================================================
# GET
# ============================================================


def test_make_get_request_success(mocker):
    response = FakeResponse(ok=True, json_data={"data": 1})

    mocker.patch(
        "cogflow.utils.network.requests.get",
        return_value=response,
    )

    result = network.make_get_request("http://x")
    assert result == {"data": 1}


def test_make_get_request_with_path_params(mocker):
    response = FakeResponse(ok=True, json_data={"id": 1})

    mocker.patch(
        "cogflow.utils.network.requests.get",
        return_value=response,
    )

    result = network.make_get_request("http://x", path_params="123")
    assert result["id"] == 1


def test_make_get_request_pagination(mocker):
    page1 = FakeResponse(
        ok=True,
        json_data={
            "data": [1, 2],
            "pagination": {"total_items": 3},
        },
    )
    page2 = FakeResponse(
        ok=True,
        json_data={
            "data": [3],
            "pagination": {"total_items": 3},
        },
    )

    mocker.patch(
        "cogflow.utils.network.requests.get",
        side_effect=[page1, page2],
    )

    result = network.make_get_request(
        "http://x",
        query_params={"limit": 2},
        paginate=True,
    )

    assert result == [1, 2, 3]


# ============================================================
# STREAM GET
# ============================================================


def test_make_get_request_stream_success(mocker):
    response = FakeResponse(ok=True)

    mocker.patch(
        "cogflow.utils.network.requests.get",
        return_value=response,
    )

    resp = network.make_get_request_stream("http://x")
    assert resp is response


def test_make_get_request_stream_failure(mocker):
    response = FakeResponse(ok=False, status_code=500)

    mocker.patch(
        "cogflow.utils.network.requests.get",
        return_value=response,
    )

    with pytest.raises(RetryError):
        network.make_get_request_stream("http://x")


# ============================================================
# DELETE
# ============================================================


@pytest.mark.parametrize("status", [200, 202, 204])
def test_make_delete_request_success(mocker, status):
    response = FakeResponse(status_code=status)

    mocker.patch(
        "cogflow.utils.network.requests.delete",
        return_value=response,
    )

    assert network.make_delete_request("http://x") is True


def test_make_delete_request_failure(mocker):
    response = FakeResponse(status_code=500, text="boom")

    mocker.patch(
        "cogflow.utils.network.requests.delete",
        return_value=response,
    )

    with pytest.raises(RetryError):
        network.make_delete_request("http://x")


# ============================================================
# PATCH
# ============================================================


def test_make_patch_request_success(mocker):
    response = FakeResponse(ok=True, json_data={"ok": True})

    mocker.patch(
        "cogflow.utils.network.requests.patch",
        return_value=response,
    )

    result = network.make_patch_request("http://x", data={"a": 1})
    assert result["ok"] is True


def test_make_patch_request_failure(mocker):
    response = FakeResponse(ok=False, status_code=400)

    mocker.patch(
        "cogflow.utils.network.requests.patch",
        return_value=response,
    )

    with pytest.raises(RetryError):
        network.make_patch_request("http://x")


# ============================================================
# RAW GET
# ============================================================


def test_make_get_request_raw_success(mocker):
    response = FakeResponse(json_data={"x": 1})

    mocker.patch(
        "cogflow.utils.network.requests.get",
        return_value=response,
    )

    result = network.make_get_request_raw("http://x")
    assert result == {"x": 1}


def test_make_get_request_raw_failure(mocker):
    mocker.patch(
        "cogflow.utils.network.requests.get",
        side_effect=requests.RequestException("boom"),
    )

    assert network.make_get_request_raw("http://x") is None


# ============================================================
# HEALTH CHECK
# ============================================================


def test_make_health_check_request_success(mocker):
    response = FakeResponse(status_code=200)

    mocker.patch(
        "cogflow.utils.network.requests.get",
        return_value=response,
    )

    assert network.make_health_check_request("http://x") is True


def test_make_health_check_request_non_2xx(mocker):
    response = FakeResponse(status_code=500)

    mocker.patch(
        "cogflow.utils.network.requests.get",
        return_value=response,
    )

    assert network.make_health_check_request("http://x") is False


def test_make_health_check_request_exception(mocker):
    mocker.patch(
        "cogflow.utils.network.requests.get",
        side_effect=requests.RequestException("boom"),
    )

    assert network.make_health_check_request("http://x") is False
