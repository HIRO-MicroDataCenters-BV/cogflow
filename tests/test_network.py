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
# Async POST (httpx-backed mirror of make_post_request)
# ============================================================


class FakeAsyncResponse:
    """Minimal stand-in for httpx.Response — only the surface
    make_async_post_request touches.
    """

    def __init__(
        self,
        is_success=True,
        status_code=200,
        json_data=None,
        text="",
    ):
        self.is_success = is_success
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        # Mirror what httpx.Response.raise_for_status raises on 4xx/5xx
        # — make_async_post_request only catches httpx.HTTPError.
        import httpx as _httpx

        request = _httpx.Request("POST", "http://x")
        response = _httpx.Response(
            status_code=self.status_code,
            request=request,
        )
        raise _httpx.HTTPStatusError(
            f"{self.status_code} error", request=request, response=response
        )


class _FakeAsyncClient:
    """Lightweight stand-in for httpx.AsyncClient used as an async
    context manager. Each verb (.post/.get/.delete/.patch) takes either
    a return value or an exception to raise; ``.responses`` lets a test
    return different responses per successive call.
    """

    def __init__(
        self,
        *,
        post_return=None,
        post_raises=None,
        get_return=None,
        get_responses=None,
        get_raises=None,
        delete_return=None,
        delete_raises=None,
        patch_return=None,
        patch_raises=None,
        captured=None,
    ):
        self._post_return = post_return
        self._post_raises = post_raises
        self._get_return = get_return
        self._get_responses = list(get_responses) if get_responses else None
        self._get_raises = get_raises
        self._delete_return = delete_return
        self._delete_raises = delete_raises
        self._patch_return = patch_return
        self._patch_raises = patch_raises
        self._captured = captured if captured is not None else []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self._captured.append({"verb": "post", "url": url, **kwargs})
        if self._post_raises is not None:
            raise self._post_raises
        return self._post_return

    async def get(self, url, **kwargs):
        self._captured.append({"verb": "get", "url": url, **kwargs})
        if self._get_raises is not None:
            raise self._get_raises
        if self._get_responses:
            return self._get_responses.pop(0)
        return self._get_return

    async def delete(self, url, **kwargs):
        self._captured.append({"verb": "delete", "url": url, **kwargs})
        if self._delete_raises is not None:
            raise self._delete_raises
        return self._delete_return

    async def patch(self, url, **kwargs):
        self._captured.append({"verb": "patch", "url": url, **kwargs})
        if self._patch_raises is not None:
            raise self._patch_raises
        return self._patch_return


@pytest.mark.asyncio
async def test_make_async_post_request_success_with_json(mocker):
    captured = []
    response = FakeAsyncResponse(is_success=True, json_data={"a": 1})
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            post_return=response, captured=captured
        ),
    )

    result = await network.make_async_post_request(
        url="http://x",
        data={"x": 1},
    )
    assert result == {"a": 1}
    # Match sync semantics: dict body sent as JSON.
    assert captured[0]["json"] == {"x": 1}


@pytest.mark.asyncio
async def test_make_async_post_request_no_body(mocker):
    """Empty / None data should send no JSON body — mirrors sync."""
    captured = []
    response = FakeAsyncResponse(is_success=True, json_data={})
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            post_return=response, captured=captured
        ),
    )

    await network.make_async_post_request(url="http://x")
    assert "json" not in captured[0]


@pytest.mark.asyncio
async def test_make_async_post_request_retries_then_raises(mocker):
    """4xx response triggers raise_for_status -> httpx.HTTPError ->
    tenacity retries up to 3x then re-raises as RetryError."""
    captured = []
    response = FakeAsyncResponse(is_success=False, status_code=400, text="bad")
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            post_return=response, captured=captured
        ),
    )

    with pytest.raises(RetryError):
        await network.make_async_post_request(url="http://x")
    assert len(captured) == 3  # stop_after_attempt(3)


@pytest.mark.asyncio
async def test_make_async_post_request_does_not_retry_logic_errors(mocker):
    """Non-httpx errors (e.g. JSON decode) should NOT be retried —
    they're logic errors, not transients. Verifies the
    retry_if_exception_type(httpx.HTTPError) restriction."""
    captured = []

    class _BoomResponse(FakeAsyncResponse):
        def json(self):
            raise ValueError("not json")

    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            post_return=_BoomResponse(is_success=True),
            captured=captured,
        ),
    )

    with pytest.raises(ValueError):
        await network.make_async_post_request(url="http://x", data={"a": 1})
    # Single attempt — no retry on non-httpx errors.
    assert len(captured) == 1


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


# ============================================================
# Async GET / DELETE / PATCH / raw / health (httpx-backed mirrors)
# ============================================================


@pytest.mark.asyncio
async def test_make_async_get_request_success(mocker):
    captured = []
    response = FakeAsyncResponse(is_success=True, json_data={"v": 1})
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            get_return=response, captured=captured
        ),
    )
    assert await network.make_async_get_request("http://x") == {"v": 1}
    assert captured[0]["verb"] == "get"


@pytest.mark.asyncio
async def test_make_async_get_request_with_path_params(mocker):
    captured = []
    response = FakeAsyncResponse(is_success=True, json_data={})
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            get_return=response, captured=captured
        ),
    )
    await network.make_async_get_request("http://x", path_params="123")
    assert captured[0]["url"] == "http://x/123"


@pytest.mark.asyncio
async def test_make_async_get_request_pagination(mocker):
    """Two pages, second is the final — total_items=2, two items each, stops."""
    captured = []
    page1 = FakeAsyncResponse(
        is_success=True,
        json_data={
            "data": [{"id": 1}, {"id": 2}],
            "pagination": {"total_items": 4},
        },
    )
    page2 = FakeAsyncResponse(
        is_success=True,
        json_data={
            "data": [{"id": 3}, {"id": 4}],
            "pagination": {"total_items": 4},
        },
    )
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            get_responses=[page1, page2], captured=captured
        ),
    )
    result = await network.make_async_get_request(
        "http://x", query_params={"limit": 2}, paginate=True
    )
    assert result == [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    # Second request should carry page=2.
    assert captured[1]["params"]["page"] == 2


@pytest.mark.parametrize("status", [200, 202, 204])
@pytest.mark.asyncio
async def test_make_async_delete_request_success(mocker, status):
    captured = []
    response = FakeAsyncResponse(is_success=True, status_code=status)
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            delete_return=response, captured=captured
        ),
    )
    assert await network.make_async_delete_request("http://x") is True


@pytest.mark.asyncio
async def test_make_async_delete_request_failure(mocker):
    response = FakeAsyncResponse(is_success=False, status_code=500, text="bad")
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(delete_return=response),
    )
    with pytest.raises(RetryError):
        await network.make_async_delete_request("http://x")


@pytest.mark.asyncio
async def test_make_async_patch_request_success(mocker):
    captured = []
    response = FakeAsyncResponse(is_success=True, json_data={"updated": True})
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            patch_return=response, captured=captured
        ),
    )
    result = await network.make_async_patch_request("http://x", data={"a": 1})
    assert result == {"updated": True}
    assert captured[0]["json"] == {"a": 1}


@pytest.mark.asyncio
async def test_make_async_patch_request_failure(mocker):
    response = FakeAsyncResponse(is_success=False, status_code=400, text="bad")
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(patch_return=response),
    )
    with pytest.raises(RetryError):
        await network.make_async_patch_request("http://x", data={})


@pytest.mark.asyncio
async def test_make_async_get_request_raw_success(mocker):
    response = FakeAsyncResponse(is_success=True, json_data={"raw": True})
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(get_return=response),
    )
    assert await network.make_async_get_request_raw("http://x") == {"raw": True}


@pytest.mark.asyncio
async def test_make_async_get_request_raw_failure(mocker):
    """Same shape as the sync raw helper: transport error → None."""
    import httpx as _httpx

    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            get_raises=_httpx.ConnectError("boom")
        ),
    )
    assert await network.make_async_get_request_raw("http://x") is None


@pytest.mark.asyncio
async def test_make_async_health_check_request_success(mocker):
    response = FakeAsyncResponse(is_success=True, status_code=200)
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(get_return=response),
    )
    assert await network.make_async_health_check_request("http://x") is True


@pytest.mark.asyncio
async def test_make_async_health_check_request_non_2xx(mocker):
    response = FakeAsyncResponse(is_success=False, status_code=500, text="bad")
    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(get_return=response),
    )
    assert await network.make_async_health_check_request("http://x") is False


@pytest.mark.asyncio
async def test_make_async_health_check_request_exception(mocker):
    """Health check is meant to fail fast — no retry, just False."""
    import httpx as _httpx

    mocker.patch(
        "cogflow.utils.network.httpx.AsyncClient",
        side_effect=lambda **_: _FakeAsyncClient(
            get_raises=_httpx.ConnectError("boom")
        ),
    )
    assert await network.make_async_health_check_request("http://x") is False
