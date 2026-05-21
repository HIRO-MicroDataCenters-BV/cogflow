"""HTTP node — single REST call via httpx, result stashed into state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..ir.model import IRNode
from .base import NodeFactory


class HTTPFactory(NodeFactory):
    ir_type = "http"

    def __init__(
        self,
        *,
        method: str = "GET",
        url: str = "",
        headers: dict[str, str] | None = None,
        body: Any = None,
        timeout: float = 30.0,
        output_key: str = "http_response",
        **extra: Any,
    ) -> None:
        super().__init__(
            httpMethod=method.upper(),
            httpUrl=url,
            httpHeaders=headers or {},
            httpBody=body if body is not None else "",
            httpTimeout=float(timeout),
            httpOutputKey=output_key,
            **extra,
        )

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        method = (self.config.get("httpMethod") or "GET").upper()
        url = self.config.get("httpUrl") or ""
        headers = self.config.get("httpHeaders") or {}
        body = self.config.get("httpBody")
        timeout = float(self.config.get("httpTimeout") or 30.0)
        output_key = self.config.get("httpOutputKey") or "http_response"

        # Allow ctx to inject a stub httpx for tests without monkey-patching
        # the import path; falls back to the real package by default.
        http_client = (ctx or {}).get("httpx") if ctx else None

        def http_node(state: dict[str, Any]) -> dict[str, Any]:
            client = http_client
            if client is None:
                import httpx  # noqa: WPS433

                client = httpx
            # Missing template keys must not crash the graph — fall back to
            # the unrendered URL so the user can see what was attempted.
            rendered_url = url
            if url:
                try:
                    rendered_url = url.format(**state)
                except (KeyError, IndexError):
                    rendered_url = url
            kwargs: dict[str, Any] = {"headers": headers, "timeout": timeout}
            if body not in (None, ""):
                kwargs["json"] = body
            response = client.request(method, rendered_url, **kwargs)
            try:
                payload: Any = response.json()
            except Exception:
                payload = getattr(response, "text", str(response))
            return {output_key: {"status": getattr(response, "status_code", None), "body": payload}}

        return http_node


def http(**kwargs: Any) -> HTTPFactory:
    return HTTPFactory(**kwargs)
