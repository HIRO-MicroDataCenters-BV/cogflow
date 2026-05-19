"""LangSmith tracing adapter (opt-in)."""

from __future__ import annotations

import os
from typing import Any

from .base import TracingAdapter


_MANAGED_ENV_VARS = ("LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_API_KEY")


class LangSmithAdapter(TracingAdapter):
    name = "langsmith"

    def __init__(self) -> None:
        self._enabled = False
        # Snapshot of any env vars that existed *before* enable() — disable()
        # restores them so this adapter is reversible across test runs.
        self._restore: dict[str, str | None] = {}

    def enable(self, *, project: str | None = None, api_key: str | None = None, **kwargs: Any) -> None:
        if self._enabled:
            return
        # Verify the optional dep is present before we mutate any environment
        # variables — otherwise tracing would silently appear enabled while
        # LangChain has no LangSmith client to talk to.
        try:
            import langsmith  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LangSmith tracing requires `langsmith`. "
                "Install with `pip install cogflow[agent-langsmith]`."
            ) from exc

        self._restore = {var: os.environ.get(var) for var in _MANAGED_ENV_VARS}

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if project:
            os.environ["LANGCHAIN_PROJECT"] = project
        if api_key:
            os.environ["LANGCHAIN_API_KEY"] = api_key
        self._enabled = True

    def disable(self) -> None:
        for var, prev in self._restore.items():
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev
        self._restore = {}
        self._enabled = False
