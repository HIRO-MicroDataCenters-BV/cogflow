"""LangSmith tracing adapter (opt-in)."""

from __future__ import annotations

import os
from typing import Any

from .base import TracingAdapter


class LangSmithAdapter(TracingAdapter):
    name = "langsmith"

    def __init__(self) -> None:
        self._enabled = False

    def enable(self, *, project: str | None = None, api_key: str | None = None, **kwargs: Any) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if project:
            os.environ["LANGCHAIN_PROJECT"] = project
        if api_key:
            os.environ["LANGCHAIN_API_KEY"] = api_key
        self._enabled = True

    def disable(self) -> None:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        self._enabled = False
