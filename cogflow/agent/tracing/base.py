"""Abstract base for tracing adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TracingAdapter(ABC):
    name: str

    @abstractmethod
    def enable(self, **kwargs: object) -> None: ...

    @abstractmethod
    def disable(self) -> None: ...
