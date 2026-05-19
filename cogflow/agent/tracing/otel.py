"""OpenTelemetry tracing adapter (opt-in)."""

from __future__ import annotations

from typing import Any

from .base import TracingAdapter


class OTelAdapter(TracingAdapter):
    name = "otel"

    def __init__(self) -> None:
        self._enabled = False
        self._instrumentor: Any = None

    def enable(self, *, endpoint: str | None = None, **kwargs: Any) -> None:
        try:
            from openinference.instrumentation.langchain import (  # type: ignore[import-not-found]
                LangChainInstrumentor,
            )
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "OTel tracing requires `openinference-instrumentation-langchain`; "
                "install with `pip install cogflow[agent]`."
            ) from exc

        if endpoint:
            try:
                from opentelemetry import trace  # type: ignore[import-not-found]
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
                from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
                    BatchSpanProcessor,
                )

                provider = TracerProvider()
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
                trace.set_tracer_provider(provider)
            except ImportError:  # pragma: no cover
                pass

        self._instrumentor = LangChainInstrumentor()
        self._instrumentor.instrument()
        self._enabled = True

    def disable(self) -> None:
        if self._enabled and self._instrumentor is not None:
            self._instrumentor.uninstrument()
        self._enabled = False
