# Tracing & observability

Tracing is **disabled** until you call `configure_tracing(...)`
explicitly — `compile()` doesn't auto-enable anything because library
code shouldn't have surprise global side-effects.

When you do enable tracing, the SDK supports two stackable backends:

- **MLflow** is the default backend (self-hosted, reuses cogflow's
  existing kubeflow infrastructure). It's what you get when you call
  `configure_tracing()` with no `backend=` argument.
- **OpenTelemetry** is opt-in via `backend="otel"`.

## MLflow (default backend)

```python
from cogflow.agent.tracing import configure_tracing

configure_tracing(backend="mlflow", experiment="my-agent")
```

What this does:

1. Imports `mlflow` (already pinned by base cogflow at `2.22.0`).
2. Reads `cogflow.config.config.MLFLOW_TRACKING_URI` and calls
   `mlflow.set_tracking_uri(...)` — your kubeflow MLflow gets the spans.
3. Calls `mlflow.set_experiment(experiment)` so traces land in a
   named experiment.
4. Calls `mlflow.langchain.autolog()` which registers a LangChain
   callback that emits a span for every LLM call, tool invocation, and
   chain step.

Override the URI inline if you need to point at a non-default MLflow:

```python
configure_tracing(backend="mlflow", experiment="local-dev",
                  tracking_uri="http://localhost:5000")
```

Disable cleanly:

```python
configure_tracing(backend=None)
```

## OpenTelemetry (opt-in)

```bash
pip install "cogflow[agent-otel]"
```

This pulls in `openinference-instrumentation-langchain` plus
`opentelemetry-sdk` and `opentelemetry-exporter-otlp`.

```python
configure_tracing(
    backend="otel",
    endpoint="http://my-otel-collector:4318/v1/traces",
)
```

If `endpoint` is passed but the OTel SDK + OTLP exporter aren't
installed, you get a clear `RuntimeError` pointing at
`pip install cogflow[agent-otel]` — not a silent fallback that "looks
enabled" while exporting nothing.

## Stacking

The two backends are independent LangChain callback registrations:

```python
configure_tracing(backend="mlflow", experiment="prod")
configure_tracing(backend="otel", endpoint="http://collector:4318/v1/traces")
# both active — spans flow to both
```

## Repeated configure calls (idempotency)

`configure_tracing(backend="mlflow", ...)` called twice doesn't stack
the autolog callback. Internally each adapter:

- `enable()` short-circuits when `self._enabled` is True
- `configure_tracing(...)` calls `adapter.disable()` then
  `adapter.enable(**kwargs)` for the same backend — so you can safely
  re-configure with different kwargs

## LangSmith is intentionally not bundled

LangSmith is a commercial hosted service from LangChain Inc. cogflow
does not ship a LangSmith adapter; if you want LangSmith tracing, set
the environment variables yourself and LangChain will pick it up at
runtime — no cogflow-side wiring needed:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls-...
export LANGCHAIN_PROJECT=my-agent
```

## Flowise-side tracing

When the same agent runs inside a Flowise canvas (not on LangGraph),
the canvas has its own analytics hooks. `compile.to_flowise.to_dict(...,
analytics={"mlflow": {...}})` injects a matching analytics block into
the exported JSON so the canvas can be wired to the same MLflow your
LangGraph runtime uses. The Flowise side is configured per Flowise's
documentation; the SDK just passes the block through.
