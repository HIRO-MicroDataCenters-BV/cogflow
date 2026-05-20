# Agent SDK overview

`cogflow.agent` is a **LangGraph-shaped Python SDK** whose graphs round-trip
to **Flowise V2 AgentFlow JSON**. Two starting points are first-class:

```text
┌──────────────────┐                         ┌──────────────────────┐
│  Python source   │  ── compile() ─────►   │  langgraph runtime   │
│ (StateGraph DSL) │                         │  (real CompiledStateGraph) │
└──────────────────┘                         └──────────────────────┘
        ▲                                              ▲
        │  to_flowise.to_dict                          │  compile.to_langgraph
        ▼                                              │
┌──────────────────┐    parse.flowise.from_file   ┌──────────────────────┐
│  Flowise V2 JSON │ ─────────────────────────►   │  IR  (single source  │
│  (canvas export) │ ◄─────────────────────────   │      of truth)       │
└──────────────────┘     compile.to_flowise        └──────────────────────┘
```

## Why it exists

| Audience | Pain | What this gives them |
|---|---|---|
| Python engineer | Wants real LangGraph runtime, but also wants their agent visible to non-developers | `g.compile()` returns vanilla LangGraph; `compile.to_flowise.to_file(g.ir, path)` ships a Flowise-importable JSON |
| Visual designer | Built an agent in Flowise but needs to deploy it without the canvas | `parse.flowise.from_file(path)` → real LangGraph runtime; or `compile.to_python.to_file(...)` for a cogflow-free `.py` they can maintain in code |
| Platform team | Wants self-hosted observability, no vendor lock-in | MLflow Tracing via cogflow's existing kubeflow infra |

## The contract

The SDK's `StateGraph` is a literal subclass of `langgraph.graph.StateGraph`.
`add_node`, `add_edge`, `add_conditional_edges`, `set_entry_point` and
`set_finish_point` are overridden to capture an in-memory **IR**
(intermediate representation) on the side; `compile()` does no extra
work — it returns the same `CompiledStateGraph` LangGraph itself would
produce.

The IR — separate from the runtime — is what enables Flowise JSON emit,
JSON import, and one-way migration to standalone Python source.

## Status (as of writing)

- **All 15 Flowise V2 AgentFlow node types** representable in IR
- **Three emit targets**: LangGraph runtime, Flowise V2 JSON, standalone Python source
- **MLflow Tracing** ships as the default observability backend; OpenTelemetry is an opt-in extra
- **LangSmith is not bundled** — it's a commercial service. Users who want it can set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` themselves and LangChain will wire it up at runtime

## Where to next

- New here? → **[Getting started](getting-started.md)**
- Want to understand why the design looks like this? → **[Architecture](architecture.md)**
- Looking up a specific node type? → **[Node types](nodes.md)**
- Migrating off Flowise? → **[Migration guide](migration.md)**
