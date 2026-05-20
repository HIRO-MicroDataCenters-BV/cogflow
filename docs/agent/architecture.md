# Architecture

The SDK is one idea applied consistently: **the IR is the source of truth**,
the runtime is the upstream LangGraph stack, and three emit/import paths
plug into the IR.

## The shape

```text
                              parse.flowise.from_file
        Flowise V2 JSON  ──────────────────────────►  ┌───────────┐
        Flowise V2 JSON  ◄──────────────────────────  │           │
                              compile.to_flowise      │           │
                                                      │    IR     │
        Python (StateGraph)  ────► add_node/edge ───► │ (IRGraph) │
                                                      │           │
        compile.to_langgraph ◄───────────────────────│           │
                ─►  CompiledStateGraph                └───────────┘
                                                            │
                                                  compile.to_python
                                                            ▼
                                                  standalone .py file
```

## Why an IR

The naive design — wrap LangGraph internals at the runtime layer — would
have forced us to maintain a parallel runtime, drift from LangGraph's
own changes, and re-implement features (subgraphs, interrupts,
checkpoints) that already exist.

Instead, the SDK's `StateGraph` is a **subclass** of
`langgraph.graph.StateGraph` whose mutator methods (`add_node`,
`add_edge`, `add_conditional_edges`, `set_entry_point`,
`set_finish_point`) override the parent to do two things:

1. Mutate an in-memory `IRGraph` to record what was added.
2. Delegate to `super()` so the runtime side is identical to LangGraph's.

`g.compile()` calls `langgraph.graph.StateGraph.compile()` unmodified
and returns the same `CompiledStateGraph` object. There is no transpile
step, no proxy, no wrapping. The IR is a side-channel.

## The IR shape

```python
class IRGraph(BaseModel):
    schema_version: Literal["1.0"]
    name: str
    description: str | None
    state: list[IRStateField]      # the TypedDict / startState fields
    nodes: list[IRNode]
    edges: list[IREdge]
    entry: str | None              # node id or None
    finish: list[str]              # nodes that route to END
    viewport: dict[str, float] | None
    flowise_provenance: dict[str, Any] | None   # round-trip escape hatch
```

Every `IRNode` and `IREdge` also has its own `flowise_provenance` slot —
when a graph comes from a Flowise JSON, the parser stashes the raw node
/ edge dict there. Emitters use it to reconstruct the JSON byte-for-byte
for MVP-7 nodes (lossless round-trip).

For nodes whose runtime semantics we don't fully model (Loop,
Iteration, HTTP, Retriever, CustomFunction, HumanInput, ExecuteFlow,
Unknown), provenance still preserves the structural fields so the JSON
re-emit stays correct even when the runtime mapping is best-effort.

## The three emit paths

### `compile.to_langgraph(ir, *, factories=None, ctx=None, ...)`

Walks the IR, registers a runtime callable for each non-skipped node,
wires entry/finish/conditional/loop edges, and returns a
`CompiledStateGraph`. Skip types: `sticky_note`. `start` nodes have no
runtime body — `START` fans out from their outgoing edges.

`ctx` is a shared mapping passed to every factory's `to_callable(node,
ctx=...)`. Bridge nodes use it for runtime dependencies they couldn't
capture at construction time:

```python
to_langgraph(
    ir,
    ctx={
        "httpx": stub_or_real_client,        # HTTP node
        "retriever": my_vectorstore,         # Retriever node
        "flows": {"id": child_graph},        # ExecuteFlow node
        "allow_custom_code": False,          # CustomFunction gate (default False)
        "runtime_node_ids": ...,             # auto-seeded by the compiler
    },
)
```

### `compile.to_flowise.to_dict(ir, *, analytics=None) / to_file(ir, path, ...)`

Reconstructs the Flowise V2 AgentFlow JSON. Provenance-backed nodes/edges
re-emit verbatim; Python-built ones get a synthesized minimal-but-valid
block. The `analytics` kwarg injects Flowise's own analytic config — use
it to point a re-imported flow at the same MLflow your runtime uses.

Round-trip on the MVP-7 marketplace fixtures (`Simple RAG.json`,
`Structured Output.json`, `Translator.json`) is **structurally
identical** on the `nodes` / `edges` blocks — the round-trip tests
normalize dict/list ordering and compare for deep equality, so field
order may differ from the original but every field and value matches.
The `Iterations.json` and `Human In The Loop.json` bridge-node
fixtures round-trip with `flowise_provenance` preserving everything we
don't explicitly model.

### `compile.to_python.to_source(ir) / to_file(ir, path)`

One-way **Flowise → Python migration tool**, not a parallel runtime.
For Python-first graphs this is redundant — `g.compile()` already
returns a real `CompiledStateGraph`. Use it to lift a Flowise-authored
agent out of the canvas into a maintainable codebase. See
[Migration guide](migration.md) for what's emitted, what's stubbed,
and the secret-redaction behaviour.

## Validation

`compile.to_langgraph` runs `cogflow.agent.ir.validate.validate(graph)`
first, which enforces:

- At most one Start node
- Either a Start node exists OR `graph.entry` overrides
- `graph.entry` / `graph.finish` reference real nodes
- No duplicate node or edge ids
- Edge `source` / `target` point at real nodes (except for the
  `END_SENTINEL` target, which encodes branch-level termination on
  conditional edges)

## Drop-in import compatibility

The agent SDK's public surface re-exports LangGraph symbols where they
exist:

| LangGraph / LangChain | cogflow.agent equivalent |
|---|---|
| `langgraph.graph.StateGraph` | `cogflow.agent.StateGraph` (subclass) |
| `langgraph.graph.START`, `END` | `cogflow.agent.START`, `END` (identity) |
| `langgraph.graph.MessagesState`, `add_messages` | `cogflow.agent.MessagesState`, `add_messages` |
| `langgraph.prebuilt.create_react_agent`, `ToolNode` | `cogflow.agent.create_react_agent`, `cogflow.agent.tools.ToolNode` |
| `langgraph.types.interrupt`, `Command` | `cogflow.agent.interrupt`, `Command` |
| `langgraph.checkpoint.memory.MemorySaver`, `SqliteSaver` | `cogflow.agent.runtime.checkpoint.MemorySaver`, `SqliteSaver` |
| `langchain_core.tools.tool` | `cogflow.agent.tools.tool` |

Provider SDKs (`langchain_openai`, `langchain_anthropic`, message
classes, runnables) stay as direct deps — the SDK is provider-agnostic
and doesn't try to abstract them.
