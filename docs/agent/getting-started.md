# Getting started

This page walks two paths in five minutes: build an agent in Python,
then import the same shape from a Flowise canvas export.

## Install

```bash
pip install "cogflow[agent]"
```

This pulls in `langgraph` and `langchain-core`. Plain `pip install
cogflow` won't import `cogflow.agent` and will give you a clear hint
pointing back here.

For MLflow tracing through cogflow's existing tracking backend, no
additional install is needed — `mlflow` is part of base cogflow. For
OpenTelemetry: `pip install "cogflow[agent-otel]"`.

## Path A: Python-first

Build with `StateGraph` exactly as you would in LangGraph; the import
path is the only difference.

```python
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage

from cogflow.agent import END, START, StateGraph, add_messages


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]


def greet(state: State) -> dict:
    return {"messages": [AIMessage(content="hello")]}


g = StateGraph(State)
g.add_node("greet", greet)
g.add_edge(START, "greet")
g.add_edge("greet", END)

app = g.compile()                                  # langgraph.graph.CompiledStateGraph
result = app.invoke({"messages": [HumanMessage(content="hi")]})
print([m.content for m in result["messages"]])
# → ['hi', 'hello']
```

What just happened:

- `StateGraph(State)` is `langgraph.graph.StateGraph`'s subclass — no proxy, no transpile.
- `g.compile()` returns the same object LangGraph would. `app.invoke` is LangGraph's `invoke`.
- An IR was captured on the side as you called `add_node` / `add_edge`. Reach it via `g.ir`.

### Export to Flowise

```python
from cogflow.agent.compile import to_flowise

to_flowise.to_file(g.ir, "agent.json")   # AgentFlow V2 JSON, import in Flowise
```

The JSON is loadable by Flowise's V2 canvas. Use this when stakeholders
need a visual view of the agent or when handing off to a non-developer
to edit.

## Path B: Import from Flowise

Got a Flowise V2 AgentFlow JSON (marketplace, your team's canvas, a
customer flow)? Run it on LangGraph in one step:

```python
from cogflow.agent.parse import flowise
from cogflow.agent.compile import to_langgraph

ir = flowise.from_file("agent.json")
app = to_langgraph(ir)                    # real CompiledStateGraph
result = app.invoke({"messages": [("user", "hi")]})
```

### Wiring runtime objects into a JSON-loaded graph

JSON doesn't carry live chat models, tool callables, retriever stores,
or compiled child flows. Two injection points cover that:

- **`factories=`** — per-node, when you want to bind a specific
  factory instance to a single node id.
- **`ctx=`** — shared with every node, threaded through to each
  factory's `to_callable(node, ctx=ctx)`. Bridge nodes (HTTP, Retriever,
  ExecuteFlow, CustomFunction, …) read their runtime dependencies from
  here.

```python
from langchain_openai import ChatOpenAI

app = to_langgraph(
    ir,
    ctx={
        "retriever": my_vectorstore.as_retriever(),
        "flows": {"child_flow_id": child_compiled_graph},
        "allow_custom_code": False,   # default — CustomFunction Python bodies refuse to exec
    },
)
```

## Path C: One-way migration off Flowise

If you want to **leave Flowise entirely** and continue in plain Python,
emit a standalone source file:

```python
from cogflow.agent.parse import flowise
from cogflow.agent.compile import to_python

ir = flowise.from_file("agent.json")
to_python.to_file(ir, "my_agent.py")
```

`my_agent.py` has no `cogflow` import — just `langgraph` and
`langchain_core`. You can hand-edit it, ship it, fork it. See the
[Migration guide](migration.md) for the full details on what's emitted
and what's stubbed.

## Tracing in one line

```python
from cogflow.agent.tracing import configure_tracing

configure_tracing(backend="mlflow", experiment="my-agent")
```

Spans now flow to the MLflow tracking URI from `cogflow.config`. See
[Tracing](tracing.md) for OpenTelemetry and stacking.
