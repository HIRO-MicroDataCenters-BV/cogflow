# cogflow

Cognitive Framework — modular ML workflow management plus a LangGraph-shaped agent SDK.

## What's in this site (today)

The Agent SDK (`cogflow.agent`). The pipeline/ML side of cogflow is
documented from docstrings in the source for now; that may join this
site later.

- **[Agent SDK Overview](agent/index.md)** — what it is, why it exists
- **[Getting started](agent/getting-started.md)** — Python-first walkthrough + Flowise import in five minutes
- **[Architecture](agent/architecture.md)** — IR-centred design, three emit targets
- **[Node types](agent/nodes.md)** — all 15 Flowise V2 AgentFlow node types, their Python factories, and the Flowise config keys they read
- **[State & channels](agent/state.md)** — TypedDict ↔ Flowise `startState`, reducers, `add_messages`
- **[Tracing](agent/tracing.md)** — MLflow default (self-hosted), OpenTelemetry opt-in
- **[Migrating off Flowise](agent/migration.md)** — using `compile.to_python` to lift an agent out of the visual canvas

## Install

The agent SDK ships as an optional extra of cogflow:

```bash
pip install "cogflow[agent]"        # core: langgraph + langchain-core
pip install "cogflow[agent-otel]"   # adds OpenTelemetry tracing (agent-otel already depends on agent)
```

Plain `pip install cogflow` works too — the agent submodule just refuses
to import without the extra installed and tells you what to install.

## Quick example

```python
from cogflow.agent import StateGraph, MessagesState, START, END
from cogflow.agent.tools import tool


@tool
def search(query: str) -> str:
    "Search the web."
    return f"results for {query}"


g = StateGraph(MessagesState)
g.add_node("agent", lambda state: {"messages": []})
g.add_edge(START, "agent")
g.add_edge("agent", END)

app = g.compile()                 # real langgraph.graph.CompiledStateGraph
result = app.invoke({"messages": [("user", "hi")]})
```

If you've used LangGraph, that's literally the same code with a different
import path. The SDK's `StateGraph` is a subclass that captures an IR
on the side, which unlocks Flowise JSON emit / import — see
[Architecture](agent/architecture.md).
