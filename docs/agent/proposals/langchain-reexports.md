# Proposal: Re-export the remaining LangChain / LangGraph primitives from `cogflow.agent`

**Status:** first slice implemented — see PR #100. Two naming refinements
landed in implementation: ``testing`` → ``fakes`` (avoids reading as "test
suite for the SDK"); ``models`` → ``chat_models`` (avoids colliding with
``cogflow.models``, the MLflow-tracked ML model registry). ``openai`` was
folded into the first PR to unblock the hiro-iac port; other providers
follow on demand.
**Scope:** `cogflow/agent/` only (new submodules + optional extras).
**Motivation:** every callable a `cogflow.agent` user needs should be reachable
via `cogflow.agent.*`, so downstream graphs can be authored without importing
`langchain*` or `langgraph*` directly.

## Background

`cogflow.agent` already re-exports the LangGraph-shaped graph surface
(`StateGraph`, `START`, `END`, `MessagesState`, `add_messages`,
`create_react_agent`, `interrupt`, `Command`, `ToolNode`, `tool`) plus the
runtime helpers (`invoke`, `stream`, `ainvoke`, `astream`, `MemorySaver`,
`SqliteSaver`). That lets you build a graph end-to-end with cogflow imports
only.

Porting an existing LangGraph application to cogflow (real example:
[hiro-iac](https://github.com/HIRO-MicroDataCenters-BV/agentic-iac) `cogflow`
branch) surfaces a handful of additional primitives that are commonly used in
agent code but are *not* yet re-exported. Today the port keeps these as
direct `langchain*` / `langgraph*` imports — the audit table below lists every
one. Adding the re-exports proposed in this doc lets the port (and every
future cogflow agent) reach zero direct LangChain/LangGraph imports.

## Gap audit (from the hiro-iac port)

| Direct import in caller code | Used for | Proposed re-export location |
|---|---|---|
| `langchain_core.messages.{BaseMessage, AIMessage, HumanMessage, SystemMessage, AIMessageChunk, ToolMessage}` | typing `chat_history`, constructing message records, filtering streamed chunks in CLI / Chainlit | **new** `cogflow.agent.messages` |
| `langchain_core.prompts.{ChatPromptTemplate, MessagesPlaceholder, PromptTemplate}` | building `prompt | llm` chains in agent nodes | **new** `cogflow.agent.prompts` |
| `langchain_core.runnables.{Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough, RunnableParallel}` | return-type annotation for chain factories, composition primitives | **new** `cogflow.agent.runnables` |
| `langchain_core.language_models.fake_chat_models.{FakeListChatModel, GenericFakeChatModel}` | scripted LLM responses in unit tests | **new** `cogflow.agent.testing` |
| `langgraph.checkpoint.base.BaseCheckpointSaver` | `isinstance(checkpointer, BaseCheckpointSaver)` checks + type annotations for optional checkpointer kwargs | add to existing `cogflow.agent.runtime` |
| `langchain_openai.ChatOpenAI` (parallel: `langchain_anthropic.ChatAnthropic`, `langchain_google_genai.ChatGoogleGenerativeAI`, `langchain_ollama.ChatOllama`, …) | the production chat model instantiated by the agent | **new** `cogflow.agent.models.<provider>` per-provider submodules, gated by `cogflow[<provider>]` install extras |

## Proposed code

All new modules are thin re-exports. The only file with real logic is the
chat-model gate (which raises a friendly error if the underlying provider
package isn't installed).

### `cogflow/agent/messages.py` (new)

```python
"""Re-exports of LangChain message classes used by every agent graph."""
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

__all__ = [
    "AIMessage",
    "AIMessageChunk",
    "BaseMessage",
    "HumanMessage",
    "SystemMessage",
    "ToolMessage",
]
```

### `cogflow/agent/prompts.py` (new)

```python
"""Re-exports of LangChain prompt templates."""
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

__all__ = ["ChatPromptTemplate", "MessagesPlaceholder", "PromptTemplate"]
```

### `cogflow/agent/runnables.py` (new)

```python
"""Re-exports of LangChain Runnable composition primitives."""
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

__all__ = [
    "Runnable",
    "RunnableConfig",
    "RunnableLambda",
    "RunnableParallel",
    "RunnablePassthrough",
]
```

### `cogflow/agent/testing.py` (new)

```python
"""Test doubles for cogflow.agent users."""
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)

__all__ = ["FakeListChatModel", "GenericFakeChatModel"]
```

### `cogflow/agent/runtime/__init__.py` (one-line addition)

Append `BaseCheckpointSaver` to the existing checkpointer re-exports:

```python
# (existing imports …)
from langgraph.checkpoint.base import BaseCheckpointSaver  # NEW

__all__ = [
    # existing entries …
    "BaseCheckpointSaver",
]
```

### `cogflow/agent/models/__init__.py` (new namespace)

```python
"""Chat model re-exports, one submodule per provider.

Each provider lives behind an optional install extra so the bare
``cogflow[agent]`` install doesn't drag in every LangChain provider
package.
"""
```

### `cogflow/agent/models/openai.py` (new)

```python
try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError as _exc:  # pragma: no cover - install-time guard
    raise ModuleNotFoundError(
        "cogflow.agent.models.openai requires `langchain-openai`. "
        "Install with `pip install cogflow[openai]`."
    ) from _exc

__all__ = ["ChatOpenAI"]
```

Parallel one-file modules for the other providers (each gated the same way):

| File | Import | Install extra |
|---|---|---|
| `cogflow/agent/models/anthropic.py` | `from langchain_anthropic import ChatAnthropic` | `cogflow[anthropic]` |
| `cogflow/agent/models/google.py` | `from langchain_google_genai import ChatGoogleGenerativeAI` | `cogflow[google]` |
| `cogflow/agent/models/ollama.py` | `from langchain_ollama import ChatOllama` | `cogflow[ollama]` |
| `cogflow/agent/models/vertex.py` | `from langchain_google_vertexai import ChatVertexAI` | `cogflow[vertex]` |

(Add providers incrementally based on demand — start with `openai` and
`anthropic`; the others can land in follow-up PRs.)

### `cogflow/agent/__init__.py` (touch)

No re-export at the top level — keep `cogflow.agent.*` the "graph surface"
and let users reach for `cogflow.agent.messages.AIMessage` etc. explicitly.
This keeps the top-level namespace small and mirrors how
`cogflow.agent.tools` and `cogflow.agent.runtime` already work.

If desired, the new submodules can be added to the import-side-effect list:

```python
# existing
from . import compile, nodes, parse, tracing
# add
from . import messages, models, prompts, runnables, testing
```

…so that `import cogflow.agent` makes them discoverable without an explicit
import. Optional — pick the convention that matches the rest of the project.

### `pyproject.toml` (extras)

```toml
[project.optional-dependencies]
agent     = ["langchain-core>=0.3.0", "langgraph>=0.2.0"]
openai    = ["langchain-openai>=0.2.0"]
anthropic = ["langchain-anthropic>=0.2.0"]
google    = ["langchain-google-genai>=2.0.0"]
ollama    = ["langchain-ollama>=0.2.0"]
vertex    = ["langchain-google-vertexai>=2.0.0"]
```

The existing `cogflow[agent]` extra is unchanged. Provider extras are
additive: `pip install cogflow[agent,openai]` brings in the OpenAI chat model
on top of the agent SDK.

## Caller-side diff after this PR lands

For the hiro-iac port, the audit becomes:

| Today (direct `langchain*` import) | After cogflow PR |
|---|---|
| `from langchain_core.messages import BaseMessage` | `from cogflow.agent.messages import BaseMessage` |
| `from langchain_core.messages import AIMessage, HumanMessage` | `from cogflow.agent.messages import AIMessage, HumanMessage` |
| `from langchain_core.messages import AIMessageChunk` | `from cogflow.agent.messages import AIMessageChunk` |
| `from langchain_core.prompts import ChatPromptTemplate` | `from cogflow.agent.prompts import ChatPromptTemplate` |
| `from langchain_core.runnables import Runnable` | `from cogflow.agent.runnables import Runnable` |
| `from langgraph.checkpoint.base import BaseCheckpointSaver` | `from cogflow.agent.runtime import BaseCheckpointSaver` |
| `from langchain_core.language_models.fake_chat_models import FakeListChatModel` | `from cogflow.agent.testing import FakeListChatModel` |
| `from langchain_openai import ChatOpenAI` | `from cogflow.agent.models.openai import ChatOpenAI` |

Net: zero direct `langchain*` / `langgraph*` imports in the downstream agent.

## Tests

Each new re-export module gets a single import-smoke test under
`cogflow/agent/tests/`:

```python
# tests/test_reexports.py
def test_messages_reexports():
    from cogflow.agent.messages import (
        AIMessage, AIMessageChunk, BaseMessage,
        HumanMessage, SystemMessage, ToolMessage,
    )
    assert AIMessage(content="x").content == "x"

def test_prompts_reexports():
    from cogflow.agent.prompts import ChatPromptTemplate
    t = ChatPromptTemplate.from_messages([("system", "hi"), ("human", "{q}")])
    assert t.invoke({"q": "ok"}).to_messages()[1].content == "ok"

def test_runnables_reexports():
    from cogflow.agent.runnables import Runnable, RunnableLambda
    r: Runnable = RunnableLambda(lambda x: x + 1)
    assert r.invoke(1) == 2

def test_testing_reexports():
    from cogflow.agent.testing import FakeListChatModel
    m = FakeListChatModel(responses=["a", "b"])
    assert m.invoke("x").content == "a"

def test_runtime_basecheckpointsaver_reexport():
    from cogflow.agent.runtime import BaseCheckpointSaver, MemorySaver
    assert issubclass(MemorySaver, BaseCheckpointSaver)
```

Provider-model modules are tested only for the gated `ModuleNotFoundError`
shape:

```python
# tests/test_models_openai.py
def test_openai_reexport_raises_friendly_error_when_uninstalled(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "langchain_openai", None)  # simulate uninstall
    with pytest.raises(ModuleNotFoundError, match="cogflow\\[openai\\]"):
        import importlib
        importlib.reload(importlib.import_module("cogflow.agent.models.openai"))
```

## Rollout

1. Land the four trivial re-export modules (`messages`, `prompts`,
   `runnables`, `testing`) plus the `BaseCheckpointSaver` line — these have
   no install-extra story and unblock the bulk of caller-side rewrites.
2. Land `cogflow.agent.models.openai` + the `cogflow[openai]` extra so the
   first downstream agent (hiro-iac) can finish the migration.
3. Add other provider modules in follow-ups, driven by which providers
   downstream cogflow agents actually use.

## Non-goals

- No new behavior, no wrappers, no validation layers. Every export in this
  PR is a verbatim re-export.
- No deprecation of direct `langchain*` imports. Callers can still reach
  into LangChain directly if they want; the re-exports are an *option*.
- No changes to `StateGraph`, `compile`, `parse`, `nodes`, `tracing`, or
  the IR. This proposal is purely about surface-area completeness.

## References

- Caller-side audit produced from
  [hiro-iac (cogflow branch)](https://github.com/HIRO-MicroDataCenters-BV/agentic-iac/tree/cogflow):
  `grep -rn -E "^(from|import) (langchain|langgraph)" src/ tests/ app.py`.
- Existing re-export pattern in `cogflow/agent/tools.py` (the friendly
  `ModuleNotFoundError` shape used for the optional `langchain-core` /
  `langgraph` install gates) is the template for the new provider modules.
