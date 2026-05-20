# Node types

All 15 Flowise V2 AgentFlow node types have a `cogflow.agent.nodes.*`
factory. The factory's `__init__` carries Python-first kwargs; the same
factory hydrates from JSON via `from_ir(ir_node)` using the
Flowise-native config keys.

## At a glance

| Flowise `data.name` | IR type | Python factory | Runtime |
|---|---|---|---|
| `startAgentflow` | `start` | `nodes.start(...)` | Native (no body — `START` fan-out) |
| `llmAgentflow` | `llm` | `nodes.llm(...)` | Native (ChatModel `.invoke`) |
| `agentAgentflow` | `agent` | `nodes.agent(...)` | Native (`bind_tools` + `.invoke`) |
| `toolAgentflow` | `tool` | `nodes.tool_node(...)` | Native (`@tool` wrapper) |
| `conditionAgentflow` | `condition` | `nodes.condition(...)` | Native (`add_conditional_edges`) |
| `conditionAgentAgentflow` | `condition_agent` | `nodes.condition_agent(...)` | Native (LLM-driven router) |
| `directReplyAgentflow` | `direct_reply` | `nodes.direct_reply(...)` | Native (writes `AIMessage`) |
| `loopAgentflow` | `loop` | `nodes.loop(...)` | Bridged (counter channel + back-edge router) |
| `iterationAgentflow` | `iteration` | `nodes.iteration(...)` | Bridged (`langgraph.types.Send` fan-out) |
| `httpAgentflow` | `http` | `nodes.http(...)` | Bridged (`httpx`) |
| `retrieverAgentflow` | `retriever` | `nodes.retriever(...)` | Bridged (LangChain Retriever) |
| `customFunctionAgentflow` | `custom_function` | `nodes.custom_function(...)` | Bridged (Python-only, gated) |
| `humanInputAgentflow` | `human_input` | `nodes.human_input(...)` | Bridged (`langgraph.types.interrupt`) |
| `executeFlowAgentflow` | `execute_flow` | `nodes.execute_flow(...)` | Bridged (nested `CompiledStateGraph.invoke`) |
| `stickyNoteAgentflow` | `sticky_note` | `nodes.sticky_note(...)` | Skipped at compile |

Unknown node types (not in the table) are preserved in IR with full
`flowise_provenance` and compile to a passthrough runtime body so
surrounding edges keep flowing.

## MVP-7 nodes

The seven "native" nodes have a clean LangGraph mapping.

### `start`

The entry point of an agentflow. No runtime body — `compile.to_langgraph`
wires `START` to whatever the Start node fans out to.

```python
from cogflow.agent.nodes import start

g.add_node("start_0", start(
    input_type="chatInput",
    ephemeral_memory=False,
    persist_state=False,
    state=[{"key": "user_id", "value": ""}],   # synthesises the FlowState
))
```

Flowise keys: `startInputType`, `startEphemeralMemory`, `startPersistState`, `startState`.

### `llm`

Single chat-model call, no tools.

```python
from cogflow.agent.nodes import llm

g.add_node("llm_0", llm(
    model=ChatOpenAI(model="gpt-4o-mini"),
    messages=[{"role": "system", "content": "You are concise."}],
    update_state=[{"key": "summary", "value": "{{output}}"}],
))
```

Flowise keys: `llmModel`, `llmMessages`, `llmUpdateState`, `llmReturnResponseAs`.

### `agent`

ReAct-style loop: a chat model with tools bound. Prepends configured
`agentMessages` to the model invocation; tool calls execute through
LangGraph's built-in tool resolution.

```python
from cogflow.agent.nodes import agent

g.add_node("agent_0", agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search, calculator],
    messages=[{"role": "system", "content": "You are helpful."}],
    enable_memory=True,
    memory_type="allMessages",
))
```

Flowise keys: `agentModel`, `agentMessages`, `agentTools`, `agentEnableMemory`, `agentMemoryType`, `agentReturnResponseAs`, `agentUpdateState`.

### `tool`

Exposes a callable for an Agent to invoke. For Python-first use, the
LangChain `@tool` decorator is usually the simpler path:

```python
from cogflow.agent.tools import tool

@tool
def search(query: str) -> str:
    "Search the web."
    return f"results for {query}"

# pass to an Agent: g.add_node("agent_0", agent(model=..., tools=[search]))
```

### `condition`

Deterministic rule-based router. Rules are evaluated in order; first
match writes the `_condition_branch` key that drives the conditional
edges.

```python
from cogflow.agent.nodes import condition

g.add_node("route_0", condition(rules=[
    {"variable": "sentiment", "operation": "Equal", "value": "positive", "output": "happy_path"},
    {"variable": "sentiment", "operation": "Equal", "value": "negative", "output": "support"},
]))
```

Operators: `Equal`, `NotEqual`, `Contains`, `NotContains`, `StartsWith`,
`EndsWith`, `RegexMatch`, `Greater`, `Less`. Operators coerce values to
strings where appropriate and swallow `re.error` / `TypeError` so a bad
rule yields no-match rather than crashing the graph.

### `condition_agent`

LLM-driven router. The model is asked to pick one scenario name; the
returned `_condition_branch` is the matched name.

```python
g.add_node("triage", condition_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    instructions="Categorize the user request.",
    scenarios=[
        {"name": "billing", "description": "Anything money-related."},
        {"name": "technical", "description": "Bug reports, errors."},
        {"name": "other", "description": "Catch-all."},
    ],
))
```

Matching is casefold-equality first, then whole-word containment, then
falls back to the first scenario.

### `direct_reply`

Terminal node — writes an `AIMessage` and routes to `END`. The message
is `.format(**state)`-templated and falls back to the raw template on
any `KeyError`/`IndexError`/`ValueError`/`TypeError`.

```python
g.add_node("reply", direct_reply(message="Hi {user_name}, here's your summary."))
```

## Bridge nodes

These have no direct LangGraph primitive; the SDK bridges them to
LangGraph patterns. Runtime behaviour is documented per node.

### `loop`

Conditional back-edge with a synthesized `_loop_count__<node_id>`
channel.

```python
g.add_node("loop_0", loop(target="agent_0", max_iterations=5))
```

The compiler reads both Python-first (`loopTarget`, `loopMaxIterations`)
and Flowise-native (`loopBackToNode`, `maxLoopCount`) keys. Flowise's
`loopBackToNode` encodes `{node_id}-{label}`; the label suffix is
stripped if the raw value doesn't resolve to a registered runtime id.
Unresolvable targets fall through to the exit edge (or `END`) so the
compiled graph never returns an unknown node id.

### `iteration`

Fan-out via `langgraph.types.Send`. Reads `iterationOver` from state
(list or callable); for JSON imports the parser also walks Flowise's
`parentNode` references and stashes the body node id under
`flowise_provenance.iteration_body_id`.

```python
g.add_node("for_each", iteration(over="items", body="worker_node"))
```

Body ids are validated against the compiler's runtime node set; an
unresolvable body silently degrades to a no-op rather than emitting
`Send(missing_id, ...)`.

### `http`

Single REST call via `httpx`. URL is `.format(**state)`-templated;
missing keys fall back to the unrendered URL.

```python
g.add_node("call_api", http(
    method="POST",
    url="https://api.example.com/v1/{endpoint}",
    headers={"Authorization": "Bearer {token}"},
    body={"q": 1},
    output_key="response",
))
```

Inject a stub client for tests: `ctx={"httpx": stub_module}`.

### `retriever`

Wraps a LangChain Retriever. Inject the live retriever either at
construction (`retriever=...`) or via `ctx={"retriever": ...}` for
JSON-loaded graphs.

```python
g.add_node("rag", retriever(
    retriever=my_vectorstore.as_retriever(),
    k=4,
    output_key="docs",
))
```

### `custom_function`

**Python-only, gated.** JavaScript bodies are rejected at construction
(or hydration) with `UnsupportedNodeError`. Python string bodies only
execute when the caller opts in via `ctx={"allow_custom_code": True}`,
and the namespace is restricted to a small set of safe builtins — **not
a security sandbox**.

```python
def double_x(state: dict) -> int:
    return state["x"] * 2

g.add_node("calc", custom_function(fn=double_x, output_key="doubled"))
```

When `fn=` is supplied, `customFunctionPython` is left empty and the
function name is stored in `customFunctionName` for canvas display.
This prevents a re-import from accidentally trying to `exec` the
function's name as code.

### `human_input`

Wraps `langgraph.types.interrupt`. Requires a checkpointer on
`compile()` (e.g., `MemorySaver`) so the suspended state can be
resumed via `Command(resume=...)`.

```python
g.add_node("approve", human_input(
    prompt="Approve {action}?",
    output_key="user_decision",
))
```

Honours both `humanInputPrompt` (Python-first) and `humanInputDescription`
(Flowise-native) for the prompt text. Prompt is `.format(**state)`-templated
with the same exception fallback as `direct_reply`.

### `execute_flow`

Invokes a nested `CompiledStateGraph`. Inject the child graph at
construction (`graph=...`) or via `ctx={"flows": {flow_id: graph}}`
for JSON-loaded graphs.

```python
g.add_node("sub", execute_flow(
    flow_id="child_flow_id",
    input_keys=["query"],            # filter state down to these keys, or omit for full state
    output_key="subflow_result",
))
```

### `sticky_note`

Pure documentation node. Preserved in IR for round-trip; skipped at
LangGraph compile.
