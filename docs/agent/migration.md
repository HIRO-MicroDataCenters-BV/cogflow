# Migrating off Flowise

`compile.to_python` is a **one-way migration tool**, not a parallel
runtime. Use it to move a Flowise-authored agent out of the visual
canvas into a maintainable Python codebase with **no runtime dependency
on cogflow** — just `langgraph` and `langchain_core`.

## When to use this

```python
# Got a Flowise JSON (visual designer, marketplace, customer flow)?
# Lift it to maintainable code:
from cogflow.agent.parse import flowise
from cogflow.agent.compile import to_python

ir = flowise.from_file("my_flow.json")
to_python.to_file(ir, "my_agent.py")
```

`my_agent.py` is now a self-contained module the team can fork, modify,
ship to environments that don't have cogflow installed, and run through
`black`/`ruff` cleanly.

## When *not* to use this

If you authored the graph in Python with `cogflow.agent.StateGraph`,
this is redundant. Your `StateGraph` is a literal subclass of
`langgraph.graph.StateGraph`; `g.compile()` already returns a real
`CompiledStateGraph`. Just call it:

```python
g = StateGraph(MyState)
g.add_node(...)
g.add_edge(START, END)
app = g.compile()                       # real CompiledStateGraph
app.invoke({"messages": [...]})        # runs on LangGraph directly
```

No emit step needed.

## What gets emitted

The emitted file is **always syntactically valid Python that execs
cleanly** for every Flowise V2 node type.

| Class | Nodes | Emitted body |
|---|---|---|
| MVP-7 + Loop + HumanInput | Start, LLM, Agent, Tool, Condition, ConditionAgent, DirectReply, Loop, HumanInput | Working — references user-supplied `MODEL` / `TOOLS` symbols where needed |
| Bridge stubs | HTTP, Retriever, CustomFunction body, ExecuteFlow, Iteration | `TODO: cogflow.agent bridge node <type> — implement manually` + IR config dump (secrets redacted) |
| Sticky note | StickyNote | Skipped entirely |
| Unknown | Unknown | Passthrough — keeps surrounding edges flowing |

## Wiring after emit

The emitted module declares:

```python
MODEL: Any = None
TOOLS: list[Any] = []
```

Before invoking `app`, set these to your real chat model and tools:

```python
import my_agent
from langchain_openai import ChatOpenAI

my_agent.MODEL = ChatOpenAI(model="gpt-4o-mini")
my_agent.TOOLS = [my_search_tool, my_calculator_tool]

result = my_agent.app.invoke({"messages": [...]})
```

For bridge-node stubs, replace the `TODO` body with your implementation
referencing the IR config dump that's already inlined as a comment.

## Secret redaction

Flowise inputs legitimately include API keys, bearer tokens, auth
headers, and passwords. The emitter walks `IRNode.config` recursively
and replaces any value whose key contains (case-insensitive) `key`,
`token`, `secret`, `password`, `passwd`, `auth`, `authorization`,
`credential`, `apikey`, `bearer`, or `private` with `<REDACTED>`. This
catches:

- Top-level keys: `apiKey`, `bearerToken`, `password`
- Nested dict values: `httpHeaders.x-api-key`, `headers.Authorization`
- List elements with secret-keyed siblings

Non-secret config (URL, method, model name, etc.) stays visible so the
TODO stub is still informative.

!!! note "Conservative over-redaction"

    Matching is by **substring**, so any config key containing
    `key`/`token`/`secret`/`password`/`auth`/… is redacted — including
    legitimately non-sensitive fields like `httpOutputKey` or
    `customFunctionOutputKey`. This is intentional: leaking a token is
    worse than hiding an output-key. If you need an over-redacted value
    in the emitted TODO stub, recover it from the original Flowise JSON.

## What this doesn't do

- It doesn't run the agent — the emitted file does that.
- It doesn't preserve Flowise-only artefacts that LangGraph has no
  equivalent for (canvas layout, sticky-note text, analytics blocks).
  Use the IR's `flowise_provenance` if you need those — they live in
  the IR, just not the emit.
- It doesn't replace `compile.to_langgraph`. For most users, the JSON
  → LangGraph runtime path is faster and doesn't require hand-tuning.

## Custom `app_var` name

```python
to_python.to_source(ir, app_var="customer_support_agent")
```

The output's final line becomes `customer_support_agent = builder.compile()`.
The argument is validated as a Python identifier (and not a keyword)
so the "always valid Python" contract holds.

## Round-trip implications

`compile.to_python` is **one-way**. Emitting Python doesn't write back
to a Flowise JSON automatically. If you need both, keep the original
JSON around and emit Python only when you're ready to leave Flowise.
