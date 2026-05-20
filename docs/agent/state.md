# State & channels

Graph state has the same shape LangGraph users expect: a TypedDict whose
fields can carry reducer annotations (`Annotated[T, reducer]`). The SDK
bridges this to Flowise's `startState` list-of-`{key, value}` form on
both ends.

## TypedDict in, TypedDict out

```python
import operator
from typing import Annotated, TypedDict

from cogflow.agent import StateGraph, MessagesState, add_messages


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_id: str
    step_count: Annotated[int, operator.add]


g = StateGraph(State)
```

The SDK calls `typing.get_type_hints(State, include_extras=True)` to
extract the reducers, records them as `IRStateField(reducer="add_messages")`
etc., and normalizes the `messages` key to
`type="messages" + reducer="add_messages"` regardless of whether the
user annotated the reducer.

!!! note "Reducers must be **named** to round-trip"

    `introspect_state` only records reducers it can identify by name
    via the registry — `add_messages`, `operator.add`, and anything
    registered via [`register_reducer(...)`](#reducers). A bare lambda
    (`Annotated[int, lambda a, b: a + b]`) **runs correctly at
    runtime** but won't carry through IR / Flowise — the IR field gets
    no `reducer` and downstream emit paths fall back to overwrite
    semantics. Use `operator.add` / a `register_reducer`-named
    function for round-trip-safe reducers.

## Synthesizing a TypedDict from Flowise state

A Flowise Start node's `startState` is a list of `{key, value}` entries:

```json
{
  "data": {
    "inputs": {
      "startState": [
        {"key": "user_id", "value": ""},
        {"key": "step_count", "value": 0}
      ]
    }
  }
}
```

`parse.flowise.from_file` reads that into a list of `IRStateField`s.
`compile.to_langgraph` then calls
`cogflow.agent.state.synthesize_typeddict(fields)` which builds an
actual `TypedDict` class at runtime, reattaching any known reducers
(`add_messages`, `operator.add`).

The `messages` field is always upgraded to `type="messages" + reducer="add_messages"`
even if the user didn't declare it in `startState`, so chat semantics
work regardless of whether the Flowise designer remembered to add it.

## Round-trip

| Direction | What happens |
|---|---|
| `StateGraph(TypedDict)` → IR | `introspect_state(...)` extracts fields + reducers via `get_type_hints` |
| Flowise JSON → IR | `_state_from_start(...)` reads the Start node's `startState` array |
| IR → LangGraph runtime | `synthesize_typeddict(...)` builds a real TypedDict; reducers reattached |
| IR → Flowise JSON | `_start_state_payload(...)` materialises the array back onto the Start node's `data.inputs.startState` (only for Python-built Start nodes — provenance-backed ones keep their original `startState` untouched for lossless round-trip) |

## Reducers

Registered reducers (declared once, reusable across graphs):

| Name | Maps to |
|---|---|
| `add_messages` | `langgraph.graph.add_messages` (intelligent message-list merger) |
| `operator.add` | `operator.add` (list/int concatenation) |

Register a custom one and it becomes available to both directions of
state synthesis:

```python
from cogflow.agent.state import register_reducer

register_reducer("my_reducer", my_reducer_fn)
```

## What survives Flowise round-trip

For an IR field with `reducer="add_messages"`, the emitted Flowise
`startState` entry contains `{"key": "messages", "value": ""}`. Flowise
doesn't model reducers — but the SDK's importer always re-upgrades the
`messages` field on the way back in, so a Python → Flowise → Python
round-trip preserves chat semantics. Custom reducers (e.g.,
`operator.add` on a counter) need to be re-declared in the receiving
codebase; they don't survive Flowise as data.

## Functional vs class-syntax TypedDict (emit path)

`compile.to_python.to_source` emits the **functional**
`TypedDict("FlowState", {...}, total=False)` form (not class syntax) so
state keys like `"user id"` or `"step-count"` — legal in Flowise's JSON
— produce valid Python. The class-syntax form would have rejected those
keys at parse time.
