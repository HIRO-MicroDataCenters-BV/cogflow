"""Apply ``llmUpdateState`` / ``agentUpdateState`` directives to a node delta.

Flowise's LLM and Agent nodes both expose an ``Update State`` slot — a list of
``{key, value}`` pairs that are evaluated against the model's response and
folded into the graph state alongside the response message. Both factories
(``LLMFactory``, ``AgentFactory``) accept the directives at construction time
and serialize them into config; this helper applies them at runtime so the
state actually mutates.

Templating semantics are Python ``str.format`` — the same widened exception
tuple used by ``direct_reply`` / ``human_input`` / ``http`` covers three
documented modes:

  - ``KeyError`` / ``IndexError`` — template references a missing keyword
    or positional slot.
  - ``ValueError`` — malformed template (unmatched brace) or a builtin
    type rejecting the format spec (``"{x:d}".format(x="hi")``).
  - ``TypeError`` — caught defensively. Most common cause is a value in
    state whose custom ``__format__`` raises on a particular spec.

On any failure the literal value passes through unrendered so the node
still updates state instead of crashing. Extra unreferenced state keys
are inert regardless of their shape — only what the template
interpolates matters.

For JSON-loaded flows whose ``value`` strings use Flowise's mustache-style
``{{ output }}`` / ``{{ $flow.state.X }}`` placeholders, the strings simply
flow through as literals — IR round-trip is preserved via ``flowise_provenance``,
but runtime substitution of those tokens is intentionally out of scope here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage


def apply_update_state(
    directives: Any,
    response: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the state delta produced by evaluating ``directives``.

    ``directives`` may be the literal ``""`` (Flowise's "unset" marker),
    ``None``, or a list of ``{"key": str, "value": str}`` dicts. The
    ``response`` argument is the model's reply (a BaseMessage or any object
    whose ``str()`` form is meaningful); it's surfaced to format strings as
    the ``response`` and ``output`` placeholders so either label works.
    """
    if not directives or not isinstance(directives, list):
        return {}

    response_text = response.content if isinstance(response, BaseMessage) else str(response)
    # Splat state first, then overwrite ``response`` / ``output`` so a prior
    # state entry under those keys can't shadow the reserved placeholders —
    # ``{response}`` in a template must always refer to the current model
    # reply, never to whatever a previous node stashed under ``state["response"]``.
    fmt_kwargs: dict[str, Any] = {**dict(state), "response": response_text, "output": response_text}

    updates: dict[str, Any] = {}
    for entry in directives:
        if not isinstance(entry, Mapping):
            continue
        key = entry.get("key")
        value = entry.get("value")
        if not isinstance(key, str) or not key:
            continue
        if isinstance(value, str):
            try:
                updates[key] = value.format(**fmt_kwargs)
            except (KeyError, IndexError, ValueError, TypeError):
                updates[key] = value
        else:
            updates[key] = value
    return updates


__all__ = ["apply_update_state"]
