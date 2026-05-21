"""Apply ``llmUpdateState`` / ``agentUpdateState`` directives to a node delta.

Flowise's LLM and Agent nodes both expose an ``Update State`` slot — a list of
``{key, value}`` pairs that are evaluated against the model's response and
folded into the graph state alongside the response message. Both factories
(``LLMFactory``, ``AgentFactory``) accept the directives at construction time
and serialize them into config; this helper applies them at runtime so the
state actually mutates.

Templating semantics are Python ``str.format`` — the same widened exception
tuple used by ``direct_reply`` / ``human_input`` / ``http`` covers the four
failure modes (missing key, malformed braces, non-identifier state keys, and
type-incompatible state values). On any failure the literal value passes
through unrendered so the node still updates state instead of crashing.

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
    # Both ``response`` and ``output`` resolve to the model reply text — keeps
    # Python-first ergonomics aligned with Flowise's ``{{ output }}`` label
    # without needing to detect which spelling the user wrote.
    fmt_kwargs: dict[str, Any] = {"response": response_text, "output": response_text, **dict(state)}

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
