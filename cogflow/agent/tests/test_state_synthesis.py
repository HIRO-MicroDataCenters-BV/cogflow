"""State channel introspection + TypedDict synthesis."""

from __future__ import annotations

from typing import Annotated, TypedDict

from cogflow.agent.state import (
    add_messages,
    introspect_state,
    reducer_by_name,
    synthesize_typeddict,
)


class _Schema(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_id: str


def test_introspect_extracts_add_messages_reducer():
    fields = introspect_state(_Schema)
    by_key = {f.key: f for f in fields}
    assert by_key["messages"].reducer == "add_messages"
    assert by_key["user_id"].reducer is None


def test_messages_field_auto_upgraded_to_add_messages():
    class _Bare(TypedDict, total=False):
        messages: list

    fields = introspect_state(_Bare)
    by_key = {f.key: f for f in fields}
    assert by_key["messages"].reducer == "add_messages"


def test_synthesize_typeddict_roundtrips_through_introspect():
    fields = introspect_state(_Schema)
    synthesized = synthesize_typeddict(fields)
    refields = introspect_state(synthesized)
    assert sorted(f.key for f in refields) == sorted(f.key for f in fields)
    by_key = {f.key: f for f in refields}
    assert by_key["messages"].reducer == "add_messages"


def test_reducer_registry_resolves_known_names():
    assert reducer_by_name("add_messages") is add_messages
    assert reducer_by_name(None) is None
