"""Integration: load Phase-2 marketplace fixtures and verify compile.

These fixtures exercise the bridge node types added in Phase 2 — Iteration
and HumanInput — to confirm the IR -> LangGraph compiler doesn't choke
when those types appear in a real flow.
"""

from __future__ import annotations

import json
from pathlib import Path

from cogflow.agent.compile.to_flowise import to_dict as ir_to_flowise
from cogflow.agent.compile.to_langgraph import to_langgraph
from cogflow.agent.parse.flowise import from_file


def _normalize(value):
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(x) for x in value]
    return value


def test_iterations_fixture_parses_to_ir(iterations_path: Path):
    ir = from_file(iterations_path)
    types = {n.type for n in ir.nodes}
    # Iterations.json must surface the iteration bridge node — proves the
    # discriminator is wired through the new factory registry.
    assert "iteration" in types


def test_iterations_fixture_round_trips_through_emit(iterations_path: Path):
    """Lossless round-trip on the nodes/edges blocks for a bridge-node fixture."""
    original = json.loads(iterations_path.read_text())
    ir = from_file(iterations_path)
    emitted = ir_to_flowise(ir)
    # Structural equality on the lists; provenance carries everything else.
    assert _normalize(emitted["nodes"]) == _normalize(original["nodes"])
    assert _normalize(emitted["edges"]) == _normalize(original["edges"])


def test_iterations_fixture_compiles_to_langgraph(iterations_path: Path):
    """Compile must succeed end-to-end for graphs containing bridge nodes."""
    ir = from_file(iterations_path)
    app = to_langgraph(ir)
    assert app is not None


def test_human_in_the_loop_fixture_parses_to_ir(human_in_the_loop_path: Path):
    ir = from_file(human_in_the_loop_path)
    types = {n.type for n in ir.nodes}
    assert "human_input" in types


def test_human_in_the_loop_fixture_compiles(human_in_the_loop_path: Path):
    ir = from_file(human_in_the_loop_path)
    app = to_langgraph(ir)
    assert app is not None
