"""Lossless round-trip: Flowise V2 JSON -> IR -> Flowise V2 JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogflow.agent.compile.to_flowise import to_dict as ir_to_flowise
from cogflow.agent.parse.flowise import from_file


def _normalize(value):
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(x) for x in value]
    return value


def _load(path: Path):
    return json.loads(path.read_text())


def test_simple_rag_roundtrips_losslessly(simple_rag_path: Path):
    original = _load(simple_rag_path)
    ir = from_file(simple_rag_path)
    emitted = ir_to_flowise(ir)
    # Nodes and edges deep-equal modulo dict ordering.
    assert _normalize(emitted["nodes"]) == _normalize(original["nodes"])
    assert _normalize(emitted["edges"]) == _normalize(original["edges"])
    if "description" in original:
        assert emitted.get("description") == original["description"]
    if "usecases" in original:
        assert emitted.get("usecases") == original["usecases"]


@pytest.mark.parametrize("filename", ["structured_output.json", "translator.json"])
def test_other_mvp7_fixtures_roundtrip(marketplace_dir: Path, filename: str):
    path = marketplace_dir / filename
    if not path.exists():
        pytest.skip(f"fixture {filename} not present in this checkout")
    original = _load(path)
    ir = from_file(path)
    emitted = ir_to_flowise(ir)
    assert _normalize(emitted["nodes"]) == _normalize(original["nodes"])
    assert _normalize(emitted["edges"]) == _normalize(original["edges"])
