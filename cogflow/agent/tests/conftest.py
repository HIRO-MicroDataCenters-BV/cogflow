"""Shared fixtures for cogflow.agent tests.

Marketplace JSON fixtures are vendored under ``fixtures/`` so the test suite
runs anywhere — no dependence on an external Flowise checkout.

The whole suite skips early if ``langgraph`` isn't installed, so a plain
``pytest`` run on a base ``cogflow`` install (without the ``agent`` extra)
collects without failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Ensure agent SDK deps are available before collecting any tests in this dir.
pytest.importorskip("langgraph", reason="cogflow.agent tests require `pip install cogflow[agent]`")


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def marketplace_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def simple_rag_path(marketplace_dir: Path) -> Path:
    return marketplace_dir / "simple_rag.json"


@pytest.fixture(scope="session")
def structured_output_path(marketplace_dir: Path) -> Path:
    return marketplace_dir / "structured_output.json"


@pytest.fixture(scope="session")
def translator_path(marketplace_dir: Path) -> Path:
    return marketplace_dir / "translator.json"
