"""Shared fixtures for cogflow.agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest


MARKETPLACE_DIR = Path(
    "/home/ali/project/coge/flow/src_flowise/Flowise/packages/server/marketplaces/agentflowsv2"
)


@pytest.fixture(scope="session")
def marketplace_dir() -> Path:
    return MARKETPLACE_DIR


@pytest.fixture(scope="session")
def simple_rag_path(marketplace_dir: Path) -> Path:
    return marketplace_dir / "Simple RAG.json"


@pytest.fixture(scope="session")
def structured_output_path(marketplace_dir: Path) -> Path:
    return marketplace_dir / "Structured Output.json"


@pytest.fixture(scope="session")
def translator_path(marketplace_dir: Path) -> Path:
    return marketplace_dir / "Translator.json"
