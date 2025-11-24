"""
Unit tests for cogflow.core.datasets.DatasetManager.

All external dependencies are patched:
- network.make_get_request
- common.normalize_uuid
- common.get_current_user
"""

from unittest.mock import MagicMock
import pytest

from cogflow.core.datasets import DatasetManager
from cogflow.utils import common, network
from cogflow.utils.exceptions import (
    CogflowConnectionError,
    CogflowArtifactError,
    CogflowValidationError,
    CogflowDatasetError,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def manager(monkeypatch):
    """
    Returns a fresh DatasetManager with all external dependencies mocked.
    """

    # Fake UUID normalization
    monkeypatch.setattr(
        common,
        "normalize_uuid",
        lambda v: str(v),  # noqa: E731  # pylint: disable=unnecessary-lambda
        raising=True,
    )

    # Fake current user
    monkeypatch.setattr(
        common, "get_current_user", lambda: "test-user", raising=True
    )  # pylint: disable=unnecessary-lambda

    # Stub API namespace
    dm = DatasetManager()
    return dm


# Helper function to patch network API
def patch_network_get(monkeypatch, response=None, side_effect=None):
    """
    Patch network.make_get_request to either:
    - return `response` dict, or
    - raise `side_effect`
    """
    if side_effect:
        monkeypatch.setattr(
            network,
            "make_get_request",
            MagicMock(side_effect=side_effect),
            raising=True,
        )
    else:
        monkeypatch.setattr(
            network, "make_get_request", MagicMock(return_value=response), raising=True
        )


# ---------------------------------------------------------------------
# Tests: get_dataset
# ---------------------------------------------------------------------


def test_get_dataset_success(manager, monkeypatch):
    """get_dataset returns resp['data'] when API responds correctly."""
    patch_network_get(monkeypatch, response={"data": {"a": 1}})

    result = manager.get_dataset("123")
    assert result == {"a": 1}


def test_get_dataset_invalid_uuid(manager, monkeypatch):
    """Invalid UUID should raise CogflowValidationError."""

    def bad_uuid(_):
        raise ValueError("invalid")

    monkeypatch.setattr(common, "normalize_uuid", bad_uuid, raising=True)

    with pytest.raises(CogflowValidationError):
        manager.get_dataset("BAD")


def test_get_dataset_network_failure(manager, monkeypatch):
    """Network failures should raise CogflowConnectionError."""
    patch_network_get(monkeypatch, side_effect=Exception("timeout"))

    with pytest.raises(CogflowConnectionError):
        manager.get_dataset("111")


def test_get_dataset_missing_data_field(manager, monkeypatch):
    """Responses without 'data' should raise CogflowArtifactError."""
    patch_network_get(monkeypatch, response={"other": 1})

    with pytest.raises(CogflowArtifactError):
        manager.get_dataset("222")


def test_get_dataset_empty_data_error(manager, monkeypatch):
    """Empty response['data'] triggers CogflowDatasetError."""
    patch_network_get(monkeypatch, response={"data": None})

    with pytest.raises(CogflowDatasetError):
        manager.get_dataset("333")


# ---------------------------------------------------------------------
# Tests: get_prometheus_dataset
# ---------------------------------------------------------------------


def test_get_prometheus_success(manager, monkeypatch):
    """get_prometheus_dataset returns resp['data'] when successful."""
    patch_network_get(monkeypatch, response={"data": {"p": "ok"}})

    result = manager.get_prometheus_dataset("999")
    assert result == {"p": "ok"}


def test_get_prometheus_invalid_uuid(manager, monkeypatch):
    """Invalid UUID in get_prometheus_dataset raises CogflowArtifactError."""

    def bad_uuid(_):
        raise ValueError("bad uuid")

    monkeypatch.setattr(common, "normalize_uuid", bad_uuid, raising=True)

    with pytest.raises(CogflowArtifactError):
        manager.get_prometheus_dataset("BAD")


def test_get_prometheus_network_failure(manager, monkeypatch):
    """Network errors in get_prometheus_dataset raise CogflowConnectionError."""
    patch_network_get(monkeypatch, side_effect=Exception("network fail"))

    with pytest.raises(CogflowConnectionError):
        manager.get_prometheus_dataset("101")


def test_get_prometheus_missing_data_field(manager, monkeypatch):
    """Missing 'data' field raises CogflowArtifactError."""
    patch_network_get(monkeypatch, response={"foo": "bar"})

    with pytest.raises(CogflowArtifactError):
        manager.get_prometheus_dataset("202")


def test_get_prometheus_empty_data(manager, monkeypatch):
    """Empty 'data' triggers CogflowDatasetError."""
    patch_network_get(monkeypatch, response={"data": None})

    with pytest.raises(CogflowDatasetError):
        manager.get_prometheus_dataset("303")
