"""
Dataset management service for Cogflow SDK."""

from typing import Union
from uuid import UUID

from ..utils.exceptions import (
    CogflowErrorHandler,
    CogflowConnectionError,
    CogflowArtifactError,
    CogflowValidationError,
    CogflowDatasetError,
)
from ..utils.logging import get_logger
from ..utils import network, common
from ..config import config

logger = get_logger(__name__)


class DatasetManager:
    """
    Service class for dataset operations.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize the DatasetManager.
        Args:
            strict (bool): If True, enforce strict validation on dataset operations.
        """
        self.base_url = f"{config.API_PATH}{config.DATASETS}"

    def get_dataset(self, dataset_id: Union[str, UUID]):
        """
        Retrieve a dataset by its ID (UUID string path param).

        API:
            GET /{id}

        Args:
            dataset_id (str | UUID): Dataset identifier.

        Returns:
            dict: Dataset details (resp["data"]) or raises Cogflow*Error.
        Example:
        >>> from cogflow import datasets
        >>> dataset = datasets.get_dataset("123e4567-e89b-12d3-a456-426614174000")
        >>> print(dataset)
        """

        # ----------------------------------------------------------
        # Validate & normalize UUID
        # ----------------------------------------------------------
        try:
            dataset_id = common.normalize_uuid(dataset_id)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Invalid dataset_id '{dataset_id}'",
                raise_as=CogflowValidationError,
                re_raise=True,
            )

        # ----------------------------------------------------------
        # Prepare request
        # ----------------------------------------------------------
        headers = {"kubeflow-userid": common.get_current_user()}

        # ----------------------------------------------------------
        # Call API
        # ----------------------------------------------------------
        resp = None
        try:
            logger.info("Fetching dataset metadata for ID=%s", dataset_id)

            resp = network.make_get_request(
                url=self.base_url,
                path_params=dataset_id,
                headers=headers,
            )

        except Exception as e:
            # Network or request failure (timeout, connection, DNS, etc.)
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Failed to fetch dataset '{dataset_id}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

        # ----------------------------------------------------------
        # Validate response
        # ----------------------------------------------------------
        if not isinstance(resp, dict):
            raise CogflowArtifactError(
                f"Unexpected API response format for dataset '{dataset_id}': {type(resp)}"
            )

        if "data" not in resp:
            raise CogflowArtifactError(
                f"Dataset API returned no 'data' field for dataset '{dataset_id}'"
            )

        data = resp.get("data")

        if not data:
            CogflowErrorHandler.log_and_raise(
                f"Dataset '{dataset_id}' not found or returned empty data",
                raise_as=CogflowDatasetError,
            )

        logger.info("Successfully retrieved dataset '%s'", dataset_id)
        return data

    def get_prometheus_dataset(self, dataset_id: Union[str, UUID]):
        """
        Retrieve a Prometheus dataset by its ID.

        API:
            GET /{id}/prometheus

        Args:
            dataset_id (str | UUID): Dataset identifier.

        Returns:
            dict: Prometheus dataset details.

        Raises:
            CogflowConnectionError: Network failure / request failure.
            CogflowArtifactError: Invalid or missing API response.
        Example:
        >>> from cogflow import datasets
        >>> prometheus_data = datasets.get_prometheus_dataset("123e4567-e89b-12d3-a456-426614174000")
        >>> print(prometheus_data)
        """

        # ----------------------------------------------------------
        # Normalize UUID
        # ----------------------------------------------------------
        try:
            dataset_id = common.normalize_uuid(dataset_id)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Invalid dataset_id '{dataset_id}'",
                raise_as=CogflowArtifactError,
                re_raise=True,
            )

        # ----------------------------------------------------------
        # Build URL
        # ----------------------------------------------------------
        url = f"{self.base_url}/{dataset_id}/prometheus"

        headers = {"kubeflow-userid": common.get_current_user()}

        resp = None  # ensure always defined

        # ----------------------------------------------------------
        # Make API call
        # ----------------------------------------------------------
        try:
            logger.info("Fetching Prometheus dataset for ID=%s", dataset_id)

            resp = network.make_get_request(
                url=url,
                headers=headers,
            )

        except Exception as e:
            # Network or request failure
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Failed to fetch Prometheus dataset '{dataset_id}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

        # ----------------------------------------------------------
        # Validate API Response
        # ----------------------------------------------------------
        if not isinstance(resp, dict):
            raise CogflowArtifactError(
                f"Unexpected API response format for Prometheus dataset '{dataset_id}'"
            )

        if "data" not in resp:
            raise CogflowArtifactError(
                f"Prometheus dataset API returned no 'data' field for dataset '{dataset_id}'"
            )

        data = resp.get("data")

        if not data:
            CogflowErrorHandler.log_and_raise(
                f"Prometheus dataset '{dataset_id}' returned empty 'data'",
                raise_as=CogflowDatasetError,
            )
        logger.info("Successfully retrieved Prometheus dataset '%s'", dataset_id)
        return data


# Create a singleton instance for the public interface
_datasets = DatasetManager()

# Exposed SDK-level methods (single source of truth)
for attr_name in dir(DatasetManager):
    # Skip private methods and dunder methods
    if attr_name.startswith("_"):
        continue

    attr = getattr(DatasetManager, attr_name)

    # Only export methods (callables) that belong to ModelManager
    if callable(attr):
        # Bind the method to the singleton instance
        globals()[attr_name] = getattr(_datasets, attr_name)
