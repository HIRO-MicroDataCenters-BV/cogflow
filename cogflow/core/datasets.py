"""
Dataset management service for Cogflow SDK."""

from uuid import UUID

from ..config import config
from ..utils import common, network
from ..utils.exceptions import (
    CogflowArtifactError,
    CogflowConnectionError,
    CogflowDatasetError,
    CogflowErrorHandler,
    CogflowValidationError,
)
from ..utils.logging import get_logger

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

    def get_dataset(self, dataset_id: str | UUID):
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
            raise CogflowArtifactError(f"Unexpected API response format for dataset '{dataset_id}': {type(resp)}")

        if "data" not in resp:
            raise CogflowArtifactError(f"Dataset API returned no 'data' field for dataset '{dataset_id}'")

        data = resp.get("data")

        if not data:
            CogflowErrorHandler.log_and_raise(
                f"Dataset '{dataset_id}' not found or returned empty data",
                raise_as=CogflowDatasetError,
            )

        logger.info("Successfully retrieved dataset '%s'", dataset_id)
        return data

    def get_prometheus_dataset(self, dataset_id: str | UUID):
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
            raise CogflowArtifactError(f"Unexpected API response format for Prometheus dataset '{dataset_id}'")

        if "data" not in resp:
            raise CogflowArtifactError(f"Prometheus dataset API returned no 'data' field for dataset '{dataset_id}'")

        data = resp.get("data")

        if not data:
            CogflowErrorHandler.log_and_raise(
                f"Prometheus dataset '{dataset_id}' returned empty 'data'",
                raise_as=CogflowDatasetError,
            )
        logger.info("Successfully retrieved Prometheus dataset '%s'", dataset_id)
        return data

    def register_dataset(
        self,
        dataset_type: int,
        name: str,
        file_path: str,
        description: str | None = None,
    ):
        """
        Register a dataset by uploading a file.

        API:
            POST /file

        Args:
            dataset_type (int): 0 (train), 1 (inference), 2 (both)
            name (str): Dataset name
            file_path (str): Path to dataset file
            description (str, optional): Dataset description

        Returns:
            dict: Registered dataset metadata (resp["data"])

        Raises:
            CogflowValidationError
            CogflowConnectionError
            CogflowDatasetError
        """

        # ----------------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------------
        if dataset_type not in (0, 1, 2):
            CogflowErrorHandler.log_and_raise(
                f"Invalid dataset_type '{dataset_type}'. Must be 0, 1, or 2.",
                raise_as=CogflowValidationError,
            )

        if not name:
            CogflowErrorHandler.log_and_raise(
                "Dataset name must not be empty",
                raise_as=CogflowValidationError,
            )

        if not common.file_exists(file_path):
            CogflowErrorHandler.log_and_raise(
                f"Dataset file not found: {file_path}",
                raise_as=CogflowValidationError,
            )

        # ----------------------------------------------------------
        # Prepare request
        # ----------------------------------------------------------
        url = f"{self.base_url}/file"

        headers = {
            "kubeflow-userid": common.get_current_user(),
        }

        data = {
            "dataset_type": str(dataset_type),
            "name": name,
            "description": description or "",
        }

        resp = None

        # ----------------------------------------------------------
        # Make API call (multipart/form-data)
        # ----------------------------------------------------------
        try:
            logger.info("Registering dataset '%s' from file '%s'", name, file_path)

            with open(file_path, "rb") as f:
                files = {"files": (common.get_filename(file_path), f)}

                resp = network.make_post_request(
                    url=url,
                    data=data,
                    files=files,
                    headers=headers,
                )

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Failed to register dataset '{name}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

        # ----------------------------------------------------------
        # Validate response
        # ----------------------------------------------------------
        if not isinstance(resp, dict):
            raise CogflowDatasetError(f"Unexpected API response while registering dataset '{name}': {type(resp)}")

        if "data" not in resp:
            raise CogflowDatasetError(f"Dataset registration API returned no 'data' for dataset '{name}'")

        data = resp.get("data")

        if not data:
            CogflowErrorHandler.log_and_raise(
                f"Dataset '{name}' registration returned empty data",
                raise_as=CogflowDatasetError,
            )

        logger.info("Successfully registered dataset '%s'", name)
        return data

    def delete_dataset(self, dataset_id: str | UUID) -> bool:
        """
        Silently delete dataset file.

        Returns True if DELETE request succeeded.
        """
        try:
            dataset_id = common.normalize_uuid(dataset_id)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Invalid dataset_id '{dataset_id}'",
                raise_as=CogflowValidationError,
                re_raise=True,
            )

        url = f"{self.base_url}/{dataset_id}/file"
        headers = {"kubeflow-userid": common.get_current_user()}

        try:
            network.make_delete_request(url=url, headers=headers)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Failed to delete dataset file '{dataset_id}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

        logger.info("Dataset file deleted successfully (dataset_id=%s)", dataset_id)
        return True

    async def async_get_dataset(self, dataset_id: str | UUID):
        """Async mirror of :meth:`get_dataset`.

        Same UUID validation, request shape, and response handling as
        the sync variant. Uses :func:`network.make_async_get_request`
        so a caller already inside an ``async def`` doesn't block the
        event loop on the round-trip.
        """
        try:
            dataset_id = common.normalize_uuid(dataset_id)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Invalid dataset_id '{dataset_id}'",
                raise_as=CogflowValidationError,
                re_raise=True,
            )

        headers = {"kubeflow-userid": common.get_current_user()}

        resp = None
        try:
            logger.info("Fetching dataset metadata for ID=%s", dataset_id)

            resp = await network.make_async_get_request(
                url=self.base_url,
                path_params=dataset_id,
                headers=headers,
            )

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Failed to fetch dataset '{dataset_id}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

        if not isinstance(resp, dict):
            raise CogflowArtifactError(f"Unexpected API response format for dataset '{dataset_id}': {type(resp)}")

        if "data" not in resp:
            raise CogflowArtifactError(f"Dataset API returned no 'data' field for dataset '{dataset_id}'")

        data = resp.get("data")

        if not data:
            CogflowErrorHandler.log_and_raise(
                f"Dataset '{dataset_id}' not found or returned empty data",
                raise_as=CogflowDatasetError,
            )

        logger.info("Successfully retrieved dataset '%s'", dataset_id)
        return data

    async def async_get_prometheus_dataset(self, dataset_id: str | UUID):
        """Async mirror of :meth:`get_prometheus_dataset`."""
        try:
            dataset_id = common.normalize_uuid(dataset_id)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Invalid dataset_id '{dataset_id}'",
                raise_as=CogflowArtifactError,
                re_raise=True,
            )

        url = f"{self.base_url}/{dataset_id}/prometheus"
        headers = {"kubeflow-userid": common.get_current_user()}

        resp = None
        try:
            logger.info("Fetching Prometheus dataset for ID=%s", dataset_id)

            resp = await network.make_async_get_request(
                url=url,
                headers=headers,
            )

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Failed to fetch Prometheus dataset '{dataset_id}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

        if not isinstance(resp, dict):
            raise CogflowArtifactError(f"Unexpected API response format for Prometheus dataset '{dataset_id}'")

        if "data" not in resp:
            raise CogflowArtifactError(f"Prometheus dataset API returned no 'data' field for dataset '{dataset_id}'")

        data = resp.get("data")

        if not data:
            CogflowErrorHandler.log_and_raise(
                f"Prometheus dataset '{dataset_id}' returned empty 'data'",
                raise_as=CogflowDatasetError,
            )
        logger.info("Successfully retrieved Prometheus dataset '%s'", dataset_id)
        return data

    async def async_delete_dataset(self, dataset_id: str | UUID) -> bool:
        """Async mirror of :meth:`delete_dataset`."""
        try:
            dataset_id = common.normalize_uuid(dataset_id)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Invalid dataset_id '{dataset_id}'",
                raise_as=CogflowValidationError,
                re_raise=True,
            )

        url = f"{self.base_url}/{dataset_id}/file"
        headers = {"kubeflow-userid": common.get_current_user()}

        try:
            await network.make_async_delete_request(url=url, headers=headers)
        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Failed to delete dataset file '{dataset_id}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

        logger.info("Dataset file deleted successfully (dataset_id=%s)", dataset_id)
        return True

    def download_dataset(
        self,
        dataset_id: str | UUID,
        output_path: str | None = None,
    ) -> str:
        """
        Download a dataset file via CogFlow API.

        API:
            GET /{id}/file/download

        Args:
            dataset_id (str | UUID): Dataset identifier.
            output_path (str, optional):
                - If None → save to CWD using server filename
                - If directory → save inside directory
                - If file path → save exactly there

        Returns:
            str: Absolute path to downloaded file

        Raises:
            CogflowValidationError
            CogflowConnectionError
            CogflowDatasetError
        """

        # ----------------------------------------------------------
        # Normalize dataset ID
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
        url = f"{self.base_url}/{dataset_id}/file/download"
        headers = {"kubeflow-userid": common.get_current_user()}

        try:
            logger.info("Downloading dataset file (id=%s)", dataset_id)

            response = network.make_get_request_stream(
                url=url,
                headers=headers,
            )

            # ------------------------------------------------------
            # Extract filename from Content-Disposition
            # ------------------------------------------------------
            filename = None
            cd = response.headers.get("Content-Disposition")

            if cd and "filename=" in cd:
                filename = cd.split("filename=")[-1].strip().strip('"')

            if not filename:
                filename = f"dataset-{dataset_id}.bin"

            # ------------------------------------------------------
            # Resolve output path
            # ------------------------------------------------------
            if output_path is None:
                output_path = common.join_path(common.cwd(), filename)
            elif common.is_dir(output_path):
                output_path = common.join_path(output_path, filename)

            # ------------------------------------------------------
            # Write binary stream
            # ------------------------------------------------------
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        except Exception as e:
            CogflowErrorHandler.handle_exception(
                e,
                context=f"Failed to download dataset '{dataset_id}'",
                raise_as=CogflowConnectionError,
                re_raise=True,
            )

        # ----------------------------------------------------------
        # Validate result
        # ----------------------------------------------------------
        if not common.file_exists(output_path):
            CogflowErrorHandler.log_and_raise(
                f"Dataset download failed for '{dataset_id}'",
                raise_as=CogflowDatasetError,
            )

        logger.info("Dataset downloaded successfully → %s", output_path)
        return output_path


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
