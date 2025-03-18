"""
This module contains unit tests for the functions of the DatasetPlugin and related functionality.
"""

import unittest
from unittest.mock import patch, MagicMock

import minio
import requests

from ..cogflow.plugins.dataset_plugin import DatasetPlugin, DatasetMetadata
from ..cogflow import (
    query_endpoint_and_download_file,
    save_to_minio,
    delete_from_minio,
    register_dataset,
    log_model_with_dataset,
)


class TestDatasetPlugin(unittest.TestCase):
    """
    This class contains test cases for the DatasetPlugin functionalities like
    querying endpoints, saving to and deleting from Minio, and dataset registration.
    """

    @patch("requests.get")
    @patch("cogflow.cogflow.plugins.dataset_plugin.DatasetPlugin.save_to_minio")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_query_endpoint_and_download_file_success(
        self, mock_plugin_activation, mock_save_to_minio, mock_requests_get
    ):
        """
        Test successful file download from the dataset endpoint.
        Verifies if file content is saved to Minio and plugin activation is checked.
        """
        # Arrange
        url = "http://dataset.com/dataset"
        output_file = "dataset.csv"
        bucket_name = "mlpipeline"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"Test dataset content"
        mock_requests_get.return_value = mock_response

        # mock_minio_instance = MagicMock()
        mock_save_to_minio.return_value = "http://dataset.com/dataset.csv"

        # Act
        success = query_endpoint_and_download_file(
            url=url, output_file=output_file, bucket_name=bucket_name
        )

        # Assert
        self.assertTrue(success)
        mock_save_to_minio.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("requests.get")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_query_endpoint_and_download_file_failure(
        self, mock_plugin_activation, mock_requests_get
    ):
        """
        test for query_endpoint_and_download_file
        """
        # Arrange
        dataset_plugin = DatasetPlugin()
        url = "http://dataset.com/dataset"
        bucket_name = "mlpipeline"
        output_file = "dataset.csv"
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests_get.return_value = mock_response

        # Act & Assert
        with self.assertRaises(Exception):
            dataset_plugin.query_endpoint_and_download_file(
                url, output_file, bucket_name
            )
        mock_plugin_activation.assert_called_once()

    @patch("requests.get")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_request_exception_query_endpoint(
        self, mock_plugin_activation, mock_requests_get
    ):
        """
        test for verify_activation
        """
        # Arrange
        dataset_plugin = DatasetPlugin()
        url = "http://dataset.com/dataset"
        bucket_name = "mlpipeline"
        output_file = "dataset.csv"
        # mock_response = MagicMock()
        # mock_response.status_code = 404
        # mock_requests_get.return_value = mock_response
        mock_requests_get.side_effect = requests.exceptions.RequestException(
            "Request failed"
        )
        with self.assertRaises(Exception):
            dataset_plugin.query_endpoint_and_download_file(
                url, output_file, bucket_name
            )
        mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.plugins.dataset_plugin.DatasetPlugin.create_minio_client")
    @patch("os.getenv")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_save_to_minio_success(
        self, mock_plugin_activation, mock_getenv, mock_create_minio_client
    ):
        """
        test for create_minio_client
        """

        # Arrange
        mock_minio_client = MagicMock()
        mock_create_minio_client.return_value = mock_minio_client
        mock_getenv.side_effect = lambda x: {
            "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
        }[x]

        file_content = b"Test dataset content"
        output_file = "dataset.csv"

        mock_minio_instance = MagicMock()
        mock_minio_instance.bucket_exists.return_value = True
        mock_minio_instance.presigned_get_object.return_value = (
            "http://dataset.com/dataset.csv"
        )

        # Act
        save_to_minio(file_content, output_file, bucket_name="mlpipeline")
        # Assertions
        mock_minio_client.bucket_exists.assert_called_once_with("mlpipeline")
        mock_minio_client.presigned_get_object.assert_called_once()
        mock_minio_client.put_object.assert_called_once()
        mock_create_minio_client.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.plugins.dataset_plugin.DatasetPlugin.create_minio_client")
    @patch("os.getenv")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_minio_bucket_creation(
        self,
        mock_plugin_activation,
        mock_getenv,
        mock_create_minio_client,
    ):
        """
        test for create_minio_client
        """
        # Arrange
        mock_minio_client = MagicMock()
        mock_create_minio_client.return_value = mock_minio_client
        mock_getenv.side_effect = lambda x: {
            "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
        }[x]

        dataset_plugin = DatasetPlugin()
        file_content = b"Test dataset content"
        output_file = "dataset.csv"

        mock_minio_client.bucket_exists.return_value = False
        dataset_plugin.save_to_minio(
            file_content, output_file, bucket_name="bucket_name"
        )
        mock_minio_client.make_bucket.assert_called_once()
        mock_minio_client.make_bucket.assert_called_once_with("bucket_name")
        mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.plugins.dataset_plugin.DatasetPlugin.create_minio_client")
    @patch("os.getenv")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_minio_bucket_creation_exception(
        self,
        mock_plugin_activation,
        mock_getenv,
        mock_create_minio_client,
    ):
        """
        test for create_minio_client
        """
        # Arrange
        mock_minio_client = MagicMock()
        mock_create_minio_client.return_value = mock_minio_client
        mock_getenv.side_effect = lambda x: {
            "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
        }[x]

        dataset_plugin = DatasetPlugin()
        file_content = b"Test dataset content"
        output_file = "dataset.csv"

        mock_minio_client.bucket_exists.return_value = False
        mock_minio_client.make_bucket.side_effect = Exception(
            "Bucket Cannot be created"
        )
        with self.assertRaises(Exception):
            dataset_plugin.save_to_minio(
                file_content, output_file, bucket_name="bucket_name"
            )
        mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.plugins.dataset_plugin.DatasetPlugin.create_minio_client")
    @patch("os.getenv")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_save_to_exception(
        self, mock_plugin_activation, mock_getenv, mock_create_minio_client
    ):
        """
        test for create_minio_client
        """
        # Arrange
        mock_minio_client = MagicMock()
        mock_create_minio_client.return_value = mock_minio_client
        mock_getenv.side_effect = lambda x: {
            "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
        }[x]

        dataset_plugin = DatasetPlugin()
        file_content = b"Test dataset content"
        output_file = "dataset.csv"

        # mock_minio_client.bucket_exists.return_value = False
        # mock_minio_client.make_bucket.side_effect = Exception("Bucket Cannot be created")
        mock_minio_client.put_object.side_effect = Exception(
            "Error while storing object"
        )
        with self.assertRaises(Exception):
            dataset_plugin.save_to_minio(
                file_content, output_file, bucket_name="bucket_name"
            )
        mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.plugins.dataset_plugin.DatasetPlugin.create_minio_client")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_delete_from_minio_success(
        self, mock_plugin_activation, mock_create_minio_client
    ):
        """
        test for delete_from_minio
        """
        # Arrange
        object_name = "test_object"
        bucket_name = "test_bucket"

        mock_minio_instance = MagicMock()
        mock_create_minio_client.return_value = mock_minio_instance
        mock_minio_instance.stat_object.return_value = True  # Object exists

        # Act
        result = delete_from_minio(object_name, bucket_name)

        # Assert
        self.assertTrue(result)
        mock_minio_instance.remove_object.assert_called_once_with(
            bucket_name, object_name
        )
        mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.plugins.dataset_plugin.DatasetPlugin.create_minio_client")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_delete_from_minio_object_not_found(
        self, mock_plugin_activation, mock_create_minio_client
    ):
        """
        test for delete_from_minio
        """
        # Arrange
        dataset_plugin = DatasetPlugin()
        object_name = "test_object"
        bucket_name = "test_bucket"

        mock_minio_instance = MagicMock()
        mock_create_minio_client.return_value = mock_minio_instance
        mock_minio_instance.stat_object.return_value = False  # Object does not exist

        # Act
        result = dataset_plugin.delete_from_minio(object_name, bucket_name)

        # Assert
        self.assertFalse(result)
        mock_minio_instance.remove_object.assert_not_called()
        mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.plugins.dataset_plugin.DatasetPlugin.create_minio_client")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_delete_from_minio_exception(
        self, mock_plugin_activation, mock_create_minio_client
    ):
        """
        test for delete_from_minio
        """
        # Arrange
        dataset_plugin = DatasetPlugin()
        object_name = "test_object"
        bucket_name = "test_bucket"

        mock_minio_instance = MagicMock()
        mock_create_minio_client.return_value = mock_minio_instance
        mock_minio_instance.stat_object.side_effect = Exception("Test error")

        # Act
        result = dataset_plugin.delete_from_minio(object_name, bucket_name)

        # Assert
        self.assertFalse(result)
        mock_minio_instance.remove_object.assert_not_called()
        mock_plugin_activation.assert_called_once()

    @patch("os.getenv")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_create_minio_client(self, mock_plugin_activation, mock_getenv):
        """
        test for create_minio_client
        """
        # Define test parameters
        mock_getenv.side_effect = lambda x: {
            "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
        }[x]

        # Create the MinioClient object
        dataset_plugin = DatasetPlugin()
        minio_client = dataset_plugin.create_minio_client()

        # Assert that the MinioClient object is created correctly
        self.assertIsInstance(minio_client, minio.api.Minio)
        mock_plugin_activation.assert_called_once()

    def test_log_model_with_dataset(self):
        """
        test for log_model_with_dataset
        """
        with patch("mlflow.sklearn.log_model") as mock_log_model:
            with patch(
                "cogflow.cogflow.plugins.notebook_plugin.NotebookPlugin.get_model_latest_version"
            ) as mock_model_version:
                with patch("mlflow.active_run") as mock_active_run:
                    with patch("os.getenv") as mock_env:
                        with patch("requests.post") as mock_requests_post:
                            # Create a mock run object
                            mock_run = MagicMock()
                            mock_run.info.run_id = "12345"

                            # Set the return value of mlflow.active_run()
                            mock_active_run.return_value = mock_run
                            mock_env.side_effect = lambda x: {
                                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                                "AWS_ACCESS_KEY_ID": "minio",
                                "AWS_SECRET_ACCESS_KEY": "minio123",
                                "API_BASEPATH": "http://randomn",
                                "TIMER_IN_SEC": "10",
                                "MLFLOW_TRACKING_URI": "http://mlflow",
                                "ML_TOOL": "ml_flow",
                                "FILE_TYPE": "0",
                            }[x]

                            mock_response = {
                                "data": {"dataset_id": 5, "id": 1},
                                "message": "Dataset linked with model successfully",
                            }
                            mock_requests_post.return_value.status_code = 201
                            mock_requests_post.return_value.json.return_value = (
                                mock_response
                            )

                            # Mock model
                            sk_model = MagicMock()
                            artifact_path = "model"
                            registered_model_name = "testmodel"

                            # Dataset details
                            source = (
                                "https://archive.ics.uci.edu/static/public/17"
                                "/breast+cancer+wisconsin+diagnostic.zip"
                            )
                            file_format = "zip"
                            name = "breast+cancer+wisconsin+diagnostic.zip"
                            description = "Breast cancer wisconsin diagnotic dataset"

                            dm = DatasetMetadata(name, description, source, file_format)
                            mock_model_version.return_value = 1
                            # Define any other necessary inputs for the log_model method

                            # Call the method under test
                            log_model_with_dataset(
                                sk_model=sk_model,
                                artifact_path=artifact_path,
                                registered_model_name=registered_model_name,
                                dataset=dm,
                            )
                            mock_log_model.assert_called_once()

    @patch("requests.post")
    @patch("os.getenv")
    def test_register_dataset(self, mock_env, mock_requests_post):
        """
        test for register_dataset
        """
        mock_env.side_effect = lambda x: {
            "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
            "API_BASEPATH": "http://randomn",
            "TIMER_IN_SEC": "10",
            "MLFLOW_TRACKING_URI": "http://mlflow_server",
            "ML_TOOL": "mlflow",
            "FILE_TYPE": "0",
        }[x]

        mock_response = {
            "data": {"dataset_id": 5, "id": 1},
            "message": "Dataset uploaded successfully",
        }
        mock_requests_post.return_value.status_code = 201
        mock_requests_post.return_value.json.return_value = mock_response
        # Dataset details
        source = (
            "https://archive.ics.uci.edu/static/public/17"
            "/breast+cancer+wisconsin+diagnostic.zip"
        )
        file_format = "zip"
        name = "breast+cancer+wisconsin+diagnostic.zip"
        description = "Breast cancer wisconsin diagnotic dataset"

        dm = DatasetMetadata(name, description, source, file_format)
        result = register_dataset(dm)
        self.assertEqual(result, mock_response)


if __name__ == "__main__":
    unittest.main()
