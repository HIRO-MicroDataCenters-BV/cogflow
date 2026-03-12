"""
    Test module for cases related to notebook_plugin
"""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from ..cogflow.plugins.notebook_plugin import NotebookPlugin


class TestNotebookPlugin(unittest.TestCase):
    """Tests for NotebookPlugin class and related functions."""

    @patch("os.getenv")
    def test_delete_pipeline_details_from_db(self, mock_env):
        """Test deleting pipeline details from DB."""
        with patch("httpx.delete") as mock_requests_delete:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://randomn",
                "TIMER_IN_SEC": "10",
                "FILE_TYPE": "2",
                "MLFLOW_TRACKING_URI": "http://mlflow",
                "ML_TOOL": "ml_flow",
            }[x]

            mock_response = {
                "errors": "None",
                "message": "Pipeline Details Deleted successfully",
                "success": "True",
            }
            mock_requests_delete.return_value.status_code = 200
            mock_requests_delete.return_value.json.return_value = mock_response
            f = StringIO()
            with redirect_stdout(f):
                NotebookPlugin().delete_pipeline_details_from_db("2")
            out = f.getvalue().strip()
            assert out == "DELETE request successful"

    @patch("os.getenv")
    def test_delete_run_details_from_db(self, mock_env):
        """Test deleting run details from DB."""
        with patch("httpx.delete") as mock_requests_delete:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://randomn",
                "TIMER_IN_SEC": "10",
                "FILE_TYPE": "2",
                "MLFLOW_TRACKING_URI": "http://mlflow",
                "ML_TOOL": "ml_flow",
            }[x]

            mock_response = {
                "errors": "None",
                "message": "Runs deleted successfully",
                "success": "True",
            }
            mock_requests_delete.return_value.status_code = 200
            mock_requests_delete.return_value.json.return_value = mock_response
            f = StringIO()
            with redirect_stdout(f):
                NotebookPlugin().delete_run_details_from_db("2")
            out = f.getvalue().strip()
            assert out == "DELETE request successful"

    @patch("os.getenv")
    def test_list_runs_by_pipeline_id(self, mock_env):
        """Test listing runs by pipeline ID."""
        with patch("httpx.get") as mock_requests_get:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://randomn",
                "TIMER_IN_SEC": "10",
                "FILE_TYPE": "2",
                "MLFLOW_TRACKING_URI": "http://mlflow",
                "ML_TOOL": "ml_flow",
            }[x]

            # Define mock response
            mock_response = {
                "data": [
                    {"run_id": "0d3ffa58-7d15-4456-a1f6-2c1355f95d22"},
                    {"run_id": "0d3ffa58-7d15-4456-a1f6-2c1355f95d23"},
                ]
            }

            mock_requests_get.return_value.status_code = 200
            mock_requests_get.return_value.json.return_value = mock_response
            result = NotebookPlugin().list_runs_by_pipeline_id("2")

            assert result == mock_response["data"]

    def test_deploy_model_for_model_not_found_exception(self):
        """Test deploy model when model is not found."""
        with patch(
            "cogflow.cogflow.plugins.mlflowplugin.MlflowPlugin.get_model_uri"
        ) as mock_model_uri:
            mock_model_uri.side_effect = Exception()
            with self.assertRaises(Exception):
                NotebookPlugin().deploy_model("Flearning", "1", "fl-svc")


if __name__ == "__main__":
    unittest.main()
