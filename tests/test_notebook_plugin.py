"""
    Test module for cases related to notebook_plugin
"""

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch, MagicMock
from ..cogflow import (
    link_model_to_dataset,
    save_dataset_details,
    save_model_details_to_db,
)
from ..cogflow.plugins.dataset_plugin import DatasetMetadata
from ..cogflow.plugins.notebook_plugin import NotebookPlugin


class TestNotebookPlugin(unittest.TestCase):
    """Tests for NotebookPlugin class and related functions."""

    @patch(
        "cogflow.cogflow.plugins.notebook_plugin.NotebookPlugin.get_model_latest_version"
    )
    @patch("os.getenv")
    def test_save_model_details_to_db(self, mock_env, mock_model_version):
        """Test save_model_details_to_db function."""
        with patch("requests.post") as mock_requests_post:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://modelregister",
            }[x]
            mock_model_version.return_value = 1

            mock_response = {
                "data": {
                    "id": 101,
                    "last_modified_time": "2024-05-16T12:33:08.890033",
                    "last_modified_user_id": 0,
                    "name": "testmodel",
                    "register_date": "2024-05-16T12:33:08.890007",
                    "register_user_id": 0,
                    "type": "sklearn",
                    "version": "1",
                },
                "errors": "None",
                "message": "Created new model.",
                "success": "True",
            }
            mock_requests_post.return_value.status_code = 201
            mock_requests_post.return_value.json.return_value = mock_response
            result = save_model_details_to_db("testmodel")
            self.assertEqual(result["data"]["id"], 101)

    @patch("os.getenv")
    def test_save_dataset_details(self, mock_env):
        """Test save_dataset_details function."""
        with patch("requests.post") as mock_requests_post:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://randomn_",
            }[x]

            mock_response = {
                "data": {
                    "dataset_id": 8,
                    "file_name": "breastcancerwisconsindiagnostic.zip",
                    "file_path": "mlflow",
                    "register_date": "2024-05-16T13:03:05.442386",
                    "user_id": 0,
                },
                "errors": "None",
                "message": "File uploaded successfully.",
                "success": "True",
            }
            mock_requests_post.return_value.status_code = 201
            mock_requests_post.return_value.json.return_value = mock_response
            # Dataset details
            source = (
                "https://archive.ics.uci.edu/static/public/17"
                "/breast+cancer+wisconsin+diagnostic1.zip"
            )
            file_format = "zip"
            name = "breast+cancer+wisconsin+diagnostic.zip"
            description = "Breast cancer wisconsin diagnotic dataset"

            dm = DatasetMetadata(name, description, source, file_format)
            result = save_dataset_details(dataset=dm)
            self.assertEqual(result, mock_response["data"]["dataset_id"])

    @patch("os.getenv")
    def test_link_model_to_dataset(self, mock_env):
        """Test linking a model to a dataset."""
        with patch("requests.post") as mock_requests_post:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://model",
            }[x]

            mock_response = {
                "data": {
                    "dataset_id": 2,
                    "linked_time": "2024-05-16 15:23:24",
                    "model_id": 1,
                    "user_id": 0,
                },
                "errors": "None",
                "message": "Dataset linked with model successfully",
                "success": "True",
            }
            mock_requests_post.return_value.status_code = 201
            mock_requests_post.return_value.json.return_value = mock_response
            f = StringIO()
            with redirect_stdout(f):
                link_model_to_dataset(2, 1)
            out = f.getvalue().strip()
            assert out == "POST request successful"

    @patch("os.getenv")
    def test_delete_pipeline_details_from_db(self, mock_env):
        """Test deleting pipeline details from DB."""
        with patch("requests.delete") as mock_requests_delete:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://randomn1",
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
        with patch("requests.delete") as mock_requests_delete:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://randomn_path",
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
        with patch("requests.get") as mock_requests_get:
            mock_env.side_effect = lambda x: {
                "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "minio123",
                "API_BASEPATH": "http://randomnpath",
            }[x]

            # Define mock response
            mock_response = {
                "runs": [
                    {"run_id": "0d3ffa58-7d15-4456-a1f6-2c1355f95d22"},
                    {"run_id": "0d3ffa58-7d15-4456-a1f6-2c1355f95d23"},
                ]
            }

            mock_requests_get.return_value.status_code = 200
            mock_requests_get.return_value.json.return_value = mock_response
            f = StringIO()
            with redirect_stdout(f):
                NotebookPlugin().list_runs_by_pipeline_id("2")
            out = f.getvalue().strip()
            assert out == "GET request successful"

    @patch("cogflow.cogflow.plugins.notebook_plugin.NotebookPlugin.load_pkl")
    @patch(
        "cogflow.cogflow.plugins.notebook_plugin.NotebookPlugin.get_model_latest_version"
    )
    @patch("mlflow.sklearn.log_model")
    def test_log_model_by_model_file(
        self, mock_log_model, mock_model_version, mock_load_pkl
    ):
        """Test logging a model by its file."""
        sk_model = MagicMock()
        mock_load_pkl.return_value = sk_model
        mock_model_version.return_value = 1
        model_file_path = "var/data/model.pkl"
        model_name = "test_model"
        result = NotebookPlugin().log_model_by_model_file(model_file_path, model_name)
        self.assertEqual(result["version"], 1)
        self.assertIn("artifact_uri", result)
        mock_log_model.assert_called_once()

    def test_deploy_model(self):
        """Test model deployment."""
        with patch("cogflow.cogflow.plugins.mlflowplugin.MlflowPlugin.get_model_uri"):
            with patch(
                "cogflow.cogflow.plugins.kubeflowplugin.KubeflowPlugin.serve_model_v1"
            ):
                result = NotebookPlugin().deploy_model("Flearning", "1", "fl-svc")
                self.assertEqual(result["status"], True)
                self.assertEqual(
                    result["msg"], "Model Flearning deployed with service fl-svc"
                )

    def test_deploy_model_for_model_not_found_exception(self):
        """Test deploy model when model is not found."""
        with patch(
            "cogflow.cogflow.plugins.mlflowplugin.MlflowPlugin.get_model_uri"
        ) as mock_model_uri:
            mock_model_uri.side_effect = Exception()
            with self.assertRaises(Exception):
                NotebookPlugin().deploy_model("Flearning", "1", "fl-svc")

    @patch.object(NotebookPlugin, "run_kubectl_command")
    def test_get_pods_in_namespace(self, mock_run_kubectl_command):
        """Test getting pods in a namespace."""
        mock_output = {
            "items": [{"metadata": {"name": "pod1"}}, {"metadata": {"name": "pod2"}}]
        }
        mock_run_kubectl_command.return_value = json.dumps(mock_output)

        pods = NotebookPlugin.get_pods_in_namespace("test-namespace")

        self.assertEqual(len(pods), 2)
        self.assertEqual(pods[0]["metadata"]["name"], "pod1")
        self.assertEqual(pods[1]["metadata"]["name"], "pod2")

    @patch.object(NotebookPlugin, "run_kubectl_command")
    def test_get_pod_logs(self, mock_run_kubectl_command):
        """Test getting pod logs."""
        mock_logs = "Mock pod logs"
        mock_run_kubectl_command.return_value = mock_logs

        logs = NotebookPlugin.get_pod_logs("test-namespace", "test-pod")

        self.assertEqual(logs, mock_logs)

    @patch.object(NotebookPlugin, "run_kubectl_command")
    def test_get_pod_prefix(self, mock_run_kubectl_command):
        """Test getting pod prefix."""
        mock_output = {
            "status": {
                "components": {"predictor": {"latestReadyRevision": "revision-123"}}
            }
        }
        mock_run_kubectl_command.return_value = json.dumps(mock_output)

        prefix = NotebookPlugin.get_pod_prefix("test-inference-service")

        self.assertEqual(prefix, "revision-123")

    @patch.object(NotebookPlugin, "get_pods_in_namespace")
    @patch.object(NotebookPlugin, "get_pod_prefix")
    @patch.object(NotebookPlugin, "get_pod_logs")
    @patch.object(json, "dump")
    def test_get_logs_for_inference_service(
        self,
        mock_get_pod_logs,
        mock_get_pod_prefix,
        mock_get_pods_in_namespace,
    ):
        """Test getting logs for inference service."""
        mock_get_pod_prefix.return_value = "revision-123"
        mock_get_pods_in_namespace.return_value = [
            {"metadata": {"name": "revision-123-pod1"}},
            {"metadata": {"name": "revision-123-pod2"}},
        ]
        mock_get_pod_logs.side_effect = (
            lambda namespace, pod_name: f"Logs for {pod_name}"
        )

        logs_output = NotebookPlugin.get_logs_for_inference_service(
            "test-namespace", "test-inference-service"
        )

        self.assertIn("revision-123-pod1", logs_output)
        self.assertIn("revision-123-pod2", logs_output)
        self.assertEqual(logs_output["revision-123-pod1"], "Logs for revision-123-pod1")
        self.assertEqual(logs_output["revision-123-pod2"], "Logs for revision-123-pod2")


if __name__ == "__main__":
    unittest.main()
