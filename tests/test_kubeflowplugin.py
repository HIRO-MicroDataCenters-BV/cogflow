"""
This module is to test functionalities of kubeflowplugin.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from kubernetes.client.models import V1EnvVar
from ..cogflow import (
    pipeline,
    create_component_from_func,
    client,
    serve_model_v2,
    serve_model_v1,
    load_component,
    delete_pipeline,
)
from ..cogflow.plugins.kubeflowplugin import KubeflowPlugin, CogContainer


class TestKubeflowPlugin(unittest.TestCase):
    """Test cases for KubeflowPlugin class."""

    def setUp(self):
        """Set up method to initialize plugin."""
        self.kfp_plugin = KubeflowPlugin()

    @patch("kfp.dsl.pipeline")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_pipeline_with_name_and_description(
        self, mock_plugin_activation, mock_pipeline
    ):
        """Test pipeline function with name and description."""
        pipeline()
        mock_pipeline.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("kfp.components.create_component_from_func")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_create_component_from_func(
        self, mock_plugin_activation, mock_create_component
    ):
        """Test create_component_from_func."""
        func = MagicMock()
        create_component_from_func(func)
        mock_create_component.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("kfp.Client")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_client(self, mock_plugin_activation, mock_client):
        """Test client function."""
        # Arrange
        client()

        # Assertion
        mock_client.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("kfp.components.load_component_from_url")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_load_component_from_url_success(
        self, mock_plugin_activation, mock_load_component
    ):
        """Test loading component from URL."""
        # Mock a successful component loading
        expected_component_spec = MagicMock()
        mock_load_component.return_value = expected_component_spec

        # Define a sample URL
        url = "http://example.com/component.tar.gz"

        # Call the function under test
        result = load_component(url=url)

        # Assert that the function returns the expected component specification
        self.assertEqual(result, expected_component_spec)

        # Assert that load_component_from_url was called with the correct URL
        mock_load_component.assert_called_once_with(url)
        mock_plugin_activation.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "DB_HOST": "postgres",
            "DB_PORT": "5432",
            "DB_USER": "hiro",
            "DB_PASSWORD": "hiropwd",
            "DB_NAME": "cognitiveDB",
            "AWS_ACCESS_KEY_ID": "access_key_id_value",
            "AWS_SECRET_ACCESS_KEY": "secret_access_key_value",
            "MINIO_BUCKET_NAME": "MLFLOW",
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:5001",
            "MINIO_ENDPOINT_URL": "localhost:9000",
            "MLFLOW_S3_ENDPOINT_URL": "C:/path/to/mlruns",
        },
    )
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_add_model_access_with_valid_env_vars(self, mock_plugin_activation):
        """Test that valid environment variables are added to the container."""

        container = CogContainer()
        container_with_env_vars = container.add_model_access()

        # List of expected environment variables
        expected_env_vars = [
            V1EnvVar(name="DB_HOST", value="postgres"),
            V1EnvVar(name="DB_PORT", value="5432"),
            V1EnvVar(name="DB_USER", value="hiro"),
            V1EnvVar(name="DB_PASSWORD", value="hiropwd"),
            V1EnvVar(name="DB_NAME", value="cognitiveDB"),
            V1EnvVar(name="AWS_ACCESS_KEY_ID", value="access_key_id_value"),
            V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="secret_access_key_value"),
            V1EnvVar(name="MINIO_BUCKET_NAME", value="MLFLOW"),
            V1EnvVar(name="MLFLOW_TRACKING_URI", value="http://127.0.0.1:5001"),
            V1EnvVar(name="MINIO_ENDPOINT_URL", value="localhost:9000"),
            V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="C:/path/to/mlruns"),
        ]

        # Check that all expected environment variables are in the container
        actual_env_vars = [env_var.to_dict() for env_var in container_with_env_vars.env]
        mock_plugin_activation.assert_called_once()
        for expected_env_var in expected_env_vars:
            self.assertIn(expected_env_var.to_dict(), actual_env_vars)

    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_serve_model_v2(self, mock_plugin_activation):
        """Test serving model v2."""
        # Patch Kubernetes client to avoid loading kube config
        with patch("kubernetes.config.load_kube_config"):
            model_uri = "sample_model_uri"
            name = "test_model_name"

            # Call the function and assert that it raises MaxRetryError
            with self.assertRaises(Exception):
                serve_model_v2(model_uri, name)
            mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_serve_model_v2_no_name(self, mock_plugin_activation):
        """Test serving model v2 without a name."""
        with patch("kubernetes.config.load_kube_config"):
            model_uri = "sample_model_uri"

            # Call the function and assert that it raises MaxRetryError
            with self.assertRaises(Exception):
                self.kfp_plugin.serve_model_v2(model_uri)
            mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_serve_model_v1_with_exception(self, mock_plugin_activation):
        """Test serving model v1 with an exception."""
        # Define input parameters
        model_uri = "example_model_uri"
        name = "test_model_name"

        # Call the function and assert that it raises MaxRetryError
        with self.assertRaises(Exception):
            serve_model_v1(model_uri, name)
        mock_plugin_activation.assert_called_once()

    # @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    # def test_get_model_url(self, mock_plugin_activation):
    #     """Test get model URL functionality."""
    #     model_name = "test_model"
    #
    #     with self.assertRaises(Exception):
    #         # Call the method you're testing here
    #         get_model_url(model_name)
    #     mock_plugin_activation.assert_called_once()

    @patch("requests.delete")
    @patch("requests.get")
    @patch("os.getenv")
    @patch("cogflow.cogflow.plugins.kubeflowplugin.KubeflowPlugin.client")
    def test_delete_pipeline(
        self, mock_client, mock_env, mock_request_get, mock_request_delete
    ):
        """Test deleting a pipeline."""
        # Arrange
        mock_env.side_effect = lambda x: {
            "MLFLOW_S3_ENDPOINT_URL": "localhost:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
            "API_BASEPATH": "http://randomn",
            "TIMER_IN_SEC": "10",
            "FILE_TYPE": "2",
            "MLFLOW_TRACKING_URI": "http://mlflow",
            "ML_TOOL": "ml_flow",
            "COGFLOW_CONFIG_FILE_PATH": "/path/to/config",
        }[x]
        mock_request_get.return_value.status_code = 200
        mock_request_delete.return_value.status_code = 200
        mock_client_instance = mock_client.return_value
        pipeline_id = "test_pipeline_id"

        # Act
        delete_pipeline(pipeline_id)

        # Assert
        mock_client_instance.delete_pipeline.assert_called_once_with(
            pipeline_id=pipeline_id
        )

    @patch("cogflow.cogflow.plugins.kubeflowplugin.KubeflowPlugin.client")
    def test_list_pipeline_versions(self, mock_client):
        """Test listing pipeline versions."""
        # Arrange
        plugin = KubeflowPlugin()
        mock_client_instance = mock_client.return_value
        pipeline_id = "test_pipeline_id"
        expected_response = "expected_response"
        mock_client_instance.list_pipeline_versions.return_value = expected_response

        # Act
        response = plugin.list_pipeline_versions(pipeline_id)

        # Assert
        mock_client_instance.list_pipeline_versions.assert_called_once_with(
            pipeline_id=pipeline_id
        )
        self.assertEqual(response, expected_response)

    @patch("cogflow.cogflow.plugins.kubeflowplugin.KubeflowPlugin.client")
    def test_delete_pipeline_version(self, mock_client):
        """Test deleting a pipeline version."""
        # Arrange
        plugin = KubeflowPlugin()
        mock_client_instance = mock_client.return_value
        version_id = "test_version_id"

        # Act
        plugin.delete_pipeline_version(version_id)

        # Assert
        mock_client_instance.delete_pipeline_version.assert_called_once_with(
            version_id=version_id
        )

    @patch("cogflow.cogflow.plugins.kubeflowplugin.KubeflowPlugin.client")
    def test_delete_runs(self, mock_client):
        """Test deleting pipeline runs."""
        # Arrange
        plugin = KubeflowPlugin()
        mock_client_instance = mock_client.return_value
        mock_client_instance.runs = MagicMock()
        run_ids = [1, 2]

        # Act
        plugin.delete_runs(run_ids)

        # Assert
        calls = [unittest.mock.call(id=1), unittest.mock.call(id=2)]
        mock_client_instance.runs.delete_run.assert_has_calls(calls, any_order=True)
        self.assertEqual(mock_client_instance.runs.delete_run.call_count, 2)

    @patch("kubernetes.config.load_kube_config")
    def test_outside_cluster_with_mock_kubeconfig(self, mock_load_kube_config):
        """Test kubeconfig outside cluster"""
        # Mock the kubeconfig return value (example structure)
        mock_load_kube_config.return_value = {
            "clusters": [
                {"name": "cluster-1", "cluster": {"server": "http://example.com"}}
            ],
            "contexts": [{"name": "context-1", "context": {"namespace": "default"}}],
            "users": [{"name": "user-1", "user": {"token": "fake-token"}}],
        }

        # Act: Now test the function with the mocked kubeconfig
        kubeflow_plugin = KubeflowPlugin()
        result = kubeflow_plugin.get_default_namespace()

        # Assert: Check the namespace in the context
        assert result == "default", f"Expected 'development', but got {result}"


if __name__ == "__main__":
    unittest.main()
