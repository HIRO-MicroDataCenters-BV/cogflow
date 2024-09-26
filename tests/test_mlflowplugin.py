"""
    Test module for cases related to mlflowplugin
"""

import unittest
from unittest.mock import patch, MagicMock
from mlflow.exceptions import MlflowException, RestException
from ..cogflow.plugins.mlflowplugin import MlflowPlugin
from ..cogflow import (
    delete_registered_model,
    search_registered_models,
    load_model,
    register_model,
    autolog,
    create_registered_model,
    create_model_version,
    set_tracking_uri,
    set_experiment,
    get_artifact_uri,
    start_run,
    end_run,
    log_param,
    log_metric,
    log_model,
    search_model_versions,
)


class TestMlflowPlugin(unittest.TestCase):
    """
    Test Class for cases related to mlflow_plugin
    """

    # pylint: disable=too-many-public-methods

    def setUp(self):
        """
            Initial setup
        :return:
        """
        self.mlflow_plugin = MlflowPlugin()

    @patch("mlflow.get_artifact_uri")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_get_artifact_uri_with_run_id(
        self, mock_plugin_activation, mock_get_artifact_uri
    ):
        """
            test for get_artifact_uri_with_run_id
        :param mock_get_artifact_uri:
        :return:
        """
        # Mocking the mlflow get_artifact_uri method to return a specific URI
        mock_get_artifact_uri.return_value = "s3://your-bucket/artifacts/123"

        result = get_artifact_uri("123")

        # Asserting that the mock method was called with the correct argument
        mock_get_artifact_uri.assert_called_once()

        # Asserting that the method returned the expected artifact URI
        self.assertEqual(result, "s3://your-bucket/artifacts/123")
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.set_experiment")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_set_experiment(self, mock_plugin_activation, mock_set_experiment):
        """
            test for set_experiment
        :param mock_set_experiment:
        :return:
        """
        # Mocking the mlflow set_experiment method to raise an exception
        mock_set_experiment.side_effect = MlflowException("Failed to set experiment")

        with self.assertRaises(MlflowException):
            set_experiment("experiment_name")
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.set_tracking_uri")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_set_tracking_uri(self, mock_plugin_activation, mock_client):
        """
            test for set_tracking_uri
        :param mock_client:
        :return:
        """
        set_tracking_uri("your_tracking_uri")

        mock_client.assert_called_once_with("your_tracking_uri")
        mock_plugin_activation.assert_called_once()

    def test_version(self):
        """
            test for version
        :return:
        """
        version = self.mlflow_plugin.version()

        # Assert that the result matches the mocked version
        self.assertEqual(version, "2.1.1")

    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_is_alive(self, mock_plugin_activation):
        """
            test for is_alive
        :return:
        """
        # Mock the requests.get function to simulate response
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200  # Assuming Mlflow UI is accessible
            mock_response.text = "OK"  # Mock response message
            mock_get.return_value = mock_response

            mlflow_plugin = MlflowPlugin()  # Create an instance of MlflowPlugin

            status_code, message = mlflow_plugin.is_alive()

            self.assertEqual(status_code, 200)  # Check if status code is 200
            self.assertEqual(message, "OK")  # Check if response message is "OK"
            mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_is_alive_request_not_success(self, mock_plugin_activation):
        """
            test when there is request not successful in is_alive method
        :return:
        """
        # Mock the requests.get function to simulate response
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404  # Assuming Mlflow UI is not-accessible
            mock_response.text = "NOT-FOUND"  # Mock response message
            mock_get.return_value = mock_response

            mlflow_plugin = MlflowPlugin()  # Create an instance of MlflowPlugin

            status_code, message = mlflow_plugin.is_alive()

            self.assertEqual(status_code, 404)
            self.assertEqual(message, "NOT-FOUND")
            mock_plugin_activation.assert_called_once()

    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_is_alive_request_exception(self, mock_plugin_activation):
        """
            test when there is exception occured in is_alive method
        :return:
        """
        # Mock the requests.get function to simulate response
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("An error occurred .")

            mlflow_plugin = MlflowPlugin()  # Create an instance of MlflowPlugin
            with self.assertRaises(Exception):
                status_code, message = mlflow_plugin.is_alive()
                self.assertIsNotNone(status_code)
                self.assertIsNotNone(message)
            mock_plugin_activation.assert_called_once()

    @patch("mlflow.tracking.client.MlflowClient.create_model_version")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_create_model_version(self, mock_plugin_activation, mock_client):
        """
            test for create_model_version
        :param mock_client:
        :return:
        """
        mock_client.side_effect = MlflowException("Error occured")
        # # Call the method under test and expect it to raise an MlflowException
        with self.assertRaises(MlflowException):
            create_model_version("model_name", "source")
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.tracking.client.MlflowClient.create_registered_model")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_create_registered_model(self, mock_plugin_activation, mock_client):
        """
            test case for create_registered_model
        :param mock_client:
        :return:
        """
        create_registered_model("model_name")
        mock_client.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.register_model")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_register_model(self, mock_plugin_activation, mock_register_model):
        """
            test case for register_model
        :param mock_register_model:
        :return:
        """
        # Define inputs
        model = MagicMock()
        model_uri = "my_model_uri"
        register_model(model, model_uri)
        mock_register_model.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.sklearn.load_model")
    @patch("mlflow.utils.rest_utils.http_request")
    def test_load_model_exception(self, mock_http_request, mock_load_model):
        """
            test for exception in load_model
        :param mock_http_request:
        :param mock_load_model:
        :return:
        """
        # Define inputs
        model_name = "my_model"
        model_version = 1
        expected_model = MagicMock()
        mock_load_model.return_value = expected_model

        # Mocking the HTTP request to raise MlflowException
        mock_http_request.side_effect = MlflowException(
            "API request failed with exception HTTPConnectionPool: "
            "Max retries exceeded with url: http://127.0.0.1:5001"
        )
        try:
            mock_load_model.assert_not_called()
        except MlflowException:
            with self.assertRaises(MlflowException):
                load_model(model_name, model_version)

    @patch("cogflow.cogflow.plugins.mlflowplugin.MlflowPlugin.load_model")
    def test_load_model(self, mock_load_model):
        """
        Test load_model method in mlflow_plugin.
        """
        # Define inputs
        model_name = "tracking-quickstart"
        model_version = 1
        expected_model = MagicMock()

        # Set the return value of the mocked load_model function
        mock_load_model.return_value = expected_model

        # Call the load_model method
        result = load_model(model_name, model_version)

        # Verify that the result is equal to the expected model
        self.assertEqual(result, expected_model)

    @patch("mlflow.autolog")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_autolog(self, mock_plugin_activation, mock_autolog):
        """
            test for autolog
        :param mock_autolog:
        :return:
        """
        # Call the method under test
        autolog()

        # Assert that autolog was called
        mock_autolog.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.tracking.client.MlflowClient.delete_registered_model")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_delete_registered_model(
        self, mock_plugin_activation, mock_delete_registered_model
    ):
        """
            test for delete_registered_model
        :param mock_delete_registered_model:
        :return:
        """
        # Mock successful deletion
        model_name = "test_model"
        mock_delete_registered_model.return_value = True

        # Call the method under test
        result = delete_registered_model(model_name)

        # Assert that the method returns True
        self.assertTrue(result)

        # Assert that delete_registered_model was called with the correct argument
        mock_delete_registered_model.assert_called_once_with(model_name)
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.search_registered_models")
    def test_search_registered_models_exception(self, mock_search_registered_models):
        """
            test for exception occurs when search for registered model
        :param mock_search_registered_models:
        :return:
        """
        # Mock any other unexpected exception
        mock_search_registered_models.side_effect = MlflowException(
            "API request failed with exception HTTPConnectionPool: "
            "Max retries exceeded with url: http://127.0.0.1:5001"
        )
        # Assert that the method raises the expected exception
        try:
            mock_search_registered_models.assert_not_called()
        except MlflowException:
            with self.assertRaises(MlflowException):
                self.mlflow_plugin.search_registered_models()

    @patch("mlflow.tracking.client.MlflowClient.search_registered_models")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_search_registered_models(
        self, mock_plugin_activation, mock_search_registered_models
    ):
        """
            test for search_registered_model
        :param mock_search_registered_models:
        :return:
        """
        mock_search_registered_models.return_value = None
        # Call the method under test
        search_registered_models()

        # Assert that autolog was called
        mock_search_registered_models.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.start_run")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_start_run_with_experiment_and_run_name(
        self, mock_plugin_activation, mock_start_run
    ):
        """
            test for start_run_with_experiment with run_name
        :param mock_start_run:
        :return:
        """
        # experiment_name = "test_experiment"
        run_name = "test_run"
        start_run(run_name=run_name)
        # Assert that start_run was called with the correct arguments
        mock_start_run.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.end_run")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_end_run(self, mock_plugin_activation, mock_end_run):
        """
            test for end_run
        :param mock_end_run:
        :return:
        """
        # Call the method under test
        end_run()
        # Assert that end_run was called
        mock_end_run.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.log_param")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_log_param(self, mock_plugin_activation, mock_log_param):
        """
            test for log_param
        :param mock_log_param:
        :return:
        """
        # Define inputs
        run = MagicMock()
        params = {"param1": 10, "param2": "value"}

        # Call the method under test
        log_param(run, params)

        # Assert that log_param was called with the correct arguments
        mock_log_param.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.log_metric")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_log_metric(self, mock_plugin_activation, mock_log_metric):
        """
            test for log_metric
        :param mock_log_metric:
        :return:
        """
        # Define inputs
        run = MagicMock()
        metrics = {"accuracy": 0.85, "loss": 0.1}

        # Call the method under test
        log_metric(run, metrics)

        # Assert that log_metric was called with the correct arguments
        mock_log_metric.assert_called_once()
        mock_plugin_activation.assert_called_once()

    @patch("requests.post")
    @patch("os.getenv")
    @patch("mlflow.active_run")
    @patch(
        "cogflow.cogflow.plugins.notebook_plugin.NotebookPlugin.get_model_latest_version"
    )
    @patch("mlflow.sklearn.log_model")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_log_model(
        self,
        mock_plugin_activation,
        mock_log_model,
        mock_model_version,
        mock_active_run,
        mock_env,
        mock_requests_post,
    ):
        """
            test for log_model
        :param mock_log_model:
        :return:
        """
        # Define inputs
        sk_model = MagicMock()
        artifact_path = "model"
        # Define any other necessary inputs for the log_model method

        mock_run = MagicMock()
        mock_run.info.run_id = "12345"

        # Set the return value of mlflow.active_run()
        mock_active_run.return_value = mock_run

        mock_env.side_effect = lambda x: {
            "API_BASEPATH": "http://randomn",
        }[x]

        mock_requests_post.return_value.status_code = 201

        mock_model_version.return_value = 1
        # Call the method under test
        log_model(sk_model=sk_model, artifact_path=artifact_path)
        # Assert that log_model was called with the correct arguments
        mock_log_model.assert_called_once()
        mock_plugin_activation.assert_called()

    @patch("requests.post")
    @patch("os.getenv")
    @patch("mlflow.sklearn.log_model")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_log_model_exception(
        self, mock_plugin_activation, mock_log_model, mock_env, mock_requests_post
    ):
        """
            test log_model when exception occurs
        :param mock_log_model:
        :return:
        """
        # Define inputs
        sk_model = MagicMock()
        artifact_path = "model"
        # Define any other necessary inputs for the log_model method

        mock_env.side_effect = lambda x: {
            "API_BASEPATH": "http://randomn",
        }[x]

        mock_requests_post.return_value.status_code = 201

        # Set up the side effect to raise MlflowException
        mock_log_model.side_effect = MlflowException(
            "API request failed with exception HTTPConnectionPool: "
            "Max retries exceeded with url: http://127.0.0.1:5001"
        )

        # Call the method under test and assert that it raises an exception
        with self.assertRaises(MlflowException):
            self.mlflow_plugin.log_model(sk_model=sk_model, artifact_path=artifact_path)
        mock_plugin_activation.assert_called_once()

    @patch("mlflow.MlflowClient.search_model_versions")
    @patch("cogflow.cogflow.pluginmanager.PluginManager.verify_activation")
    def test_search_model_versions(
        self, mock_verify_activation, mock_search_model_versions
    ):
        """
        test for search_model_versions
        """
        filter_string = "name='custom_model'"

        # Call the method being tested
        search_model_versions(
            filter_string=filter_string,
        )
        mock_search_model_versions.assert_called_once()
        mock_verify_activation.assert_called_once()

    @patch("mlflow.tracking.client.MlflowClient.get_model_version")
    def test_get_model_uri(self, mock_client):
        """
        test for get_model_uri
        """
        mock_model_version = MagicMock()
        mock_model_version.source = "s3://bucket/model/1"

        # Set up the mock client to return the mock model version
        mock_client.return_value = mock_model_version

        # Call the function
        mfp = MlflowPlugin()
        model_uri = mfp.get_model_uri("my_model", "1")

        # Assertions
        self.assertEqual(model_uri, "s3://bucket/model/1")
        mock_client.assert_called_once_with(name="my_model", version="1")

    @patch("mlflow.tracking.MlflowClient.get_model_version")
    def test_model_version_not_found(self, mock_client):
        """
        test for model_version
        """
        # Set up the mock client to raise a RestException
        mock_client.side_effect = RestException(
            json={
                "error_code": "RESOURCE_DOES_NOT_EXIST",
                "message": "Model version not found",
            }
        )

        # Call the function
        mfp = MlflowPlugin()
        with self.assertRaises(RestException):
            mfp.get_model_uri("my_model", "999")

        # Assertions
        mock_client.assert_called_once_with(name="my_model", version="999")


if __name__ == "__main__":
    unittest.main()
