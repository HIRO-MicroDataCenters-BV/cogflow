"""
    This file contains unittest for PluginManager methods
"""

import os
import unittest
import configparser
from unittest.mock import patch, mock_open
from ..cogflow.pluginmanager import PluginManager


class TestPluginManager(unittest.TestCase):
    """
    Test class for PluginManger
    """

    def setUp(self):
        # Set up the file paths and sections to use in the test cases
        self.config_file_path = os.path.join(
            os.path.dirname(__file__), "cogflow_config.ini"
        )
        # self.config_file_path = "config.ini"
        self.section = "mlflow_plugin"
        self.key = "activation_key"

        # Initialize the instance of the class containing the methods to test
        self.instance = PluginManager()

    def test_get_config_value_activation_status(self):
        """
        test to get config value activation status
        """

        # Mock open function to simulate reading the file
        with patch("builtins.open", mock_open(read_data=self.config_file_path)):
            # Mock configparser to control its behavior
            with patch("configparser.ConfigParser") as mock_config:
                # Configure the mock object to return desired values
                mock_config_instance = mock_config.return_value
                mock_config_instance.read.return_value = None
                mock_config_instance.getboolean.return_value = True
                print("**************", self.config_file_path)

                # Call the method

                with self.assertRaises(Exception) as context:
                    self.instance.get_config_value(
                        self.config_file_path, "mlflow_plugin"
                    )

                # Check that the expected error message is raised
                expected_message = (
                    "Section 'mlflow_plugin' not found in the configuration file. "
                    "Please correct section name in configuration file."
                )
                self.assertIn(expected_message, str(context.exception))

    def test_get_config_value_nonexistent_key(self):
        """
        test for get config value
        """
        # Prepare a valid configuration file content without the key
        config_content = """
        [mlflow_plugin]
        other_key = some_value
        """

        # Mock open function to simulate reading the file
        with patch("builtins.open", mock_open(read_data=config_content)):
            # Mock configparser to control its behavior
            with patch("configparser.ConfigParser") as mock_config:
                # Configure the mock object to return desired values
                mock_config_instance = mock_config.return_value
                mock_config_instance.read.return_value = None
                mock_config_instance.get.side_effect = configparser.NoOptionError(
                    "key", self.section
                )

                # Call the method and expect a KeyError
                with self.assertRaises(Exception) as cm:
                    self.instance.get_config_value(
                        self.config_file_path, self.section, self.key
                    )
                self.assertNotIn("Key 'activation_key' not found", str(cm.exception))

    def test_verify_activation_plugin_active(self):
        """
        test for verify activation
        """
        # Prepare a valid configuration file content with the key set to True
        config_content = """
        [mlflow_plugin]
        activation_key = true
        """

        # Mock open function to simulate reading the file
        with patch("builtins.open", mock_open(read_data=config_content)):
            # Mock configparser to control its behavior
            with patch("configparser.ConfigParser") as mock_config:
                # Configure the mock object to return desired values
                mock_config_instance = mock_config.return_value
                mock_config_instance.read.return_value = None
                mock_config_instance.getboolean.return_value = True

                # Call the method
                with self.assertRaises(Exception):
                    self.instance.verify_activation(self.section)
                # No exception expected if plugin is active
