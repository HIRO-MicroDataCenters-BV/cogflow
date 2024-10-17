"""
Plugin Manager Module

This module provides a PluginManager class responsible for managing plugins such as MlflowPlugin,
KubeflowPlugin, and DatasetPlugin.
It also includes functions to activate, deactivate, and check the status of plugins.

Attributes:
"""

import os
import configparser
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PluginManager:
    """
    Class responsible for managing plugins.

    Attributes:
    """

    def __init__(self):
        """
        Initializes the PluginManager with plugin classes.
        """
        self.config_file_path = os.path.join(
            os.path.dirname(__file__), "cogflow_config.ini"
        )

    @staticmethod
    def get_config_value(config_file_path, section, key="activation_key"):
        """
        Reads the activation status of a plugin from an INI configuration file.

        Args:
            config_file_path (str): The path to the INI configuration file.
            section (str): The section in the INI file that contains the plugin settings.
            key (str, optional): The key used to store activation status in the section.
            Default is 'activation_key'.

        Returns:
            bool: True if the plugin is activated, False otherwise.
            str or None: The value of the key if no exceptions occur and the plugin is activated.
        """
        if not config_file_path:
            raise FileNotFoundError("Configuration file path not provided.")

        # Create a ConfigParser instance
        config = configparser.ConfigParser()

        try:
            # Read the INI configuration file
            config.read(config_file_path)

            # Check if the configuration file is empty or improperly formatted
            if not config.sections():
                raise Exception(
                    "Configuration file is empty or not properly formatted."
                )

            # Check if the specified section exists in the configuration file
            if section not in config:
                raise KeyError(
                    f"Section '{section}' not found in the configuration file. "
                    f"Please correct section name in configuration file."
                )

            value = config.get(section, key, fallback=None)

            # Check if the key value can be converted to boolean
            if value is not None:
                try:
                    # Try to convert the value to boolean
                    activation_status = config.getboolean(section, key, fallback=False)
                    return activation_status
                except ValueError:
                    # If conversion to boolean fails, return the value as a string
                    return value
            else:
                raise KeyError(
                    f"Key '{key}' not found in section '{section}' of the configuration file."
                    f" Please correct key name in configuration file"
                )

        except (FileNotFoundError, KeyError, Exception) as exp:
            # Stop execution immediately and raise the exception with a specific message
            raise Exception(f"Error : {str(exp)}")

    def verify_activation(self, section):
        """
        Verify if the plugin is activated.

        Raises:
            Exception: If the plugin is not activated.
        """
        try:
            # Call read_activation_status to check the activation status
            activation_status = self.get_config_value(self.config_file_path, section)
            # Raise an exception if the activation status is False
            if not activation_status:
                raise Exception(
                    "Plugin is not activated. Please activate the "
                    "plugin before performing this action."
                )

        except Exception as exp:
            error_message = f"{str(exp)}"
            # Log or print the error message if necessary
            print(error_message)
            raise

    def load_config(self):
        """Load configuration from the config.ini file."""
        config = configparser.ConfigParser()
        config.read(self.config_file_path)

        # Helper function to set env variable only if it's not already set
        def set_env_if_not_exists(var_name, value):
            if not os.getenv(var_name):
                os.environ[var_name] = value

        # Set environment variables from the config file if not already set
        set_env_if_not_exists("API_BASEPATH", config.get("settings", "API_BASEPATH"))
        set_env_if_not_exists("TIMER_IN_SEC", config.get("settings", "TIMER_IN_SEC"))
        set_env_if_not_exists("FILE_TYPE", config.get("settings", "FILE_TYPE"))
        set_env_if_not_exists(
            "MLFLOW_TRACKING_URI", config.get("settings", "MLFLOW_TRACKING_URI")
        )
        set_env_if_not_exists(
            "MLFLOW_S3_ENDPOINT_URL", config.get("settings", "MLFLOW_S3_ENDPOINT_URL")
        )
        set_env_if_not_exists("ML_TOOL", config.get("settings", "ML_TOOL"))

        # Validate that the environment variables are set
        required_vars = [
            "API_BASEPATH",
            "TIMER_IN_SEC",
            "FILE_TYPE",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_S3_ENDPOINT_URL",
            "ML_TOOL",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"The following environment variables are missing or not set: {', '.join(missing_vars)}"
            )

    def load_path(self, path_name):
        """
        loads the path variable for the api endpoints from config.ini
        """
        return self.get_config_value(
            config_file_path=self.config_file_path, section="path", key=path_name
        )
