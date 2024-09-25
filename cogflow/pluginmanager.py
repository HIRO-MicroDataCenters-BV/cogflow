"""
Plugin Manager Module

This module provides a PluginManager class responsible for managing plugins such as MlflowPlugin,
KubeflowPlugin, and DatasetPlugin.
It also includes functions to activate, deactivate, and check the status of plugins.

Attributes:
"""

import os
import configparser
import pandas as pd
from sqlalchemy import create_engine, Column, FLOAT, BIGINT, String
from sqlalchemy.orm import declarative_base

from . import plugin_config

Base = declarative_base()


class ModelTraining(Base):
    """
    SQLAlchemy ORM class representing the 'Model_training' table in the PostgreSQL database.

    This table stores information related to model training, including parameters, metrics,
    and other metadata associated with training runs.

    Attributes:
        param_key (str): Key for the parameter.
        param_value (str): Value of the parameter.
        model_name (str): Name of the model.
        model_version (float): Version of the model.
        creation_time (int): Creation time of the model, represented as a Unix timestamp.
        metric_key (str): Key for the metric.
        metric_value (float): Value of the metric.
        run_name (str): Name of the training run.
        run_uuid (str): UUID of the training run, serves as the primary key.
        user_id (str): Identifier for the user who initiated the training run.
    """

    __tablename__ = "Model_training"
    param_key = Column(String)
    param_value = Column(String)
    model_name = Column(String)
    model_version = Column(FLOAT)
    creation_time = Column(BIGINT)
    metric_key = Column(String)
    metric_value = Column(FLOAT)
    run_name = Column(String)
    run_uuid = Column(String, primary_key=True)
    user_id = Column(String)


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
    def connect_to_cogflow_db(dialect="postgresql+psycopg2"):
        """
        Create and return an SQLAlchemy engine for a any database.

        Args:
            dialect (str): The connection string for the database. By default postgres DB

        Returns:
            sqlalchemy.engine.Engine: The SQLAlchemy engine.
        """
        postgres_connection_str = (
            f"{dialect}://{plugin_config.COGFLOW_DB_USERNAME}:"
            f"{plugin_config.COGFLOW_DB_PASSWORD}@{plugin_config.COGFLOW_DB_HOST}:"
            f"{plugin_config.COGFLOW_DB_PORT}/{plugin_config.COGFLOW_DB_NAME}"
        )
        engine = create_engine(postgres_connection_str)
        engine.connect()
        return engine

    def insert_data_to_table(self, if_exists: str = "append"):
        """
        Insert data from a pandas DataFrame into a PostgreSQL table.

        Args:
            df (pd.DataFrame): The DataFrame containing data to insert.
            engine: The SQLAlchemy engine to use for the connection.
            table_name (str): The name of the PostgreSQL table to insert data into.
            if_exists (str): Behavior if the table already exists. 'append' to add data,
                             'replace' to drop and recreate the table. Defaults to 'append'.

        Returns:
            None
        """
        data_frame = self.get_params_and_metrics_from_mlflow_db()
        engine = self.connect_to_cogflow_db()
        table_name = ModelTraining.__tablename__
        data_frame.to_sql(table_name, engine, if_exists=if_exists, index=False)

    @staticmethod
    def connect_to_mlflow_db(dialect="mysql+pymysql"):
        """
        Establishes a connection to the MLflow database.

        This method creates an SQLAlchemy engine using the provided database dialect and connection
        parameters from the configuration. It then establishes a connection to the MLflow database
        and returns the engine.

        Args:
            dialect (str): The database dialect to use for the connection (e.g., "mysql+pymysql").
                           Defaults to "mysql+pymysql".

        Returns:
            sqlalchemy.engine.Engine: The SQLAlchemy engine connected to the MLflow database.
        """

        connection_str = (
            f"{dialect}://{plugin_config.ML_DB_USERNAME}:{plugin_config.ML_DB_PASSWORD}"
            f"@{plugin_config.ML_DB_HOST}:{plugin_config.ML_DB_PORT}/{plugin_config.ML_DB_NAME}"
        )
        engine = create_engine(connection_str)
        engine.connect()
        return engine

    def get_params_and_metrics_from_mlflow_db(self):
        """
        Retrieves parameters and metrics from the MLflow database.

        This method queries the MLflow database to retrieve parameters, metrics, and associated
        information such as model name, version, creation time, and other metadata for each run.
        The method executes a SQL query to join multiple tables and returns the results as a pandas
        DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the parameters and metrics data from the
                          MLflow database. Each row represents a run with columns for the
                          parameter key,
                          parameter value, model name, model version, creation time, metric key,
                          metric value, run name, run UUID, and user ID.
        """

        metric_query = """
        SELECT p.key AS param_key, p.value AS param_value,
               mv.name AS model_name, mv.version AS model_version, mv.creation_time,
               m.key AS metric_key, m.value AS metric_value,
               r.name AS run_name, r.run_uuid AS run_uuid, r.user_id
        FROM params p
        JOIN model_versions mv ON p.run_uuid = mv.run_id
        JOIN metrics m ON p.run_uuid = m.run_uuid
        JOIN runs r ON p.run_uuid = r.run_uuid;
        """
        engine = self.connect_to_mlflow_db()

        data_frame = pd.read_sql_query(metric_query, engine)
        return data_frame

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
        set_env_if_not_exists(
            "JUPYTER_USER_ID", config.get("settings", "JUPYTER_USER_ID")
        )
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
            "JUPYTER_USER_ID",
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
