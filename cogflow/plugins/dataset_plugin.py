"""
This module provides functionality related to Dataset upload via plugin.
"""

import io
import os
import json
from typing import Union
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import requests
from minio import Minio
from mlflow.models.signature import ModelSignature
from scipy.sparse import csr_matrix, csc_matrix
from .. import plugin_config
from ..pluginmanager import PluginManager
from ..util import make_post_request
from .notebook_plugin import NotebookPlugin
from .mlflowplugin import MlflowPlugin


class DatasetMetadata:
    """
    Class used for  metadata of Dataset
    """

    def __init__(self, name, description, source, fmt: str):
        self.name = name
        self.description = description
        self.source = source
        self.format = fmt

    def is_file_path(self):
        """
            method to check if the source of  dataset
            is local file path
        :return: boolean true or false
        """
        return os.path.isfile(self.source)

    def is_external_url(self):
        """
            method to check if source of dataset is
            external url
        :return: boolean true or false
        """
        parsed_url = urlparse(self.source)
        return bool(parsed_url.scheme) and parsed_url.netloc

    def to_dict(self):
        """
        return  object as dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "format": self.format,
        }


class DatasetPlugin:
    """
    A class to handle dataset-related operations.
    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes DatasetPlugin with environment variables.
        """
        # Retrieve MinIO connection details from environment variables
        self.minio_endpoint = os.getenv(plugin_config.MINIO_ENDPOINT_URL)
        # Check if the environment variable exists and has a value
        if self.minio_endpoint:
            # Remove the http:// or https:// prefix using string manipulation
            if self.minio_endpoint.startswith(("http://", "https://")):
                # Find the index where the protocol ends
                protocol_end_index = self.minio_endpoint.find("//") + 2
                # Get the remaining part of the URL (without the protocol)
                self.minio_endpoint = self.minio_endpoint[protocol_end_index:]
        else:
            print("MLFLOW_S3_ENDPOINT_URL environment variable is not set.")
        self.minio_access_key = os.getenv(plugin_config.MINIO_ACCESS_KEY)
        self.minio_secret_key = os.getenv(plugin_config.MINIO_SECRET_ACCESS_KEY)
        self.section = "dataset_plugin"

    def create_minio_client(self):
        """
        Creates a MinIO client object.
        Returns:
            Minio: The MinIO client object.
        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)
        return Minio(
            self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            secure=False,
        )  # Change to True if using HTTPS

    def query_endpoint_and_download_file(self, url, output_file, bucket_name):
        """
        Queries an endpoint and downloads a file from it.
        Args:
            url (str): The URL of the endpoint.
            output_file (str): The name of the output file to save.
            bucket_name (str): The name of the bucket.
        Returns:
            tuple: A tuple containing a boolean indicating success and the file URL if successful.
        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                self.save_to_minio(response.content, output_file, bucket_name)
                return True
            print(f"Request failed with status code {response.status_code}")
            raise Exception("Request could not be successful due to error")
        except requests.exceptions.RequestException as exp:
            print(f"An error occurred: {exp}")
            raise Exception("Exception occurred during the requested operation")

    def save_to_minio(self, file_content, output_file, bucket_name):
        """
        Saves a file to MinIO.
        Args:
            file_content (bytes): The content of the file to be uploaded.
            output_file (str): The name of the file to be uploaded.
            bucket_name (str): The name of the bucket to upload the file to.
        Returns:
            str: The presigned URL of the uploaded file.
        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)
        # Initialize MinIO client
        minio_client = self.create_minio_client()
        object_name = output_file
        # Check if the bucket exists, if not, create it
        bucket_exists = minio_client.bucket_exists(bucket_name)
        if not bucket_exists:
            try:
                minio_client.make_bucket(bucket_name)
                print(f"Bucket '{bucket_name}' created successfully.")
            except Exception as exp:
                print(f"Bucket '{bucket_name}' couldnot be created.")
                raise exp
        # Put file to MinIO
        try:
            # Upload content to MinIO bucket
            minio_client.put_object(
                bucket_name,
                object_name,
                io.BytesIO(file_content),
                len(file_content),
            )
            print(
                f"File {output_file} uploaded successfully to MinIO bucket"
                f" {bucket_name} as {object_name}."
            )
            presigned_url = minio_client.presigned_get_object(bucket_name, object_name)
            print(f"Access URL for '{object_name}': {presigned_url}")
            return presigned_url
        except Exception as err:
            print(f"Error uploading file: {err}")
            raise Exception(f"Error uploading file: {err}")

    def delete_from_minio(self, object_name, bucket_name):
        """
        Deletes a file from MinIO.
        Args:
            object_name (str): The name of the object (file) to be deleted.
            bucket_name (str): The name of the bucket containing the file.
        Returns:
            bool: True if the file was successfully deleted, False otherwise.
        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)
        # Initialize MinIO client
        minio_client = self.create_minio_client()
        try:
            # Check if the object exists
            object_exists = minio_client.stat_object(bucket_name, object_name)
            if object_exists:
                # Delete the object from the bucket
                minio_client.remove_object(bucket_name, object_name)
                print(
                    f"File '{object_name}' deleted successfully from bucket '{bucket_name}'."
                )
                return True
            print(f"File '{object_name}' does not exist in bucket '{bucket_name}'.")
            return False
        except Exception as err:
            print(
                f"Error deleting file '{object_name}' from bucket '{bucket_name}': {err}"
            )
            return False

    @staticmethod
    def register_dataset(details: DatasetMetadata):
        """
        Register a dataset with the given details.

        Args:
            details (DatasetMetadata): Details of the dataset to register, including name, source,
                description, and other metadata.

        Returns:
            dict: A dictionary containing information about the registered dataset, including its
                ID, name, description, and other metadata.

        Raises:
            Exception: If an error occurs during the registration process.
        """
        # Verify plugin activation
        PluginManager().verify_activation(DatasetPlugin().section)

        PluginManager().load_config()

        try:
            output_file = details.name.replace(
                " ", "_"
            )  # if details.name has spaces in it
            params = None
            data = None
            files = None
            if details.is_external_url():
                # If the dataset is hosted online
                path = PluginManager().load_path("dataset_register")
                data = {
                    "url": details.source,
                    "file_name": output_file,
                }
            elif details.is_file_path():
                path = PluginManager().load_path("dataset")
                params = {
                    "dataset_type": 1,
                    "dataset_source_type": 0,
                    "dataset_name": details.name,
                    "description": details.description,
                }
                files = details.source
            else:
                print("Not a valid source")
                raise Exception("Not a valid source")
            url = os.getenv(plugin_config.API_BASEPATH) + path
            return make_post_request(url=url, data=data, params=params, files=files)
        except Exception as exp:
            print(str(exp))
            raise exp

    def save_dataset_details(self, dataset):
        """
            method to save dataset details
        :param dataset: dataset details
        :return: dataset_id from the db
        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)

        response = self.register_dataset(dataset)
        dataset_id = response["data"]["dataset_id"]
        return dataset_id

    def log_model_with_dataset(
        self,
        sk_model,
        artifact_path,
        dataset: DatasetMetadata,
        conda_env=None,
        code_paths=None,
        serialization_format="cloudpickle",
        registered_model_name=None,
        signature: ModelSignature = None,
        input_example: Union[
            pd.DataFrame,
            np.ndarray,
            dict,
            list,
            csr_matrix,
            csc_matrix,
            str,
            bytes,
            tuple,
        ] = None,
        await_registration_for=300,
        pip_requirements=None,
        extra_pip_requirements=None,
        pyfunc_predict_fn="predict",
        metadata=None,
    ):
        """
        Log a scikit-learn model to Mlflow and link dataset to model.

        Args:
            sk_model: The scikit-learn model to be logged.
            artifact_path (str): The run-relative artifact path to which the model artifacts will
            be saved.
            conda_env (str, optional): The path to a Conda environment YAML file. Defaults to None.
            code_paths (list, optional): A list of local filesystem paths to Python files that
            contain code to be
            included as part of the model's logged artifacts. Defaults to None.
            dataset (DatasetMetadata): Metadata of the dataset to link with the model.
            serialization_format (str, optional): The format used to serialize the model. Defaults
            to "cloudpickle".
            registered_model_name (str, optional): The name under which to register the model with
            Mlflow. Defaults to None.
            signature (ModelSignature, optional): The signature defining model input and output
            data types and shapes. Defaults to None.
            input_example (Union[pd.DataFrame, np.ndarray, dict, list, csr_matrix, csc_matrix, str,
            bytes, tuple], optional): An example input to the model. Defaults to None.
            await_registration_for (int, optional): The duration, in seconds, to wait for the
            model version to finish being created and is in the READY status. Defaults to 300.
            pip_requirements (str, optional): A file in pip requirements format specifying
            additional pip dependencies for the model environment. Defaults to None.
            extra_pip_requirements (str, optional): A string containing additional pip dependencies
            that should be added to the environment. Defaults to None.
            pyfunc_predict_fn (str, optional): The name of the function to invoke for prediction,
            when the model is a PyFunc model. Defaults to "predict".
            metadata (dict, optional): A dictionary of metadata to log with the model.
            Defaults to None.

        Returns:
            Model: The logged scikit-learn model.

        Raises:
            Exception: If an error occurs during the logging process.

        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)

        result = MlflowPlugin().log_model(
            sk_model=sk_model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            code_paths=code_paths,
            serialization_format=serialization_format,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            await_registration_for=await_registration_for,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            pyfunc_predict_fn=pyfunc_predict_fn,
            metadata=metadata,
        )
        # save model details in DB
        response = NotebookPlugin().save_model_details_to_db(registered_model_name)
        model_id = response["data"]["id"]
        # save the dataset details
        dataset_id = self.save_dataset_details(dataset)
        # link model and dataset
        NotebookPlugin().link_model_to_dataset(dataset_id, model_id)
        return result

    def get_dataset(self, name):
        """
        get dataset file after register it by giving the name
        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)

        PluginManager().load_config()

        path = f"{PluginManager().load_path('dataset')}/{name}"
        url = os.getenv(plugin_config.API_BASEPATH) + path
        response = requests.get(url, timeout=10)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.text
            client = self.create_minio_client()
            # Define the S3 bucket name and the file name
            result = json.loads(result)
            bucket_name = result["data"][0]["dataset_uploads"][0]["file_path"].split(
                "//"
            )[-1]
            file_name = result["data"][0]["dataset_uploads"][0]["file_name"]

            client.fget_object(bucket_name, file_name, file_name)

            print(f"Downloaded {file_name} from {bucket_name}")
            return file_name

        return None
