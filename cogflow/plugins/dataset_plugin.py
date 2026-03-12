"""
This module provides functionality related to Dataset upload via plugin.
"""

import io
import os
import uuid
from typing import Union
from urllib.parse import urlparse
from uuid import UUID

import numpy as np
import pandas as pd
import httpx
from minio import Minio
from mlflow.models.signature import ModelSignature
from scipy.sparse import csr_matrix, csc_matrix
from .. import plugin_config
from ..pluginmanager import PluginManager
from ..util import make_post_request, make_get_request, async_make_get_request
from .notebook_plugin import NotebookPlugin
from .mlflowplugin import MlflowPlugin
from .kubeflowplugin import KubeflowPlugin


class DatasetMetadata:
    """
    Class used for  metadata of Dataset
    """

    def __init__(self, name, description, file_path, dataset_type: str):
        self.name = name
        self.description = description
        self.file_path = file_path
        self.dataset_type = dataset_type

    def is_file_path(self):
        """
            method to check if the source of  dataset
            is local file path
        :return: boolean true or false
        """
        return os.path.isfile(self.file_path)

    def is_external_url(self):
        """
            method to check if source of dataset is
            external url
        :return: boolean true or false
        """
        parsed_url = urlparse(self.file_path)
        return bool(parsed_url.scheme) and parsed_url.netloc

    def to_dict(self):
        """
        return  object as dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "dataset_type": self.dataset_type,
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
            response = httpx.get(url, timeout=10)
            if response.status_code == 200:
                self.save_to_minio(response.content, output_file, bucket_name)
                return True
            print(f"Request failed with status code {response.status_code}")
            raise Exception("Request could not be successful due to error")
        except httpx.HTTPError as exp:
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
    def register_dataset(
        dataset_type: int, name: str, file_path: str, description: str = None
    ):
        """
        Register a dataset by uploading a file to the API using make_post_request.

        :param dataset_type: 0 (train), 1 (inference), 2 (both)
        :param name: Dataset name
        :param description: Optional dataset description
        :param file_path: Path to the dataset file
        :return: API response in JSON
        """
        PluginManager().load_config()

        url = f"{os.getenv('API_PATH')}/datasets/file"

        # Form fields for multipart/form-data
        form_data = {
            "dataset_type": str(dataset_type),
            "name": name,
            "description": description or "",
        }

        # Header with kubeflow user id
        headers = {
            "kubeflow-userid": KubeflowPlugin().get_current_user_from_namespace()
        }

        # Files dictionary; key must match FastAPI parameter name "files"
        with open(file_path, "rb") as f:
            files = {"files": (os.path.basename(file_path), f)}
            response = make_post_request(
                url=url, data=form_data, files=files, headers=headers
            )

        return response

    def save_dataset_details(self, dataset_metadata):
        """
        Save dataset details by registering the dataset via the API.

        Parameters
        ----------
        dataset_metadata : DatasetMetadata
            Instance containing name, description, file_path, and dataset_type.

        Returns
        -------
        str
            The dataset_id returned by the API.
        """
        # Verify plugin activation
        PluginManager().verify_activation(self.section)

        response = self.register_dataset(
            dataset_type=int(dataset_metadata.dataset_type),
            name=dataset_metadata.name,
            file_path=dataset_metadata.file_path,
            description=dataset_metadata.description,
        )
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

    @staticmethod
    def get_dataset(dataset_id: UUID, endpoint: str, user_id: str = None):
        """
        Generic method to call dataset API endpoints like /datasets/prometheus/{id}.

        :param dataset_id: Dataset ID to fetch
        :param endpoint: API endpoint path (e.g., "/datasets/prometheus")
        :param user_id: Optional user ID to include in the request header.
        :return: API JSON response
        """
        PluginManager().load_config()

        url = f"{os.getenv(plugin_config.API_BASEPATH)}{endpoint}"

        headers = {
            "kubeflow-userid": user_id if user_id else KubeflowPlugin().get_current_user_from_namespace()
        }

        resp = make_get_request(
            url=url,
            path_params=dataset_id,
            headers=headers,
        )

        return resp.get("data")

    def download_from_s3(self, file_path: str, file_name: str, output_file_path: str):
        """
        Download a file from S3/MinIO storage.

        :param file_path: S3 path (e.g., "s3://bucket-name/path")
        :param file_name: Name of the file to download
        :param output_file_path: Local path where file will be saved
        :return: str: Path to the downloaded file
        """
        # Parse S3 URL to extract bucket and object path
        if file_path.startswith("s3://"):
            # Remove s3:// prefix and split bucket from path
            s3_path = file_path[5:]  # Remove 's3://' prefix
            bucket_name = s3_path.split("/")[0]  # First part is bucket name
            object_prefix = "/".join(s3_path.split("/")[1:])  # Rest is object prefix

            # Construct full object name
            if object_prefix:
                object_name = f"{object_prefix.rstrip('/')}/{file_name}"
            else:
                object_name = file_name
        else:
            raise Exception(f"Invalid S3 path format: {file_path}")

        # Create MinIO client and download file
        minio_client = self.create_minio_client()

        try:
            # Download the file from S3 using MinIO client
            minio_client.fget_object(bucket_name, object_name, output_file_path)
            return output_file_path
        except Exception as e:
            raise Exception(f"Failed to download file from S3 location: {str(e)}")

    @staticmethod
    def download_dataset(dataset_id: UUID, output_file_path: str = None):
        """
        Download a dataset by its ID and save it to a specified output file.

        :param dataset_id: The ID of the dataset to download.
        :param output_file_path: The path where the downloaded dataset will be saved.
                           If None, saves to current working directory with original filename.
        :return: str: Path to the downloaded file
        """
        # Verify plugin activation
        PluginManager().verify_activation("dataset_plugin")

        PluginManager().load_config()

        # Get dataset file metadata
        download_url = f"{os.getenv('API_PATH')}/datasets/{dataset_id}/file"
        headers = {
            "kubeflow-userid": KubeflowPlugin().get_current_user_from_namespace()
        }

        # Get dataset file info with S3 location details
        dataset_response = make_get_request(download_url, headers=headers)

        if not dataset_response or "data" not in dataset_response:
            raise Exception(f"Failed to get dataset {dataset_id} information.")

        dataset_data = dataset_response["data"]

        # Check if file_name exists in response
        if "file_name" not in dataset_data or not dataset_data["file_name"]:
            raise Exception(f"File is not present for dataset {dataset_id}.")

        # Extract S3 path components
        file_path = dataset_data["file_path"]
        file_name = dataset_data["file_name"]

        # If output_file_path is not provided, use current working directory with original filename
        if output_file_path is None:
            output_file_path = os.path.join(os.getcwd(), file_name)
        else:
            # If output_file_path is a directory, join it with the file_name
            if os.path.isdir(output_file_path):
                output_file_path = os.path.join(output_file_path, file_name)

        # Create DatasetPlugin instance to call download_from_s3 method
        dataset_plugin = DatasetPlugin()
        dataset_plugin.download_from_s3(file_path, file_name, output_file_path)

        return output_file_path
