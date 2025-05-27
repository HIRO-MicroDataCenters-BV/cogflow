"""
This module provides functionality related to components used for training builder.
"""

import io
import yaml
import requests
from ..pluginmanager import PluginManager
from .dataset_plugin import DatasetPlugin


class ComponentPlugin:
    """
    A class to handle component-related operations, including parsing YAML,
    uploading to MinIO, and registering components.
    """

    def __init__(self):
        """
        Initializes the ComponentPlugin with a predefined section identifier.
        """
        self.section = "component_plugin"

    @staticmethod
    def parse_component_yaml(yaml_path):
        """
        Parses a component definition YAML file and extracts key metadata.

        Args:
            yaml_path (str): Path to the component YAML file.

        Returns:
            dict: A dictionary containing:
                - 'name' (str): Name of the component
                - 'inputs' (list): List of input parameters (optional)
                - 'outputs' (list): List of output parameters (optional)
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        name = data.get("name")
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        return {"name": name, "inputs": inputs, "outputs": outputs}

    def save_yaml_to_minio(self, yaml_path, bucket_name, object_name=None):
        """
        Uploads a component YAML file to MinIO storage and returns a public URL.

        Args:
            yaml_path (str): Local path to the YAML file.
            bucket_name (str): Target MinIO bucket name.
            object_name (str, optional): Override for object name in bucket.
                                         Defaults to the component name with `.yaml`.

        Returns:
            tuple:
                - url (str): Presigned URL to access the uploaded YAML.
                - object_name (str): Final object name in MinIO.
        """
        PluginManager().verify_activation(self.section)
        minio_client = DatasetPlugin().create_minio_client()
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        if not object_name:
            parsed = self.parse_component_yaml(yaml_path)
            object_name = f"{parsed['name'].replace(' ', '_')}.yaml"
        with open(yaml_path, "rb") as f:
            content = f.read()
        minio_client.put_object(
            bucket_name,
            object_name,
            io.BytesIO(content),
            len(content),
            content_type="application/x-yaml",
        )
        url = minio_client.presigned_get_object(bucket_name, object_name)
        return url, object_name

    def register_component(self, yaml_path, bucket_name, api_base_url, api_key=None):
        """
        Registers a component by uploading its YAML definition to MinIO and
        posting its metadata to a registry API.

        Args:
            yaml_path (str): Path to the component YAML file.
            bucket_name (str): MinIO bucket to upload the YAML.
            api_base_url (str): Base URL of the component registration API.
            api_key (str, optional): Bearer token for authorization. Defaults to None.

        Returns:
            dict: JSON response from the registration API.

        Raises:
            requests.HTTPError: If the API returns an error status.
        """
        # Parse YAML
        parsed = self.parse_component_yaml(yaml_path)
        # Upload YAML to MinIO
        minio_url, object_name = self.save_yaml_to_minio(yaml_path, bucket_name)
        # Prepare data for API
        data = {
            "name": parsed["name"],
            "input_path": minio_url,
            "output_path": object_name,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        components = PluginManager().load_path("components")
        url = f"{api_base_url}{components}"
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
