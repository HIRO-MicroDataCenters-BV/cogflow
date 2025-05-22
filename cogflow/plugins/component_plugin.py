import os
import io
import yaml
from minio import Minio
from .. import plugin_config
from ..pluginmanager import PluginManager
import requests


class ComponentPlugin:
    """
    A class to handle component-related operations, including parsing YAML,
    uploading to MinIO, and registering components.
    """

    def __init__(self):
        self.minio_endpoint = os.getenv(plugin_config.MINIO_ENDPOINT_URL)
        if self.minio_endpoint and self.minio_endpoint.startswith(("http://", "https://")):
            protocol_end_index = self.minio_endpoint.find("//") + 2
            self.minio_endpoint = self.minio_endpoint[protocol_end_index:]
        self.minio_access_key = os.getenv(plugin_config.MINIO_ACCESS_KEY)
        self.minio_secret_key = os.getenv(plugin_config.MINIO_SECRET_ACCESS_KEY)
        self.section = "component_plugin"

    def create_minio_client(self):
        PluginManager().verify_activation(self.section)
        return Minio(
            self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            secure=False,
        )

    def parse_component_yaml(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        name = data.get("name")
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        return {"name": name, "inputs": inputs, "outputs": outputs}

    def save_yaml_to_minio(self, yaml_path, bucket_name, object_name=None):
        PluginManager().verify_activation(self.section)
        minio_client = self.create_minio_client()
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
            content_type="application/x-yaml"
        )
        url = minio_client.presigned_get_object(bucket_name, object_name)
        return url, object_name

    def register_component(self, yaml_path, bucket_name, api_base_url, api_key=None):
        # Parse YAML
        parsed = self.parse_component_yaml(yaml_path)
        # Upload YAML to MinIO
        minio_url, object_name = self.save_yaml_to_minio(yaml_path, bucket_name)
        # Prepare data for API
        data = {
            "name": parsed["name"],
            "input_path": minio_url,
            "output_path": object_name
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{api_base_url}/components"
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()