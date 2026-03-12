"""
This module provides functionality related to components used for training builder.
"""

import os
import io
from urllib.parse import urlparse
from uuid import UUID

import yaml
import httpx
from minio import S3Error
from .. import plugin_config

from ..util import make_get_request

from ..pluginmanager import PluginManager
from .dataset_plugin import DatasetPlugin
from .kubeflowplugin import KubeflowPlugin


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
    def parse_component_yaml(yaml_path: str = None, yaml_data: str = None):
        """
        Parses a component definition YAML file and extracts key metadata.

        Args:
            yaml_path (str, Optional): Path to the component YAML file.
            yaml_data (str, Optional): YAML content as a string.

        Returns:
            dict: A dictionary containing:
                - 'name' (str): Name of the component
                - 'inputs' (list): List of input parameters (optional)
                - 'outputs' (list): List of output parameters (optional)
        """
        if yaml_data:
            data = yaml.safe_load(yaml_data)
        elif yaml_path:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Either 'yaml_path' or 'yaml_data' must be provided.")

        name = data.get("name")
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        return {"name": name, "inputs": inputs, "outputs": outputs}

    def save_yaml_to_minio(
        self,
        bucket_name: str,
        category: str = None,
        object_name: str = None,
        yaml_path: str = None,
        file_obj: io.BytesIO = None,
        overwrite: bool = True,
    ):
        """
        Uploads YAML (from path or memory) to MinIO and returns its s3-style URL.

        Args:
            bucket_name (str): Target MinIO bucket name.
            category (str, optional): Category / logical namespace.
            object_name (str, optional): Desired object name in MinIO. If not provided, derived from YAML 'name'.
            yaml_path (str, optional): Local path to the YAML file.
            file_obj (io.BytesIO, optional): In-memory bytes buffer of the YAML file.
            overwrite (bool): If False, will raise an error if object already exists.

        Returns:
            tuple: (s3_url, object_name)
        """
        PluginManager().verify_activation(self.section)
        minio_client = DatasetPlugin().create_minio_client()

        # --- Ensure bucket exists ---
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # --- Derive object name if needed ---
        if not object_name:
            if yaml_path:
                parsed = self.parse_component_yaml(yaml_path=yaml_path)
            elif file_obj:
                yaml_str = file_obj.getvalue().decode("utf-8")
                parsed = self.parse_component_yaml(yaml_data=yaml_str)
            else:
                raise ValueError("Either yaml_path or file_obj must be provided.")
            object_name = f"{parsed['name'].replace(' ', '_')}.yaml"

        # --- Check for existing object (overwrite safeguard) ---
        try:
            minio_client.stat_object(bucket_name, object_name)
            if not overwrite:
                raise FileExistsError(
                    f"Component '{object_name}' already exists in bucket '{bucket_name}'. "
                    "Set overwrite=True to replace it."
                )
        except S3Error as e:
            # Only ignore if object doesn't exist
            if e.code not in ("NoSuchKey", "NoSuchObject"):
                raise

        # --- Prepare bytes ---
        if file_obj:
            data_bytes = file_obj.getvalue()
        elif yaml_path:
            with open(yaml_path, "rb") as f:
                data_bytes = f.read()
        else:
            raise ValueError("Either yaml_path or file_obj must be provided to upload.")

        # --- Upload to MinIO ---
        try:
            minio_client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=io.BytesIO(data_bytes),
                length=len(data_bytes),
                content_type="application/x-yaml",
            )
        except Exception as ex:
            raise RuntimeError(f"Failed to upload {object_name} to MinIO: {ex}") from ex

        # --- Build s3:// URL ---
        if category:
            url = f"s3://{category}/{bucket_name}/{object_name}"
        else:
            url = f"s3://{bucket_name}/{object_name}"

        return url, object_name

    def register_component(
        self,
        name: str = None,
        yaml_path: str = None,
        yaml_data: str = None,
        bucket_name: str = None,
        category: str = None,
        creator: str = None,
        overwrite: bool = False,
    ):
        """
        Registers or updates a component.

        Behavior:
        - If component doesn't exist → POST (create)
        - If component exists:
            • overwrite=True  → PATCH (update)
            • overwrite=False → raise error
        """
        PluginManager().load_config()

        if not (yaml_path or yaml_data):
            raise ValueError("Either 'yaml_path' or 'yaml_data' must be provided.")

        # --- Parse YAML metadata ---
        parsed = self.parse_component_yaml(yaml_path=yaml_path, yaml_data=yaml_data)
        component_name = name if name else parsed["name"]

        # --- Prepare YAML content ---
        data_bytes = (
            yaml_data.encode("utf-8") if yaml_data else open(yaml_path, "rb").read()
        )

        # --- Upload YAML to MinIO ---
        minio_url, object_name = self.save_yaml_to_minio(
            bucket_name=bucket_name,
            category=category,
            object_name=f"{component_name.replace(' ', '_')}.yaml",
            file_obj=io.BytesIO(data_bytes),
            overwrite=True,  # file always replaced if same name
        )

        # --- Prepare payload for registry ---
        payload = {
            "name": component_name,
            "input_path": parsed["inputs"],
            "output_path": parsed["outputs"],
            "component_file": minio_url,
            "category": category,
        }

        headers = {"Content-Type": "application/json"}
        base_url = os.getenv("API_PATH")
        components_path = PluginManager().load_path("components")
        base_endpoint = f"{base_url}{components_path}"

        # --- Step 1: Check if component exists ---
        check_url = f"{base_endpoint}?name={component_name}"
        try:
            check_resp = httpx.get(check_url, timeout=10)
            check_resp.raise_for_status()
            existing = check_resp.json().get("data", [])
        except httpx.HTTPStatusError as ex:
            if (
                hasattr(ex, "response")
                and ex.response is not None
                and ex.response.status_code == 404
            ):
                existing = []
            else:
                raise RuntimeError(f"Failed to check existing component: {ex}")

        # --- Step 2: Branch logic ---
        if existing:
            existing_component = existing[0]
            component_id = existing_component.get("id")

            if not overwrite:
                raise ValueError(
                    f"Component '{component_name}' already exists. "
                    f"Use overwrite=True to update."
                )

            # 🔁 PATCH (update)
            update_url = f"{base_endpoint}?creator={creator}"
            patch_resp = httpx.patch(
                update_url, json=payload, headers=headers, timeout=10
            )
            patch_resp.raise_for_status()
            print("Updated existing component successfully.")
            return patch_resp.json().get("data", {})

        else:
            # Attach creator if provided
            post_url = f"{base_url}{components_path}"
            if creator:  # 👈 append creator only in POST case
                post_url += f"?creator={creator}"
            # 🆕 POST (create)
            post_resp = httpx.post(
                post_url, json=payload, headers=headers, timeout=10
            )
            post_resp.raise_for_status()
            print("Registered new component successfully.")
            return post_resp.json().get("data", {})

    @staticmethod
    def download_yaml_from_minio(bucket_name, object_name, local_path):
        """
        Downloads a YAML file from MinIO storage to a local path.

        Args:
            bucket_name (str): MinIO bucket name.
            object_name (str): Object name in the bucket.
            local_path (str): Local file path to save the downloaded file.

        Raises:
            RuntimeError: If the download fails.
        """
        minio_client = DatasetPlugin().create_minio_client()
        try:
            minio_client.fget_object(bucket_name, object_name, local_path)
        except Exception as ex:
            raise RuntimeError(
                f"Failed to download {object_name} from MinIO: {ex}"
            ) from ex

    @staticmethod
    def load_component_from_id(component_id: UUID):
        """
        Fetches component metadata by ID via internal API, loads its YAML from MinIO,
        and returns a Kubeflow component object.

        Supports both styles:
            s3://{category}/{bucket}/{object_name}
            s3://{bucket}/{object_name}

        Args:
            component_id (UUID): Unique component ID.

        Returns:
            kfp.components.Component: Loaded Kubeflow component.

        Raises:
            RuntimeError: If API or MinIO fetch fails.
            ValueError: If component_file format is invalid.
        """

        PluginManager().load_config()

        # --- Step 1: Fetch metadata from internal API ---
        try:
            url = f"{os.getenv('API_PATH')}{plugin_config.TRAINING_BUILDER_COMPONENTS}/{component_id}"
            resp = make_get_request(
                url,
                timeout=10,
            )
            data = resp.json() if hasattr(resp, "json") else resp
            metadata = (
                data.get("data") if isinstance(data, dict) and "data" in data else data
            )
        except Exception as ex:
            raise RuntimeError(
                f"Failed to fetch component metadata for ID {component_id}: {ex}"
            ) from ex

        # --- Step 2: Validate and parse component_file (must be s3:// style) ---
        component_file = metadata.get("component_file")
        if not component_file or not component_file.startswith("s3://"):
            raise ValueError(f"Invalid or unsupported component_file: {component_file}")

        # --- Step 3: Parse s3://category/bucket/object_name or s3://bucket/object_name ---
        parsed = urlparse(component_file)
        netloc = parsed.netloc.strip().replace(" ", "_")
        path_parts = parsed.path.strip("/").split("/")

        # Possible structures:
        #   s3://category/bucket/object_name   → len(path_parts) == 2
        #   s3://bucket/object_name            → len(path_parts) == 1

        if len(path_parts) == 2:
            # category/bucket/object_name → category=netloc
            category = netloc
            bucket_name, object_name = path_parts
        elif len(path_parts) == 1:
            # bucket/object_name → no category
            category = None
            bucket_name = netloc
            object_name = path_parts[0]
        else:
            raise ValueError(
                f"Malformed S3 path. Expected s3://<category>/<bucket>/<object> "
                f"or s3://<bucket>/<object>, got: {component_file}"
            )

        bucket_name = bucket_name.replace(" ", "_")
        object_name = object_name.replace(" ", "_")

        # print(
        #     f"Resolving MinIO path → category='{category}', bucket='{bucket_name}', object='{object_name}'"
        # )

        # --- Step 4: Fetch YAML from MinIO ---
        minio_client = DatasetPlugin().create_minio_client()
        response = None
        yaml_content = None
        try:
            response = minio_client.get_object(bucket_name, object_name)
            yaml_content = response.read().decode("utf-8")
        except Exception as ex:
            raise RuntimeError(
                f"Failed to read object '{object_name}' from MinIO bucket '{bucket_name}': {ex}"
            ) from ex
        finally:
            if response is not None:
                try:
                    response.close()
                    response.release_conn()
                except Exception:
                    pass

        # --- Step 5: Load into Kubeflow component ---
        if yaml_content is None:
            raise RuntimeError(f"YAML content could not be loaded for '{object_name}'")
        try:
            full_path = (
                f"s3://{category}/{bucket_name}/{object_name}"
                if category
                else f"s3://{bucket_name}/{object_name}"
            )
            print(f"Loading component from {full_path}")
            return KubeflowPlugin().load_component_from_text(text=yaml_content)
        except Exception as ex:
            raise RuntimeError(
                f"Failed to parse KFP component YAML for '{object_name}': {ex}"
            ) from ex
