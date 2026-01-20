"""
CogFlow Pipeline Components
---------------------------

Handles:
- Parsing component YAML
- Uploading component YAML to MinIO
- Registering components in the registry
- Loading components back into KFP
- The @cogcomponent decorator

Depends ONLY on:
    utils.storage
    utils.common
    utils.exceptions
    utils.network
    pipelines.orchestration (safe 1-way import)
"""

from __future__ import annotations

import io
from typing import Mapping, List, Optional
from uuid import UUID
from urllib.parse import urlparse

import yaml

from ...utils.storage import minio_client
from ...utils import common
from ...utils.network import (
    make_get_request,
    make_post_request,
    make_patch_request,
    make_get_request_raw,
)
from ...utils.exceptions import (
    CogflowErrorHandler,
    CogflowComponentError,
    CogflowComponentValidationError,
    CogflowComponentRegistryError,
    CogflowComponentStorageError,
)
from ...utils.logging import get_logger
from ...config import config

logger = get_logger(__name__)


def _orc():
    """Lazy loader for orchestration functions to avoid heavy imports."""
    from . import orchestration

    return orchestration


# ============================================================================


def _parse_s3_uri(uri: str):
    """Parse s3:// URI into (category, bucket, object_name)."""
    if not uri or not uri.startswith("s3://"):
        raise CogflowComponentValidationError(
            f"Invalid s3 path '{uri}'. Must begin with s3://."
        )

    parsed = urlparse(uri)
    netloc = parsed.netloc.strip()
    parts = parsed.path.strip("/").split("/")

    if len(parts) == 2:
        return netloc, parts[0], parts[1]  # category/bucket/object
    if len(parts) == 1:
        return None, netloc, parts[0]  # bucket/object

    raise CogflowComponentValidationError(f"Malformed s3 path '{uri}'.")


# ============================================================================


def parse_component_yaml(yaml_path: str = None, yaml_data: str = None) -> dict:
    """Load YAML and extract name, inputs, outputs."""
    try:
        if yaml_data:
            content = yaml.safe_load(yaml_data)
        elif yaml_path:
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        else:
            raise CogflowComponentValidationError(
                "Either yaml_path or yaml_data must be provided."
            )
    except Exception as exc:
        raise CogflowComponentValidationError("Failed parsing component YAML.") from exc

    return {
        "name": content.get("name"),
        "inputs": content.get("inputs", []),
        "outputs": content.get("outputs", []),
    }


# ============================================================================


def _upload_yaml_to_minio(
    *, bucket_name: str, object_name: str, data_bytes: bytes, overwrite: bool = True
) -> str:
    """Upload YAML bytes to MinIO and return s3://bucket/object."""
    client = minio_client()

    # Ensure bucket exists
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
    except Exception as exc:
        raise CogflowComponentStorageError(
            f"Failed accessing bucket '{bucket_name}'."
        ) from exc

    # Check existing
    try:
        client.stat_object(bucket_name, object_name)
        if not overwrite:
            raise CogflowComponentValidationError(
                f"Object '{object_name}' already exists (overwrite=False)."
            )
    except Exception:
        pass  # OK if not exists

    # Upload
    try:
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(data_bytes),
            length=len(data_bytes),
            content_type="application/x-yaml",
        )
    except Exception as exc:
        raise CogflowComponentStorageError(
            f"Failed uploading '{object_name}'."
        ) from exc

    return f"s3://{bucket_name}/{object_name}"


# ============================================================================


def register_component(
    *,
    name: str = None,
    yaml_path: str = None,
    yaml_data: str = None,
    bucket_name: str = None,
    category: str = None,
    creator: str = None,
    overwrite: bool = False,
):
    """Upload YAML to MinIO + register OR update in registry."""
    creator = creator or common.get_current_user()
    bucket_name = bucket_name or config.COMPONENTS_BUCKET_NAME
    time_out = config.TIME_OUT

    parsed = parse_component_yaml(yaml_path=yaml_path, yaml_data=yaml_data)
    component_name = name or parsed["name"]
    if not component_name:
        raise CogflowComponentValidationError("Component name missing in YAML.")

    # Load YAML bytes
    try:
        if yaml_data:
            data_bytes = yaml_data.encode("utf-8")
        else:
            with open(yaml_path, "rb") as f:
                data_bytes = f.read()
    except Exception as exc:
        raise CogflowComponentValidationError("Failed reading YAML.") from exc

    object_name = f"{component_name.replace(' ', '_')}.yaml"

    s3_url = _upload_yaml_to_minio(
        bucket_name=bucket_name,
        object_name=object_name,
        data_bytes=data_bytes,
        overwrite=True,
    )

    base_api = config.API_PATH.rstrip("/")
    comp = config.COMPONENTS.lstrip("/")
    endpoint = f"{base_api}/{comp}"

    payload = {
        "name": component_name,
        "input_path": parsed["inputs"],
        "output_path": parsed["outputs"],
        "component_file": s3_url,
        "category": category,
    }
    headers = {"Content-Type": "application/json"}

    # Check if exists
    check_url = f"{endpoint}?name={component_name}"
    try:
        resp = make_get_request_raw(check_url, timeout=time_out)
        if resp is None:
            existing = []
        existing = resp.get("data", [])
    except Exception as ex:
        raise CogflowComponentRegistryError(
            f"Failed checking registry for '{component_name}'"
        )

    # --- Update existing component ---
    if existing:
        # Stop immediately if overwrite is not allowed
        if not overwrite:
            raise CogflowComponentValidationError(
                f"Component '{component_name}' already exists; overwrite=False."
            )

        # Otherwise continue with patch request
        try:
            url = f"{endpoint}?creator={creator}" if creator else endpoint
            resp = make_patch_request(
                url, data=payload, headers=headers, timeout=time_out
            )
            return resp.get("data")
        except Exception as exc:
            raise CogflowComponentRegistryError(
                f"Failed updating component '{component_name}'"
            ) from exc

    # Create
    try:
        url = f"{endpoint}?creator={creator}" if creator else endpoint
        resp = make_post_request(url, data=payload, headers=headers, timeout=time_out)
        return resp.get("data")
    except Exception as exc:
        raise CogflowComponentRegistryError(
            f"Failed creating component '{component_name}'"
        ) from exc


# ============================================================================


def load_component_from_id(component_id: UUID):
    """
    Load a component from registry + MinIO and return a KFP component
    with env vars injected.
    """
    base_api = config.API_PATH.rstrip("/")
    comp = config.COMPONENTS.lstrip("/")
    url = f"{base_api}/{comp}/{component_id}"

    # Fetch metadata
    try:
        resp = make_get_request(url, timeout=10)
        data = resp.json() if hasattr(resp, "json") else resp
        metadata = (
            data.get("data") if isinstance(data, dict) and "data" in data else data
        )
    except Exception as exc:
        logger.exception("Failed fetching metadata for component %s", component_id)
        raise CogflowComponentRegistryError(
            f"Failed fetching metadata for component '{component_id}'"
        ) from exc

    if not metadata:
        raise CogflowComponentRegistryError(
            f"Component '{component_id}' not found in registry."
        )

    component_file = metadata.get("component_file")
    if not component_file:
        raise CogflowComponentValidationError(
            f"Component '{component_id}' missing component_file."
        )

    _, bucket, object_name = _parse_s3_uri(component_file)

    # Download YAML from MinIO
    client = minio_client()
    try:
        obj = client.get_object(bucket, object_name)
        yaml_content = obj.read().decode("utf-8")
    except Exception as exc:
        raise CogflowComponentStorageError(
            f"Failed reading MinIO object '{object_name}'."
        ) from exc
    finally:
        try:
            obj.close()
            obj.release_conn()
        except Exception:
            pass

    # Convert YAML → KFP component object
    try:
        component = _orc().load_component_from_text(yaml_content)
    except Exception as exc:
        raise CogflowComponentError(
            f"Failed parsing component YAML '{object_name}'."
        ) from exc

    # Inject env vars (runtime enhancement)
    return _orc()._inject_env_into_container_op(component)


# ============================================================================


def cogcomponent(
    *,
    output_component_file: Optional[str] = None,
    base_image: str = config.COMP_BASE_IMAGE,
    packages_to_install: Optional[List[str]] = None,
    annotations: Optional[Mapping[str, str]] = None,
    name: Optional[str] = None,
    category: Optional[str] = None,
    register: bool = False,
    bucket_name: Optional[str] = None,
    overwrite: bool = False,
):
    """Decorator to convert Python function into KFP component + optional registry."""

    def decorator(func):
        component_op = _orc().create_component_from_func(
            func=func,
            output_component_file=output_component_file,
            base_image=base_image,
            packages_to_install=packages_to_install,
            annotations=annotations,
        )

        if register:
            try:
                yaml_data = yaml.safe_dump(
                    component_op.component_spec.to_dict(),
                    sort_keys=False,
                )
                resolved = name or component_op.component_spec.name

                register_component(
                    name=resolved,
                    yaml_data=yaml_data,
                    bucket_name=bucket_name,
                    category=category,
                    overwrite=overwrite,
                )
            except CogflowComponentValidationError:
                # rethrow exactly as-is, do NOT wrap
                raise

            except Exception as exc:
                # wrap everything else into registry error
                CogflowErrorHandler.handle_exception(
                    exc,
                    context="Automatic component registration",
                    raise_as=CogflowComponentRegistryError,
                    re_raise=True,
                )

        return component_op

    return decorator


# ============================================================================
# DOWNLOAD YAML FROM MINIO → LOCAL PATH
# ============================================================================


def download_yaml_from_minio(bucket_name: str, object_name: str, local_path: str):
    """
    Download a YAML file from MinIO to a local file.

    Args:
        bucket_name (str): MinIO bucket name.
        object_name (str): Object key inside the bucket.
        local_path (str): Local file path to save the content.

    Raises:
        CogflowComponentStorageError: If download fails.
    """
    logger.info(
        "Downloading YAML from MinIO (bucket=%s, object=%s) → %s",
        bucket_name,
        object_name,
        local_path,
    )

    client = minio_client()

    try:
        client.fget_object(bucket_name, object_name, local_path)
        logger.debug(
            "YAML download completed from bucket=%s object=%s → %s",
            bucket_name,
            object_name,
            local_path,
        )
    except Exception as exc:
        raise CogflowComponentStorageError(
            f"Failed to download '{object_name}' from bucket '{bucket_name}'."
        ) from exc
