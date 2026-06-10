"""
cogflow.utils.common
--------------------

A collection of general-purpose utility functions used across CogFlow modules.

This includes:
    - Serialization helpers
    - UUID format converters
    - URI validators
    - Namespace and ownership retrieval for Kubeflow environments

These utilities are intentionally lightweight and free of heavy dependencies
to remain import-safe across all submodules.
"""

import os
import re
from datetime import datetime
from typing import Any
from uuid import UUID

from kubernetes import client, config

from .logging import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------------
# 🔹 Serialization Utilities
# -------------------------------------------------------------------------
def custom_serializer(obj: Any) -> str:
    """
    Serialize unsupported objects (like datetime) into a JSON-safe format.

    Args:
        obj (Any): Object to serialize.

    Returns:
        str: Serialized string representation.

    Raises:
        TypeError: If the object type is unsupported.

    Example:
        >>> json.dumps({"time": datetime.utcnow()}, default=custom_serializer)
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def serialize_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    """
    Convert artifacts into a JSON-serializable format for API transmission.

    Args:
        artifacts (dict): Mapping of artifact names to artifact objects.

    Returns:
        dict: Dictionary of artifact URIs or string representations.

    Example:
        >>> serialize_artifacts({"roc_curve": Artifact(uri="s3://...")})
        {'validation_artifacts': {'roc_curve': 's3://...'}}
    """
    serialized_artifacts = {}
    for key, artifact in artifacts.items():
        if hasattr(artifact, "uri"):
            serialized_artifacts[key] = artifact.uri
        else:
            serialized_artifacts[key] = str(artifact)
    return {"validation_artifacts": serialized_artifacts}


# -------------------------------------------------------------------------
# 🔹 Validation Utilities
# -------------------------------------------------------------------------
def is_valid_s3_uri(uri: str) -> bool:
    """
    Validate if a string is a proper S3 URI (e.g., s3://bucket/key).

    Args:
        uri (str): The URI string to validate.

    Returns:
        bool: True if valid S3 URI, False otherwise.

    Example:
        >>> is_valid_s3_uri("s3://my-bucket/model.pkl")
        True
    """
    s3_uri_regex = re.compile(r"^s3://([a-z0-9.-]+)/(.*)$")
    match = s3_uri_regex.match(uri)
    valid = bool(match and match.group(1) and match.group(2))
    logger.debug("Validating S3 URI '%s': %s", uri, valid)
    return valid


# -------------------------------------------------------------------------
# 🔹 UUID Utilities
# -------------------------------------------------------------------------
UUID_COMPACT_RE = re.compile(r"^[0-9a-fA-F]{32}$")
UUID_HYPHEN_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def normalize_uuid(value) -> str:
    """
    Normalize UUID to canonical hyphenated form.

    Accepts:
        - UUID object
        - hyphenated UUID string
        - 32-char compact hex UUID string

    Returns:
        str: canonical hyphenated UUID string

    Raises:
        ValueError: if not a valid UUID
    Example:
        >>> normalize_uuid("7a1f6cf81d7e4d40b9a91c94ce6c3c0a")
        '7a1f6cf8-1d7e-4d40-b9a9-1c94ce6c3c0a'
        >>> normalize_uuid("7a1f6cf8-1d7e-4d40-b9a9-1c94ce6c3c0a")
        '7a1f6cf8-1d7e-4d40-b9a9-1c94ce6c3c0a'
    """

    if isinstance(value, UUID):
        return str(value)

    if not isinstance(value, str):
        raise ValueError(f"Invalid UUID type: {type(value)}")

    value = value.strip()

    # Case 1 — 32-char compact UUID → parse & return canonical
    if UUID_COMPACT_RE.match(value):
        return str(UUID(hex=value))

    # Case 2 — hyphenated UUID
    if UUID_HYPHEN_RE.match(value):
        return str(UUID(value))

    # Final fallback — try parsing anyway
    try:
        return str(UUID(value))
    except Exception as exc:
        raise ValueError(f"Invalid UUID value: {value}") from exc


def uuid_to_hex(value: str) -> str:
    """
    Convert a UUID to non-hyphenated (hex) format.

    Args:
        value (str): Canonical UUID string.

    Returns:
        str: Hex (non-hyphenated) UUID string.

    Raises:
        ValueError: If the input is invalid.

    Example:
        >>> uuid_to_hex("7a1f6cf8-1d7e-4d40-b9a9-1c94ce6c3c0a")
        '7a1f6cf81d7e4d40b9a91c94ce6c3c0a'
    """
    try:
        hex_value = UUID(value).hex
        logger.debug("Converted UUID to hex: %s", hex_value)
        return hex_value
    except (ValueError, AttributeError, TypeError) as exc:
        logger.error("Invalid UUID value: %s", value)
        raise ValueError(f"Invalid UUID value: {value!r}") from exc


# -------------------------------------------------------------------------
# 🔹 Kubernetes / Kubeflow Utilities
# -------------------------------------------------------------------------
def get_namespace() -> str:
    """
    Retrieve the default Kubernetes namespace based on the active configuration.

    This function first loads Kubernetes configuration using `load_k8s_config()`.
    Then, it attempts to read the namespace from:
        1. The in-cluster service account file (if running in a Pod)
        2. The current kubeconfig context (if running locally)
        3. Defaults to "default" as a fallback

    Returns:
        str: The default namespace.

    Example:
        >>> get_namespace()
        'abc-namespace'
    """
    try:
        # 1️⃣ Ensure Kubernetes configuration is loaded
        load_k8s_config()

        # 2️⃣ Try reading the in-cluster namespace file
        try:
            with open(
                "/var/run/secrets/kubernetes.io/serviceaccount/namespace",
                encoding="utf-8",
            ) as f:
                namespace = f.read().strip()
                logger.debug("Resolved in-cluster namespace: %s", namespace)
                return namespace
        except FileNotFoundError:
            pass  # Not running inside a pod

        # 3️⃣ Fallback: use the namespace from kubeconfig
        _, current_context = config.list_kube_config_contexts()
        namespace = current_context["context"].get("namespace", "default")
        logger.debug("Resolved kubeconfig namespace: %s", namespace)
        return namespace

    except Exception as e:
        logger.warning("Could not determine namespace, defaulting to 'default': %s", e)
        return "default"


def load_k8s_config() -> None:
    """
    Load the Kubernetes configuration.

    Tries to load the in-cluster configuration if running inside a pod.
    Falls back to local kubeconfig for external environments.

    Raises:
        ConfigException: If configuration could not be loaded.
    Example:
        >>> load_k8s_config()
    """
    try:
        config.load_incluster_config()
        logger.debug("Loaded in-cluster Kubernetes configuration.")
    except config.config_exception.ConfigException:
        try:
            config.load_kube_config()
            logger.debug("Loaded local kubeconfig file.")
        except config.config_exception.ConfigException:
            logger.error("Failed to load Kubernetes configuration.")
            raise


def get_current_user() -> str:
    """
    Fetch the current Kubeflow user ID by reading the owner annotation
    from the user's namespace.

    Returns:
        str: The user ID of the notebook owner.

    Raises:
        RuntimeError: If the owner annotation is not found.
    Example:
        >>> get_current_user()
        'user@email.com'
    """

    try:
        namespace_name = get_namespace()

        v1 = client.CoreV1Api()
        ns_obj = v1.read_namespace(name=namespace_name)

        annotations = ns_obj.metadata.annotations or {}
        owner = annotations.get("owner")

        if not owner:
            raise RuntimeError(f"No owner annotation found in namespace: {namespace_name}")

        logger.debug("Resolved namespace owner: %s", owner)
        return owner

    except Exception as e:
        logger.error("Failed to fetch user from namespace: %s", e)
        raise RuntimeError("Unable to resolve current user ID") from e


def file_exists(path: str) -> bool:
    """
    Check whether a file exists and is a regular file.

    Args:
        path (str): File path

    Returns:
        bool: True if file exists and is a file, else False
    """
    if not path:
        return False

    return os.path.isfile(path)


def get_filename(path: str) -> str:
    """
    Extract filename from a file path.

    Args:
        path (str): File path

    Returns:
        str: Filename
    """
    if not path:
        return ""

    return os.path.basename(path)


def cwd() -> str:
    """
    Return the current working directory.

    Returns:
        str: Absolute path of current working directory.
    """
    return os.getcwd()


def is_dir(path: str) -> bool:
    """
    Check whether a path exists and is a directory.

    Args:
        path (str): Path to check

    Returns:
        bool: True if path is a directory, else False
    """
    if not path:
        return False

    return os.path.isdir(path)


def join_path(*parts: Any) -> str:
    """
    Join path components safely.

    Args:
        *parts: Path components

    Returns:
        str: Joined path
    """
    return os.path.join(*(str(p) for p in parts if p is not None))
