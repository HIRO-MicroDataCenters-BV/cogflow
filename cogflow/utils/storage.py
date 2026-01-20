"""
Utility functions for MinIO storage interactions.
"""

from urllib.parse import urlparse
from minio import Minio

from ..config import config


def minio_client() -> Minio:
    """Return MinIO client using config vars, stripping protocol if needed."""

    raw = config.MLFLOW_S3_ENDPOINT_URL
    if not raw:
        raise RuntimeError("MLFLOW_S3_ENDPOINT_URL is not set")

    parsed = urlparse(raw)

    # If protocol included → extract host:port
    if parsed.scheme:
        endpoint = f"{parsed.hostname}:{parsed.port}"
        secure = parsed.scheme == "https"
    else:
        # Raw host:port
        endpoint = raw
        secure = config.MINIO_SECURE

    return Minio(
        endpoint=endpoint,
        access_key=config.AWS_ACCESS_KEY_ID,
        secret_key=config.AWS_SECRET_ACCESS_KEY,
        secure=secure,
    )
