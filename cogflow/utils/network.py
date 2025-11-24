"""
CogFlow Network Utilities

Provides standardized helpers for making HTTP/HTTPS API requests,
validating URIs, handling UUID conversions, and serializing datetime objects.
"""

from typing import List, Optional, Union
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .logging import get_logger

logger = get_logger(__name__)

# Default request timeout (seconds)
DEFAULT_TIMEOUT = 15


# ---------------------------------------------------------------------
# CORE HTTP/HTTPS REQUEST HELPERS
# ---------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def make_post_request(
    url: str,
    data: Optional[dict] = None,
    params: Optional[dict] = None,
    files: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Make a POST request with optional payloads and file upload support.
    Retries on transient network errors.
    """
    try:
        if files:
            response = requests.post(
                url,
                data=data,
                files=files,
                headers=headers,
                params=params,
                timeout=timeout,
            )
        elif data:
            response = requests.post(
                url, json=data, params=params, headers=headers, timeout=timeout
            )
        else:
            response = requests.post(
                url, params=params, headers=headers, timeout=timeout
            )

        if response.ok:
            logger.info("POST %s succeeded with status %s", url, response.status_code)
            return response.json()

        logger.warning(
            "POST %s failed: %s - %s", url, response.status_code, response.text[:200]
        )
        response.raise_for_status()

    except requests.RequestException as exp:
        logger.exception("Error making POST request to %s: %s", url, exp)
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def make_get_request(
    url: str,
    path_params: Optional[str] = None,
    query_params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    paginate: bool = False,
) -> Union[dict, List[dict]]:
    """
    Make a GET request (supports optional pagination).
    """
    try:
        full_url = (
            f"{url.rstrip('/')}/{str(path_params).lstrip('/')}" if path_params else url
        )

        if not paginate:
            response = requests.get(
                full_url, params=query_params, headers=headers, timeout=timeout
            )
            if response.ok:
                logger.info(
                    "GET %s succeeded with status %s", full_url, response.status_code
                )
                return response.json()

            logger.warning(
                "GET %s failed: %s - %s",
                full_url,
                response.status_code,
                response.text[:200],
            )
            response.raise_for_status()

        # Pagination mode
        all_data, page = [], 1
        limit = (query_params or {}).get("limit", 10)

        while True:
            page_params = dict(query_params or {})
            page_params.update({"page": page, "limit": limit})

            response = requests.get(
                full_url, params=page_params, headers=headers, timeout=timeout
            )
            if not response.ok:
                logger.warning("GET pagination failed on page %s", page)
                break

            payload = response.json()
            data = payload.get("data", [])
            all_data.extend(data)

            pagination = payload.get("pagination", {})
            total = pagination.get("total_items", len(data))

            if len(all_data) >= total:
                break
            page += 1

        logger.info("GET %s completed with %s total items.", full_url, len(all_data))
        return all_data

    except requests.RequestException as exp:
        logger.exception("Error making GET request to %s: %s", url, exp)
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def make_delete_request(
    url: str,
    path_params: Optional[str] = None,
    query_params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Make a DELETE request with retries.
    """
    try:
        full_url = f"{url.rstrip('/')}/{path_params}" if path_params else url
        response = requests.delete(
            full_url, params=query_params, headers=headers, timeout=timeout
        )

        if response.ok:
            logger.info(
                "DELETE %s succeeded with status %s", full_url, response.status_code
            )
            return response.json()

        logger.warning(
            "DELETE %s failed: %s - %s",
            full_url,
            response.status_code,
            response.text[:200],
        )

        response.raise_for_status()

    except requests.RequestException as exp:
        logger.exception("Error making DELETE request to %s: %s", url, exp)
        raise


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------


def make_health_check_request(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    headers: Optional[dict] = None,
) -> bool:
    """
    Lightweight GET request used only for health checks.

    - Does NOT expect JSON.
    - Treats ANY 2xx response as success.
    - Does NOT retry aggressively (configurable if needed).
    """

    try:
        response = requests.get(url, headers=headers, timeout=timeout)

        if 200 <= response.status_code < 300:
            logger.info("Health check to %s succeeded (%s).", url, response.status_code)
            return True

        logger.warning(
            "Health check to %s failed (%s): %s",
            url,
            response.status_code,
            response.text[:200],
        )
        return False

    except requests.RequestException as exp:
        logger.exception("Health check request to %s failed: %s", url, exp)
        return False
