"""
    Utility functions
"""

import re
import uuid
from datetime import datetime
import requests
from . import plugin_config

DEFAULT_TIMEOUT = plugin_config.TIMER_IN_SEC  # Set a default timeout in seconds


def make_post_request(
    url, data=None, params=None, files=None, headers=None, timeout=DEFAULT_TIMEOUT
):
    """
    Utility function to make POST requests
    :param url: URL of the API endpoint
    :param data: JSON payload (dict)
    :param params: Request params (dict)
    :param files: File path (str) to upload
    :param headers: Request headers (dict)
    :param timeout: Timeout for the request
    :return: Response for the POST request in JSON format
    """
    try:
        if files:
            # 'files' should be a dict: {'param_name': (filename, file_obj)}
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

        if response.status_code == 201:
            return response.json()
        # If not the success response
        print(f"POST request failed with status code {response.status_code}")
        raise Exception(response.json())
    except requests.exceptions.RequestException as exp:
        print(f"Error making POST request: {exp}")
        raise Exception(f"Error making POST request: {exp}")


def custom_serializer(obj):
    """
    Method to serialize obj to datetime ISO format
    :param obj:
    :return:
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def is_valid_s3_uri(uri):
    """
    Method to check if the URL is a valid S3 URL
    :param uri: URL to check
    :return:
    """
    # Regular expression for S3 URI
    s3_uri_regex = re.compile(r"^s3://([a-z0-9.-]+)/(.*)$")

    # Check against the regex pattern
    match = s3_uri_regex.match(uri)

    if match:
        bucket_name = match.group(1)
        object_key = match.group(2)

        # Additional checks for bucket name and object key can be added here
        if bucket_name and object_key:
            return True

    return False


def make_delete_request(
    url, path_params=None, query_params=None, timeout=DEFAULT_TIMEOUT
):
    """
    Utility function to make DELETE requests
    :param url: URL of the API endpoint
    :param path_params: Path params
    :param query_params: Query params
    :param timeout: Timeout for the request
    :return: Response for the DELETE request in JSON format
    """
    try:
        if query_params:
            response = requests.delete(url, params=query_params, timeout=timeout)
        else:
            # Make the DELETE request with path params
            response = requests.delete(url + "/" + path_params, timeout=timeout)
        if response.status_code == 200:
            print("DELETE request successful")
            return response.json()
        # If not the success response
        print(f"DELETE request failed with status code {response.status_code}")
        raise Exception("Request failed")
    except requests.exceptions.RequestException as exp:
        print(f"Error making DELETE request: {exp}")
        raise Exception(f"Error making DELETE request: {exp}")


def make_get_request(
    url,
    path_params=None,
    query_params=None,
    headers=None,
    timeout=DEFAULT_TIMEOUT,
    paginate=False,
):
    """
    Utility function to make GET requests (with optional pagination)

    :param url: Base API URL (e.g., https://api.example.com/resource)
    :param path_params: Additional path (e.g., "123/details")
    :param query_params: Dictionary of query parameters
    :param headers: Request headers (dict)
    :param timeout: Timeout in seconds
    :param paginate: If True, handles paginated responses
    :return: List (if paginate) or dict (JSON response)
    """
    try:
        # join base URL with path parameters
        full_url = (
            f"{url.rstrip('/')}/{str(path_params).lstrip('/')}" if path_params else url
        )

        if not paginate:
            response = requests.get(
                full_url, params=query_params, headers=headers, timeout=timeout
            )

            if response.status_code == 200:
                return response.json()
            print(f"GET request failed with status code {response.status_code}")
            raise Exception("Request failed")

        # Pagination mode
        all_data = []
        page = 1
        limit = query_params.get("limit", 10) if query_params else 10

        while True:
            page_params = query_params.copy() if query_params else {}
            page_params["page"] = page
            page_params["limit"] = limit

            response = requests.get(
                full_url, params=page_params, headers=headers, timeout=timeout
            )
            if response.status_code != 200:
                print(f"GET request failed with status code {response.status_code}")
                break

            json_data = response.json()
            data = json_data.get("data", [])
            all_data.extend(data)

            pagination = json_data.get("pagination", {})
            total_items = pagination.get("total_items", len(data))

            if len(all_data) >= total_items:
                break

            page += 1

        return all_data

    except requests.exceptions.RequestException as exp:
        print(f"Error making GET request: {exp}")
        raise Exception(f"Error making GET request: {exp}")


def uuid_to_canonical(value: str) -> str:
    """
    Convert a non-canonical (32-character hex) UUID string into
    canonical (hyphenated) UUID string format.

    Example:
        '123e4567e89b12d3a456426614174000' -> '123e4567-e89b-12d3-a456-426614174000'

    Performs full validation and raises ValueError for invalid inputs.
    """
    if not isinstance(value, str):
        raise ValueError(f"UUID must be a string, got {type(value).__name__}")

    try:
        u = uuid.UUID(value)  # Accepts both hyphenated and non-hyphenated
        return str(u)  # Always returns canonical (hyphenated) form
    except (ValueError, AttributeError, TypeError):
        raise ValueError(f"Invalid UUID value: {value!r}")


def uuid_to_hex(value: str) -> str:
    """
    Convert a UUID (canonical or hex string) to non-canonical (32-character hex) form.

    Example:
        '123e4567-e89b-12d3-a456-426614174000' -> '123e4567e89b12d3a456426614174000'
        '123e4567e89b12d3a456426614174000'     -> '123e4567e89b12d3a456426614174000'

    Performs full validation to ensure the input is a valid UUID.
    Raises ValueError for invalid formats or types.
    """
    try:
        # Convert to a UUID object (validates input)
        u = uuid.UUID(value)
        # Return its hex-only version (no hyphens)
        return u.hex
    except (ValueError, AttributeError, TypeError):
        raise ValueError(f"Invalid UUID value: {value!r}")


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL and save it to the specified path.

    Args:
        url (str): The URL of the file to download.
        output_path (str): The local path where the file will be saved.
        chunk_size (int): Size of chunks to read at a time (default: 8192 bytes).
    """
    with requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT) as response:
        response.raise_for_status()  # Raise error for bad status codes
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)
    print(f"✅ Downloaded: {output_path}")
