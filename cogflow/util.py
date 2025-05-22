"""
    Utility functions
"""

import re
from datetime import datetime
import requests
from . import plugin_config

DEFAULT_TIMEOUT = plugin_config.TIMER_IN_SEC  # Set a default timeout in seconds


def make_post_request(url, data=None, params=None, files=None, timeout=DEFAULT_TIMEOUT):
    """
    Utility function to make POST requests
    :param url: URL of the API endpoint
    :param data: JSON payload
    :param params: Request params
    :param files: File
    :param timeout: Timeout for the request
    :return: Response for the POST request in JSON format
    """
    try:
        if data:
            response = requests.post(url, json=data, params=params, timeout=timeout)
        elif files:
            with open(files, "rb") as file_data:
                file = {"file": file_data}
                response = requests.post(
                    url, params=params, files=file, timeout=timeout
                )
                file_data.close()
        else:
            response = requests.post(url, params=params, timeout=timeout)

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


def make_get_request(url, path_params=None, query_params=None, timeout=DEFAULT_TIMEOUT):
    """
    Utility function to make GET requests
    :param url: URL of the API endpoint
    :param path_params: Path params
    :param query_params: Query params
    :param timeout: Timeout for the request
    :return: Response for the GET request in JSON format
    """
    try:
        if query_params:
            response = requests.get(url, params=query_params, timeout=timeout)
        elif path_params:
            # Make the GET request with path params
            response = requests.get(url + "/" + path_params, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        # If not the success response
        print(f"GET request failed with status code {response.status_code}")
        raise Exception("Request failed")
    except requests.exceptions.RequestException as exp:
        print(f"Error making GET request: {exp}")
        raise Exception(f"Error making GET request: {exp}")
