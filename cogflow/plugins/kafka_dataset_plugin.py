"""
kafka dataset plugin implementation class
"""

import os
from dataclasses import asdict

from .. import plugin_config
from ..schema.kafka_dataset_metadata import KafkaDatasetRequest
from ..util import make_post_request, make_get_request

kafka_datasets_url = plugin_config.KAFKA_DATASETS_URL
kafka_datasets_register = plugin_config.KAFKA_DATASETS_REGISTER


class KafkaDatasetPlugin:
    """
    kafka dataset plugin implementation
    """

    def __init__(self):
        api_base_path = os.getenv(plugin_config.API_BASEPATH)
        if api_base_path:
            self.kafka_api_dataset_url = api_base_path + kafka_datasets_url
        else:
            raise Exception(
                f"Failed to initialize KafkaDatasetPlugin,: {plugin_config.API_BASEPATH} "
                f"env variable is not set"
            )

    def register_kafka_dataset(self, request: KafkaDatasetRequest):
        """
        register kafka dataset details
        :param request:
        :return:
        """
        url = self.kafka_api_dataset_url + kafka_datasets_register
        try:
            print("Registering kafka server ..")
            response = make_post_request(url=url, data=asdict(request))
            dataset_id = response["data"]["dataset"]["id"]
            print(f"Dataset registered with dataset_id : {dataset_id}")
            return dataset_id
        except ConnectionError as ce:
            print(f"Network issue: Unable to connect to {url}")
            print(f"Error: {str(ce)}")
        except ValueError as ve:
            print("Invalid response or data format encountered.")
            print(f"Error: {str(ve)}")
        except Exception as ex:
            print(
                f"An unexpected while registering kafka dataset error occurred: {str(ex)}"
            )

    def get_kafka_dataset(self, dataset_id):
        """
        get kafka dataset details by dataset_id
        :param dataset_id:
        :return:
        """
        url = f"{self.kafka_api_dataset_url}/{dataset_id}/kafka"
        try:
            response = make_get_request(url=url)
            return response
        except ConnectionError as ce:
            print(f"Network issue: Unable to connect to {url}")
            print(f"Error: {str(ce)}")
        except ValueError as ve:
            print("Invalid response or data format encountered.")
            print(f"Error: {str(ve)}")
        except Exception as ex:
            print(
                f"An unexpected while registering kafka dataset error occurred: {str(ex)}, "
                f"for url : {str(url)}"
            )
