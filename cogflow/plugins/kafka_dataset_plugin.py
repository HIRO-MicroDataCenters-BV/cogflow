import os

from cogflow import plugin_config
from ..schema.kafka_dataset_metadata import KafkaDatasetRequest

from ..util import make_post_request, make_get_request
from dataclasses import asdict

kafka_datasets_url = "/datasets"
kafka_datasets_register = "/kafka/register"


class KafkaDatasetPlugin:

    def __init__(self):
        api_base_path = os.getenv(plugin_config.API_BASEPATH)
        if api_base_path:
            self.KAFKA_API_DATASET_URL = api_base_path + kafka_datasets_url
        else:
            raise Exception(
                f"Failed to initialize KafkaDatasetPlugin,: {plugin_config.API_BASEPATH} "
                f"env variable is not set"
            )

    def register_kafka_dataset(self, request: KafkaDatasetRequest):
        url = self.KAFKA_API_DATASET_URL + kafka_datasets_register
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
            print(f"Invalid response or data format encountered.")
            print(f"Error: {str(ve)}")
        except Exception as ex:
            print(
                f"An unexpected while registering kafka dataset error occurred: {str(ex)}"
            )

    def get_kafka_dataset(self, dataset_id):
        """
        get kafka dataset details by server_id
        """
        url = f"{self.KAFKA_API_DATASET_URL}/{dataset_id}/kafka"
        try:
            response = make_get_request(url=url)
            return response
        except ConnectionError as ce:
            print(f"Network issue: Unable to connect to {url}")
            print(f"Error: {str(ce)}")
        except ValueError as ve:
            print(f"Invalid response or data format encountered.")
            print(f"Error: {str(ve)}")
        except Exception as ex:
            print(
                f"An unexpected while registering kafka dataset error occurred: {str(ex)}, "
                f"for url : {str(url)}"
            )
