"""
Message Broker dataset plugin implementation class
"""

import os
import re
from dataclasses import asdict
from .. import plugin_config

from ..schema.message_broker_metadata import (
    MessageBrokerRequest,
    MessageBrokerTopicRequest,
    MessageBrokerTopicDataSetRegisterRequest,
    MessageBrokerTopicDetail,
)

from ..util import make_post_request, make_get_request

message_broker_datasets_url = plugin_config.MESSAGE_BROKER_DATASETS_URL
message_broker_register = plugin_config.MESSAGE_BROKER_DATASETS_REGISTER
message_broker_topic_register = plugin_config.MESSAGE_BROKER_TOPIC_REGISTER
message_broker_topic_datasets_register = (
    plugin_config.MESSAGE_BROKER_TOPIC_DATASETS_REGISTER
)
message_broker_topic_datasets_details = (
    plugin_config.MESSAGE_BROKER_TOPIC_DATASETS_DETAILS
)


class MessageBrokerDatasetPlugin:
    """
    message broker dataset plugin dataset plugin implementation
    """

    def __init__(self):
        api_base_path = os.getenv(plugin_config.API_BASEPATH)
        if api_base_path:
            self.message_broker_api_dataset_url = (
                api_base_path + message_broker_datasets_url
            )
        else:
            raise Exception(
                f"Failed to initialize MessageBrokerDatasetPlugin,: {plugin_config.API_BASEPATH} "
                f"env variable is not set"
            )
        print(self.message_broker_api_dataset_url)

    def register_message_broker_dataset(
        self,
        dataset_name: str,
        broker_name: str,
        broker_ip: str,
        topic_name: str,
        broker_port: int,
    ):
        """
        Registers a dataset, message broker, and topic in the system.

        This method performs the following steps:
        1. Registers a message broker with the specified name, IP, and port.
        2. Registers a topic for the message broker.
        3. Associates the topic with a dataset.

        Args:
            dataset_name (str): The name of the dataset to be registered.
            broker_name (str): The name of the message broker (e.g., Kafka, RabbitMQ).
            broker_ip (str): The IP address of the message broker.
            topic_name (str): The name of the topic to be registered with the broker.
            broker_port (int): The port number of the message broker.

        Returns:
            None
        """

        print(f"Start registering broker : [{broker_name}]")
        message_broker_id = self.register_message_broker(
            broker_name, broker_ip, broker_port
        )
        print(
            f"Start registering [{topic_name}] topic for message broker {message_broker_id}"
        )
        topic_id = self.register_message_topic(message_broker_id, topic_name)
        print(f"Topic [{topic_id}] is register with broker [{message_broker_id}]")

        print(f"Start registering [{dataset_name}] topic [{topic_id}] with dataset")
        dataset_id = self.register_topic_dataset(
            dataset_name, message_broker_id, topic_id
        )
        print(f"new registered data set dataset_id [{dataset_id}]")

    def get_message_broker_details(self, dataset_id):
        """
        Retrieves message broker details and topic information for a given dataset.

        Args:
            dataset_id (int): The ID of the dataset for which the broker and topic details are to be fetched.

        Returns:
            MessageBrokerTopicDetail: An object containing the broker's IP, port, topic name, and schema.

        Raises:
            Exception: Logs and raises any exception that occurs during the API request or data processing.
        """
        url = f"{self.message_broker_api_dataset_url}{message_broker_topic_datasets_details}?dataset_id={dataset_id}"
        try:
            response = make_get_request(url)
            print(response)
            broker_ip = response["data"]["broker_details"]["broker_ip"]
            broker_port = response["data"]["broker_details"]["broker_port"]
            topic_name = response["data"]["topic_details"]["topic_name"]
            topic_schema = response["data"]["topic_details"]["topic_schema"]
            topic_detail = MessageBrokerTopicDetail(
                broker_ip=broker_ip,
                broker_port=broker_port,
                topic_name=topic_name,
                topic_schema=topic_schema,
            )
            return topic_detail
        except Exception as ex:
            print(ex)

    def register_topic_dataset(self, dataset_name, message_broker_id, topic_id):
        """
        Registers a dataset with a message broker and topic.

        This method sends a POST request to register a new dataset for a specific message broker and topic.
        It constructs a `MessageBrokerTopicDataSetRegisterRequest` with the given parameters and logs the
        response, including the IDs of the dataset, broker, and topic.

        Args:
            dataset_name (str): The name of the dataset to be registered.
            message_broker_id (int): The ID of the message broker where the dataset will be associated.
            topic_id (int): The ID of the topic with which the dataset will be linked.

        Returns:
            int: The ID of the newly registered dataset.

        Raises:
            Exception: Logs and raises any exception that occurs during the API request or data processing.
        """
        url = (
            self.message_broker_api_dataset_url + message_broker_topic_datasets_register
        )
        request = MessageBrokerTopicDataSetRegisterRequest(
            0, dataset_name, "done via jupyter notebook", message_broker_id, topic_id
        )
        try:
            response = make_post_request(url=url, data=asdict(request))
            if response:
                dataset_id = response["data"]["dataset"]["id"]
                broker_id = response["data"]["broker_details"]["id"]
                topic_id = response["data"]["topic_details"]["id"]
                topic_name = response["data"]["topic_details"]["topic_name"]
                print(
                    f"Dataset [{dataset_id}] registered with topic id : [{topic_id}], "
                    f"topic name: {topic_name}, broker id {broker_id}"
                )
                return dataset_id
        except Exception as ex:
            print(ex)

    def register_message_topic(self, message_broker_id, topic_name):
        """
        Registers a new message topic with the specified message broker.

        Args:
            message_broker_id (int): The ID of the message broker to which the topic will be registered.
            topic_name (str): The name of the topic to be registered.

        Returns:
            int: The ID of the newly created topic or the existing topic if already registered.

        Raises:
            ConnectionError: Raised if there is a network issue preventing the API request from completing.
            ValueError: Raised if the response data format is invalid or if an unexpected response is received.
            Exception: Catches unexpected errors and checks for the "Topic Already Exists" condition.
        """

        url = self.message_broker_api_dataset_url + message_broker_topic_register
        try:
            request = MessageBrokerTopicRequest(topic_name, {}, message_broker_id)
            response = make_post_request(url=url, data=asdict(request))
            if response:
                message_broker_topic_id = response["data"]["id"]
                print(f"New topic is created with id [{message_broker_topic_id}]")
                return message_broker_topic_id
        except ConnectionError as ce:
            print(f"Network issue: Unable to connect to {url}")
            print(f"Error: {str(ce)}")
        except ValueError as ve:
            print("Invalid response or data format encountered.")
            print(f"Error: {str(ve)}")
        except Exception as ex:

            if ex.args:
                response_json = ex.args[0]

                pattern = r"Topic Already Exists."
                match = re.search(pattern, response_json["detail"]["message"])
                if match:
                    topic_id = response_json["detail"]["topic_id"]
                    print(response_json["detail"]["message"])
                    print(
                        f"Topic [{topic_name}] already registered for broker id {message_broker_id}"
                    )
                    return topic_id
            print(
                f"An unexpected while registering message broker dataset error occurred: {str(ex)}"
            )

    def register_message_broker(
        self, broker_name: str, broker_ip: str, broker_port: int
    ):
        """
        Registers a new message topic with the specified message broker.

        This method sends a POST request to register a new topic for a given message broker.
        If the topic is already registered, it will identify the existing topic and return its ID.
        In case of errors (e.g., network issues, invalid responses, or other exceptions),
        detailed error messages are printed.

        Args:
            broker_name (str): broker name to be registered.
            broker_ip (str): broker ip to be registered.
            broker_port (int): broker port to be registered.

        Returns:
            int: The ID of the newly created or existing topic.

        Raises:
            ConnectionError: Raised if there is a network issue preventing connection to the API endpoint.
            ValueError: Raised if the response or data format from the API is invalid or unexpected.
            Exception: Handles unexpected errors, including checking for existing topics.

        """
        url = self.message_broker_api_dataset_url + message_broker_register
        try:

            request = MessageBrokerRequest(broker_name, broker_ip, broker_port)
            response = make_post_request(url=url, data=asdict(request))
            if response:
                message_broker_ip = response["data"]["id"]
                print(f"New message broker is created with id {message_broker_ip}")
                return message_broker_ip
        except ConnectionError as ce:
            print(f"Network issue: Unable to connect to {url}")
            print(f"Error: {str(ce)}")
        except ValueError as ve:
            print("Invalid response or data format encountered.")
            print(f"Error: {str(ve)}")
        except Exception as ex:
            if ex.args:
                response_json = ex.args[0]
                pattern = r"Broker id (\d+) already exists\."
                match = re.search(pattern, response_json["detail"]["message"])
                if match:
                    broker_id = response_json["detail"]["broker_id"]
                    print(response_json["detail"]["message"])
                    print(f"Already message broker exists {broker_id}")
                    return broker_id
            print(
                f"An unexpected while registering message broker dataset error occurred: {str(ex)}"
            )
