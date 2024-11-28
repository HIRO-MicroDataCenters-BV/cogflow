"""
Message Broker dataset plugin implementation class
"""
import json
import os
import re
from dataclasses import asdict
from .. import plugin_config

from ..schema.message_broker_metadata import MessageBrokerRequest, MessageBrokerTopicRequest, \
    MessageBrokerTopicDataSetRegisterRequest, MessageBrokerTopicDetail

from ..util import make_post_request, make_get_request

message_broker_datasets_url = plugin_config.MESSAGE_BROKER_DATASETS_URL
message_broker_register = plugin_config.MESSAGE_BROKER_DATASETS_REGISTER
message_broker_topic_register = plugin_config.MESSAGE_BROKER_TOPIC_REGISTER
message_broker_topic_datasets_register = plugin_config.MESSAGE_BROKER_TOPIC_DATASETS_REGISTER
message_broker_topic_datasets_details = plugin_config.MESSAGE_BROKER_TOPIC_DATASETS_DETAILS


class MessageBrokerDatasetPlugin:
    """
    kafka dataset plugin implementation
    """

    def __init__(self):
        api_base_path = os.getenv(plugin_config.API_BASEPATH)
        if api_base_path:
            self.kafka_api_dataset_url = api_base_path + message_broker_datasets_url
        else:
            raise Exception(
                f"Failed to initialize KafkaDatasetPlugin,: {plugin_config.API_BASEPATH} "
                f"env variable is not set"
            )
        print(self.kafka_api_dataset_url)

    def register_message_broker_dataset(self, dataset_name: str, broker_name: str, broker_ip: str, topic_name: str):

        message_broker_id = self.register_message_broker(broker_name, broker_ip)
        print(f"Start registering topic for message broker {message_broker_id}")
        topic_id = self.register_message_topic(message_broker_id, topic_name)
        print(f"Topic [{topic_id}] is register with broker [{message_broker_id}]")

        print(f"Start registering topic {topic_id} with dataset")
        dataset_id = self.register_topic_dataset(dataset_name, message_broker_id, topic_id)
        print(dataset_id)

    def get_message_broker_details(self, dataset_id):
        url = f"{self.kafka_api_dataset_url}{message_broker_topic_datasets_details}?dataset_id={dataset_id}"
        print(url)
        try:
            response = make_get_request(url)
            broker_ip = response["data"]["BrokerDetails"]["broker_ip"]
            topic_name = response["data"]["TopicDetails"]["topic_name"]
            topic_schema = response["data"]["TopicDetails"]["topic_schema"]
            topic_detail = MessageBrokerTopicDetail(
                broker_ip=broker_ip,
                topic_name=topic_name,
                topic_schema=topic_schema
            )
            print(topic_detail)
            return topic_detail
        except Exception as ex:
            print(ex)

    def register_topic_dataset(self, dataset_name, message_broker_id, topic_id):

        url = self.kafka_api_dataset_url + message_broker_topic_datasets_register
        request = MessageBrokerTopicDataSetRegisterRequest(0, dataset_name, "done via jupyter notebook",
                                                           message_broker_id,
                                                           topic_id)
        try:
            response = make_post_request(url=url, data=asdict(request))
            if response:
                dataset_id = response["data"]["dataset"]["id"]
                broker_ip = response["data"]["broker_details"]["broker_ip"]
                topic_id = response["data"]["topic_details"]["id"]
                topic_name = response["data"]["topic_details"]["topic_name"]
                print(f"Dataset [{dataset_id}] registered with topic id : [{topic_id}], "
                      f"topic name: {topic_name}, broker id {broker_ip}")
                return dataset_id
        except Exception as ex:
            print(ex)

    def register_message_topic(self, message_broker_id, topic_name):

        url = self.kafka_api_dataset_url + message_broker_topic_register
        try:
            request = MessageBrokerTopicRequest(topic_name, {}, message_broker_id)
            response = make_post_request(url=url, data=asdict(request))
            if response:
                message_broker_topic_id = response["data"]["id"]
                print(f"New topic is created with id {message_broker_topic_id}")
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
                        f"Topic [{topic_name}] already registered for broker id {message_broker_id}")
                    return topic_id
                else:
                    print(
                        f"An unexpected while registering kafka dataset error occurred: {str(ex)}"
                    )
            print(
                f"An unexpected while registering kafka dataset error occurred: {str(ex)}"
            )

    def register_message_broker(self, broker_name: str, broker_ip: str):
        url = self.kafka_api_dataset_url + message_broker_register
        try:
            request = MessageBrokerRequest(broker_name, broker_ip)
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
                else:
                    print(
                        f"An unexpected while registering kafka dataset error occurred: {str(ex)}"
                    )
            print(
                f"An unexpected while registering kafka dataset error occurred: {str(ex)}"
            )
