"""
Message dataset metadata schema class
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class MessageBrokerRequest:
    """
    Class used for  metadata of MessageBrokerRequest
    """

    broker_name: str
    broker_ip: str
    broker_port: int


@dataclass
class MessageBrokerTopicRequest:
    """
    Class used for  metadata of MessageBrokerTopicRequest
    """

    topic_name: str
    topic_schema: Dict
    broker_id: int


@dataclass
class MessageBrokerTopicDataSetRegisterRequest:
    """
    Class used for  metadata of MessageBrokerTopicDataSetRegisterRequest
    """

    dataset_type: int
    dataset_name: str
    description: str
    broker_id: int
    topic_id: int


@dataclass
class MessageBrokerTopicDetail:
    """
    Class used for  metadata of MessageBrokerTopicDetail
    """

    broker_ip: str
    broker_port: int
    topic_name: str
    topic_schema: Dict
