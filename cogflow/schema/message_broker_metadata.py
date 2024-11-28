"""
kafka dataset metadata schema class
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class MessageBrokerRequest:
    """
    Class used for  metadata of Dataset
    """
    broker_name: str
    broker_ip: str


@dataclass
class MessageBrokerTopicRequest:
    """
    Class used for  metadata of Dataset
    """
    topic_name: str
    topic_schema: Dict
    broker_id: int


@dataclass
class MessageBrokerTopicDataSetRegisterRequest:
    """
    Class used for  metadata of Dataset
    """
    dataset_type: int
    dataset_name: str
    description: str
    broker_id: int
    topic_id: int


@dataclass
class MessageBrokerTopicDetail:
    broker_ip: str
    topic_name: str
    topic_schema: Dict
