"""
kafka dataset metadata schema class
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TopicDetail:
    """
    topic details class
    """

    topic_name: str
    topic_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KafkaDatasetRequest:
    """
    Class used for  metadata of Dataset
    """

    dataset_name: str
    dataset_description: str
    host_name: str
    server_ip: str
    topic_details: List[TopicDetail] = field(default_factory=list)
