"""
Kafka Producer implementation class
"""

import json
import time

from kafka import KafkaProducer

KAFKA_BROKER_URL = "localhost:9092"
TOPIC_NAME = "metrics"


def send_message(producer, topic_name, message):
    """Send a message to the Kafka topic."""
    try:
        future = producer.send(topic_name, value=message)
        future.get(timeout=10)  # Ensure the message was sent successfully
        print(f"Message sent: {message}")
    except Exception as e:
        print(f"Failed to send message: {e}")


def get_kafka_producer(kafka_broker_url):
    """
    Initialize and return a Kafka producer.

    This function sets up a Kafka producer configured with retry and acknowledgment settings
    to ensure reliable message delivery. It serializes the messages to JSON format before
    sending to Kafka. The producer attempts to resend messages up to 5 times in case of
    failure, and waits for acknowledgment from all replicas to confirm message persistence.

    Parameters:
    - kafka_broker_url (str): The URL of the Kafka broker to connect to.

    Returns:
    - KafkaProducer: An instance of KafkaProducer configured for JSON message serialization.

    Kafka Configuration:
    - `bootstrap_servers`: Specifies the Kafka broker URL.
    - `retries`: Number of retries in case of message delivery failure (set to 5).
    - `acks`: Configured to 'all' to ensure the message is fully committed to the log by
      waiting for acknowledgment from all in-sync replicas.
    """
    kafka_config = {
        "bootstrap_servers": kafka_broker_url,
        "retries": 5,  # Retry sending messages on failure
        "acks": "all",  # Ensure message is committed to the log
    }

    return KafkaProducer(
        **kafka_config,
        value_serializer=lambda v: json.dumps(v).encode(
            "utf-8"
        ),  # Convert Python dict to JSON
    )


def send_kafka_message():
    """
    Sends a sequence of messages to a specified Kafka topic at regular intervals.

    This function initiates a Kafka producer, then sends messages to a Kafka topic
    every second in a loop, with each message containing a sequential number and a
    timestamp. The producer connection is closed after all messages are sent.

    Parameters:
    - None

    Returns:
    - None

    Notes:
    - This function is hardcoded to send messages with a 'number' field starting at 14 and incrementing up to 100,
      along with the current timestamp.
    - The Kafka producer is initialized with `KAFKA_BROKER_URL` and sends messages to `TOPIC_NAME`.
    """
    kafka_producer = get_kafka_producer(KAFKA_BROKER_URL)

    # Simulate sending messages every second
    for i in range(14, 100):
        message1 = {"number": i, "timestamp": time.time()}
        send_message(kafka_producer, TOPIC_NAME, message1)
        time.sleep(1)

    kafka_producer.close()


send_kafka_message()
