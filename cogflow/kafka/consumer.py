"""
Kafka consumer implementation class
"""

import json
import threading

from kafka import KafkaConsumer


class KafkaThread:
    """
    A class to manage the Kafka consumer and its dedicated thread for message consumption.

    Attributes:
    - stop_flag (threading.Event): Event flag to signal when the consumer thread should stop.
      Set to 'True' to stop the thread safely.
    - CONSUMER_THREAD (Thread): Reference to the thread running the Kafka consumer to facilitate
      starting and stopping operations.
    - CONSUMER (KafkaConsumer): Holds the KafkaConsumer instance for managing message consumption
      from a specified Kafka topic.

    This class is intended for managing a Kafka consumer within a multi-threaded environment,
    allowing asynchronous message processing and easy control of the consumer lifecycle.
    """

    stop_flag = threading.Event()
    CONSUMER_THREAD = None  # Holds the reference to the consumer thread
    CONSUMER = None  # Holds the Kafka consumer instance


def get_kafka_consumer(kafka_broker_url, topic_name, group_id):
    """
    Initialize and return a Kafka consumer.

    This function sets up a Kafka consumer with specified configurations to read messages
    from a given topic. It is designed to start reading from the latest available message
    if no previous offset is found. The consumer supports automatic offset committing.

    Parameters:
    - kafka_broker_url (str): The URL of the Kafka broker to connect to.
    - topic_name (str): The name of the Kafka topic to subscribe to.
    - group_id (str): The consumer group ID that the consumer will join.

    Returns:
    - KafkaConsumer: An instance of KafkaConsumer configured to read from the specified topic.
    """

    # Consumer configurations
    kafka_config_for_consumer = {
        "bootstrap_servers": kafka_broker_url,
        "auto_offset_reset": "latest",
        "enable_auto_commit": True,  # Enable automatic offset commits
        "group_id": group_id,  # Consumer group for load balancing
        "session_timeout_ms": 30000,  # Consumer will time out after 30 seconds of inactivity
    }

    return KafkaConsumer(
        topic_name,
        **kafka_config_for_consumer,
        value_deserializer=lambda v: json.loads(
            v.decode("utf-8")
        ),  # Convert JSON to Python dict
    )


def read_messages(kafka_consumer):
    """Reads messages from Kafka topic."""
    print("Listening for messages...")
    # stop_flag.clear()
    KafkaThread.stop_flag.clear()
    try:
        for message in kafka_consumer:
            if KafkaThread.stop_flag.is_set():  # Check if a stop request was made
                print("Stop signal received. Exiting...")
                break  # Exit the loop and close the consumer
            try:
                print(f"Received message: {message.value}")
            except Exception as ex:
                print(f"Failed to process message {message}: {ex}")
    finally:
        kafka_consumer.close()
        print("Consumer closed.")


def start_consumer_thread(kafka_broker_url, topic_name, group_id):
    """Sets up and starts the Kafka consumer in a separate thread."""
    print("Starting Kafka consumer...")

    # Get Kafka consumer instance
    KafkaThread.CONSUMER = get_kafka_consumer(kafka_broker_url, topic_name, group_id)

    # Create and start a new thread for consuming messages
    KafkaThread.CONSUMER_THREAD = threading.Thread(
        target=read_messages, args=(KafkaThread.CONSUMER,)
    )
    # consumer_thread.daemon = True  # Daemon thread exits when the main program exits
    KafkaThread.CONSUMER_THREAD.start()


def stop_consumer():
    """Stops the Kafka consumer by setting the stop flag."""
    print("trying to stop ...")
    if KafkaThread.CONSUMER and KafkaThread.CONSUMER_THREAD:
        print("Stopping Kafka consumer...")
        KafkaThread.stop_flag.set()  # Signal the consumer to stop
        KafkaThread.CONSUMER_THREAD.join()  # Wait for the thread to finish
        print("Kafka consumer stopped.")
    else:
        print("kafka consumer is not started")
