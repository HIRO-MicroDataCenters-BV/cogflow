"""
This module defines constants for configuring Mlflow and AWS S3, as well as database credentials
 and MinIO settings.

Attributes:
    TRACKING_URI (str): The URI of the Mlflow tracking server. Used to specify where Mlflow logs and
    artifacts should be stored.
    S3_ENDPOINT_URL (str): The endpoint URL for the AWS S3 service. Needed for accessing and
     storing data in an S3 bucket.
    ACCESS_KEY_ID (str): The access key ID for AWS S3 authentication. Used in conjunction with
     SECRET_ACCESS_KEY for secure access to S3.
    SECRET_ACCESS_KEY (str): The secret access key for AWS S3 authentication. Paired with
     ACCESS_KEY_ID for secure access to S3.
    TIMER_IN_SEC (int): The interval in seconds for operations that require timing, such as
     periodic checks or updates. Default is set to 10 seconds.
    ML_TOOL (str): The name of the machine learning tool. Currently set to "mlflow" to
     specify the use of the Mlflow framework.
    MINIO_ENDPOINT_URL (str): The endpoint URL for the MinIO service, an alternative to AWS S3.
     Used for accessing and storing data.
    MINIO_ACCESS_KEY (str): The access key for MinIO authentication. Used with
     MINIO_SECRET_ACCESS_KEY for secure access to MinIO.
    MINIO_SECRET_ACCESS_KEY (str): The secret access key for MinIO authentication. Combined with
     MINIO_ACCESS_KEY for secure access.
     nats_kafka_connector_json (str): nats_kafka_connector_json
"""

COGFLOW_CONFIG_FILE_PATH = "COGFLOW_CONFIG_FILE_PATH"

TRACKING_URI = "MLFLOW_TRACKING_URI"
S3_ENDPOINT_URL = "MLFLOW_S3_ENDPOINT_URL"
ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
TIMER_IN_SEC = 10
ML_TOOL = "mlflow"

MINIO_ENDPOINT_URL = "MLFLOW_S3_ENDPOINT_URL"
MINIO_ACCESS_KEY = "AWS_ACCESS_KEY_ID"
MINIO_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"

SQLALCHEMY_TEST_DATABASE_URI = "sqlite:///test.db"
TESTING_CONFIG = "config.app_config.TestingConfig"

FILE_TYPE = 0

API_BASEPATH = "API_BASEPATH"

BASE_IMAGE = "hiroregistry/cogflow:latest"
FL_COGFLOW_BASE_IMAGE = "hiroregistry/flcogflow:latest"

# MESSAGE_BROKER plugin
MESSAGE_BROKER_DATASETS_URL = "/datasets"
MESSAGE_BROKER_DATASETS_REGISTER = "/broker/register"
MESSAGE_BROKER_TOPIC_REGISTER = "/topic/register"
MESSAGE_BROKER_TOPIC_DATASETS_REGISTER = "/message/register"
MESSAGE_BROKER_TOPIC_DATASETS_DETAILS = "/message/details"

# KNATIVE PLUGIN
NATS_KAFKA_CONNECTOR_JSON = """
reconnectinterval: 5000,

connecttimeout: 5000,

logging: {
  time: true,
  debug: false,
  trace: false,
  colors: true,
  pid: false,
}

monitoring: {
  httpport: 9222,
}

nats: {
  Servers: ["nats://cog-nats.nats.svc.cluster.local:4222"],
  ConnectTimeout: 5000,
  MaxReconnects: 120,
  ReconnectWait: 5000,
}


connect: [
  {
    id: "IRISNATs"
    type: "NATSToKafka",
    subject: "iris.stream",
    topic: "iris-requests",
    brokers: ["kafka-cluster-kafka-bootstrap.kafka:9092"],
  },
  {
    type: "JetStreamToKafka",
    brokers: ["kafka-cluster-kafka-bootstrap.kafka:9092"]
    id: "irisstream-1",
    topic: "iris-requests-1",
    subject: "iris.stream",
    durablename: "durable_iris_consumer",
    queuename: "iris_consumers"
  },
]
"""
