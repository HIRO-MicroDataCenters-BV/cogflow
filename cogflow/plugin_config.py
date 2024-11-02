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

    ML_DB_USERNAME (str): Username for connecting to the machine learning database (ML_DB).
     Essential for database access authentication.
    ML_DB_PASSWORD (str): Password for connecting to the machine learning database (ML_DB).
     Should be kept secure.
    ML_DB_HOST (str): Host address for connecting to the machine learning database (ML_DB).
    ML_DB_PORT (str): Port number for connecting to the machine learning database (ML_DB).
    ML_DB_NAME (str): Name of the machine learning database (ML_DB).

    Cogflow_DB_USERNAME (str): Username for connecting to the CogFlow database. Important for
     authenticating access to the database.
    Cogflow_DB_PASSWORD (str): Password for connecting to the CogFlow database.
     Should be stored securely.
    Cogflow_DB_HOST (str): Host address for connecting to the CogFlow database.
    Cogflow_DB_PORT (str): Port number for connecting to the CogFlow database.
    Cogflow_DB_NAME (str): Name of the CogFlow database.

    MINIO_ENDPOINT_URL (str): The endpoint URL for the MinIO service, an alternative to AWS S3.
     Used for accessing and storing data.
    MINIO_ACCESS_KEY (str): The access key for MinIO authentication. Used with
     MINIO_SECRET_ACCESS_KEY for secure access to MinIO.
    MINIO_SECRET_ACCESS_KEY (str): The secret access key for MinIO authentication. Combined with
     MINIO_ACCESS_KEY for secure access.
"""

COGFLOW_CONFIG_FILE_PATH = "COGFLOW_CONFIG_FILE_PATH"

TRACKING_URI = "MLFLOW_TRACKING_URI"
S3_ENDPOINT_URL = "MLFLOW_S3_ENDPOINT_URL"
ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
TIMER_IN_SEC = 10
ML_TOOL = "mlflow"

ML_DB_USERNAME = "ML_DB_USERNAME"
ML_DB_PASSWORD = "ML_DB_PASSWORD"
ML_DB_HOST = "ML_DB_HOST"
ML_DB_PORT = "ML_DB_PORT"
ML_DB_NAME = "ML_DB_NAME"

COGFLOW_DB_USERNAME = "DB_USER"
COGFLOW_DB_PASSWORD = "DB_PASSWORD"
COGFLOW_DB_HOST = "DB_HOST"
COGFLOW_DB_PORT = "DB_PORT"
COGFLOW_DB_NAME = "DB_NAME"

MINIO_ENDPOINT_URL = "MLFLOW_S3_ENDPOINT_URL"
MINIO_ACCESS_KEY = "AWS_ACCESS_KEY_ID"
MINIO_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"

SQLALCHEMY_TEST_DATABASE_URI = "sqlite:///test.db"
TESTING_CONFIG = "config.app_config.TestingConfig"

JUPYTER_USER_ID = 0

API_BASEPATH = "API_BASEPATH"
