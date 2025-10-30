"""
CogFlow Configuration Module

Centralized configuration for CogFlow — fully environment-driven using Pydantic.
Environment variables directly match field names (no prefix required).
"""

import logging
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class CogFlowSettings(BaseSettings):
    """Global configuration model for CogFlow."""

    # ------------------------------------------------------------------
    # ⚙️ Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = Field(default="INFO", description="Logging level for CogFlow")

    # ------------------------------------------------------------------
    # 🌐 Core Runtime Settings
    # ------------------------------------------------------------------
    COG_API_PATH: str = Field(default="http://cog-api.kubeflow/cogapi")
    TIMER_IN_SEC: int = Field(default=10)
    FILE_TYPE: int = Field(default=0)
    TIME_OUT: int = Field(default=300)

    # ------------------------------------------------------------------
    # 🧠 MLflow / Storage Settings
    # ------------------------------------------------------------------
    MLFLOW_TRACKING_URI: str = Field(
        default="http://mlflow-server.kubeflow.svc.cluster.local:5000",
        description="Primary tracking URI for MLflow.",
    )
    MLFLOW_S3_ENDPOINT_URL: str = Field(default="http://mlflow-minio.kubeflow:9000")
    ML_TOOL: str = Field(default="mlflow")
    BUCKET_NAME: str = Field(default="mlflow")

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    MINIO_ENDPOINT_URL: str = Field(default="http://mlflow-minio.kubeflow:9000")
    MINIO_ACCESS_KEY: Optional[str] = None
    MINIO_SECRET_ACCESS_KEY: Optional[str] = None

    # ------------------------------------------------------------------
    # 🧩 Paths / API Endpoints
    # ------------------------------------------------------------------
    MODELS_URI: str = "/models/uri"
    VALIDATION_METRICS: str = "/validation/metrics"
    VALIDATION_ARTIFACTS: str = "/validation/artifacts"
    DATASET_REGISTER: str = "/dataset/register_by_api"
    DATASET: str = "/datasets"
    LINK_DATASET_MODEL: str = "/link_dataset_model"
    MODELS: str = "/models"
    PIPELINE: str = "/pipeline"
    PIPELINE_RUNS: str = "/pipeline/runs"
    MODEL_RECOMMEND: str = "/models/recommend"
    COMPONENTS: str = "/training-builder-components"
    PROMETHEUS_DATASETS: str = "/datasets/prometheus"
    LOG_MODEL: str = "/models/log"

    # ------------------------------------------------------------------
    # 🧰 MLflow Behavior
    # ------------------------------------------------------------------
    SERIALIZATION_FORMAT: str = "cloudpickle"
    AWAIT_REGISTRATION_FOR: int = 300
    PYFUNC_PREDICT_FN: str = "predict"
    ENV_MANAGER: str = "local"
    MAX_RESULTS: int = 100
    CONTAINER_PORT: int = 8080

    # ------------------------------------------------------------------
    # 🧪 Testing / Database
    # ------------------------------------------------------------------
    SQLALCHEMY_TEST_DATABASE_URI: str = "sqlite:///test.db"
    TESTING_CONFIG: str = "config.app_config.TestingConfig"

    # ------------------------------------------------------------------
    # ⚙️ DEX Authentication
    # ------------------------------------------------------------------
    POD_NAME: str = "dex-auth-0"
    NAMESPACE: str = "kubeflow"
    CONTAINER: str = "dex"
    CONFIG_PATH: str = "/etc/dex/config.docker.yaml"
    GRPC_PORT: int = 5557

    # ------------------------------------------------------------------
    # 🐳 Docker Images
    # ------------------------------------------------------------------
    BASE_IMAGE: str = "hiroregistry/cogflow:latest"
    FL_COGFLOW_BASE_IMAGE: str = "hiroregistry/flcogflow:latest"
    TRANSFORMER_BASE_IMAGE: str = "hiroregistry/k8-transformer:latest"

    # ------------------------------------------------------------------
    # 📨 Message Broker
    # ------------------------------------------------------------------
    MESSAGE_BROKER_DATASETS_URL: str = "/datasets"
    MESSAGE_BROKER_DATASETS_REGISTER: str = "/broker/register"
    MESSAGE_BROKER_TOPIC_REGISTER: str = "/topic/register"
    MESSAGE_BROKER_TOPIC_DATASETS_REGISTER: str = "/message/register"
    MESSAGE_BROKER_TOPIC_DATASETS_DETAILS: str = "/message/details"

    # ------------------------------------------------------------------
    # 🌍 Network / Retry Settings
    # ------------------------------------------------------------------
    DEFAULT_TIMEOUT: int = 15
    RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_MIN: int = 2
    RETRY_BACKOFF_MAX: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",  # load from .env if present
        case_sensitive=False,  # env vars not case-sensitive
        extra="ignore",
    )


@lru_cache
def get_settings() -> CogFlowSettings:
    """Load and cache CogFlow settings."""
    return CogFlowSettings()


# Global config instance
config = get_settings()

# ----------------------------------------------------------------------
# 🔊 Configure Global Logging Level
# ----------------------------------------------------------------------
logging.getLogger("cogflow").setLevel(
    getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
)
