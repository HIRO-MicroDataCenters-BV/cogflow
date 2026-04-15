"""
CogFlow Configuration Module
=============================

Centralized configuration for CogFlow — environment-driven via Pydantic.
Environment variables always take precedence; defaults are used otherwise.

Compatible with:
  • Pydantic 1.x and 2.x  ✅
  • Python 3.10+
"""

import logging
from functools import lru_cache
from typing import Optional

from pydantic import Field

try:
    from pydantic_settings import BaseSettings  # pydantic 2
except ImportError:  # pydantic 1
    from pydantic import BaseSettings  # type: ignore[no-redef,assignment]


# ----------------------------------------------------------------------
# ⚙️ Unified Settings Class
# ----------------------------------------------------------------------
class CogFlowSettings(BaseSettings):
    """Global configuration model for CogFlow."""

    # ---------- Logging ----------
    LOG_LEVEL: str = Field("WARNING", description="Logging level for CogFlow")

    # ---------- Core Runtime ----------
    API_PATH: str = Field("http://cog-api.kubeflow/cogapi")
    TIMER_IN_SEC: int = 10
    FILE_TYPE: int = 0
    TIME_OUT: int = 300

    # ---------- MLflow / Storage ----------
    MLFLOW_TRACKING_URI: str = Field(
        "http://mlflow-server.kubeflow.svc.cluster.local:5000",
        description="Primary tracking URI for MLflow.",
    )
    MLFLOW_S3_ENDPOINT_URL: str = "http://mlflow-minio.kubeflow:9000"
    ML_TOOL: str = "mlflow"
    BUCKET_NAME: str = "mlflow"

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    MINIO_SECURE: Optional[bool] = False

    # ---------- API Paths ----------
    MODELS_URI: str = "/models/uri"
    VALIDATION_METRICS: str = "/validation/metrics"
    VALIDATION_ARTIFACTS: str = "/validation/artifacts"
    DATASET_REGISTER: str = "/dataset/register_by_api"
    DATASETS: str = "/datasets"
    LINK_DATASET_MODEL: str = "/link_dataset_model"
    MODELS: str = "/models"
    PIPELINE: str = "/pipeline"
    PIPELINE_RUNS: str = "/pipeline/runs"
    MODEL_RECOMMEND: str = "/models/recommend"
    COMPONENTS: str = "/training-builder-components"
    PROMETHEUS_DATASETS: str = "/datasets/prometheus"
    LOG_MODEL: str = "/models/log"

    # ---------- MLflow Behavior ----------
    SERIALIZATION_FORMAT: str = "cloudpickle"
    AWAIT_REGISTRATION_FOR: int = 300
    PYFUNC_PREDICT_FN: str = "predict"
    ENV_MANAGER: str = "local"
    MAX_RESULTS: int = 100
    CONTAINER_PORT: int = 8080

    # ---------- Testing / Database ----------
    SQLALCHEMY_TEST_DATABASE_URI: str = "sqlite:///test.db"
    TESTING_CONFIG: str = "config.app_config.TestingConfig"

    # ---------- DEX Authentication ----------
    POD_NAME: str = "dex-auth-0"
    NAMESPACE: str = "kubeflow"
    CONTAINER: str = "dex"
    CONFIG_PATH: str = "/etc/dex/config.docker.yaml"
    GRPC_PORT: int = 5557

    # ---------- Docker Images ----------
    COMP_BASE_IMAGE: str = "hiroregistry/cogflow:latest"
    FL_COGFLOW_BASE_IMAGE: str = "hiroregistry/flcogflow:latest"
    TRANSFORMER_BASE_IMAGE: str = "hiroregistry/k8-transformer:latest"
    FL_LINKS_BASE_IMAGE: str = "hiroregistry/cogflow_lite:latest"

    # ---------- Message Broker ----------
    MESSAGE_BROKER_DATASETS_URL: str = "/datasets"
    MESSAGE_BROKER_DATASETS_REGISTER: str = "/broker/register"
    MESSAGE_BROKER_TOPIC_REGISTER: str = "/topic/register"
    MESSAGE_BROKER_TOPIC_DATASETS_REGISTER: str = "/message/register"
    MESSAGE_BROKER_TOPIC_DATASETS_DETAILS: str = "/message/details"

    # ---------- Network / Retry ----------
    DEFAULT_TIMEOUT: int = 15
    RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_MIN: int = 2
    RETRY_BACKOFF_MAX: int = 10

    # ---------- Component Storage ----------
    COMPONENTS_BUCKET_NAME: str = "components"

    # ---------- Config behavior ----------
    class Config:
        """ "
        Configuration for Pydantic BaseSettings.
        """

        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# ----------------------------------------------------------------------
# 🔁 Cached settings loader
# ----------------------------------------------------------------------
@lru_cache
def get_settings() -> CogFlowSettings:
    """Load and cache CogFlow settings (env vars take precedence)."""
    return CogFlowSettings()


config = get_settings()

# ----------------------------------------------------------------------
# 🔊 Configure global logging
# ----------------------------------------------------------------------
logging.getLogger("cogflow").setLevel(
    getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
)
