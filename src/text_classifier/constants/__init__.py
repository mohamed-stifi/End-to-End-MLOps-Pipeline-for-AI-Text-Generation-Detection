import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
print(PROJECT_ROOT)
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Model constants
MODEL_TYPES = ["lstm", "bert", "roberta"]
MAX_LENGTH = 512
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10

# Data constants
TARGET_COLUMN = "generated"
TEXT_COLUMN = "text"
TEST_SIZE = 0.2
VAL_SIZE = 0.1


# API constants (can be overridden by environment variables in api/main.py if needed)
API_HOST = "0.0.0.0"  # Default host for Uvicorn
API_PORT = 8000       # Default port for Uvicorn


# Kafka Constants (can be overridden by environment variables)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TASKS_TOPIC = "mlops_tasks"
KAFKA_TASK_STATUS_TOPIC = "mlops_task_status" # Optional: for reporting task status