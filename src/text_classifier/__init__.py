import logging
import sys
import os
from .constants import LOGS_DIR # Import LOGS_DIR

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)
log_filepath = os.path.join(LOGS_DIR, "text_classifier.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s: %(name)s: %(message)s]',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("textClassifierLogger")

# This ensures that PyTorch Lightning logs also go through this basicConfig
# logging.getLogger("pytorch_lightning").addHandler(logging.StreamHandler(sys.stdout))
# logging.getLogger("pytorch_lightning").setLevel(logging.INFO)