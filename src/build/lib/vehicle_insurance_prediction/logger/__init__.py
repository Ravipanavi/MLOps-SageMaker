import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3

# Always create logs/ inside the project directory
PROJECT_ROOT = os.getcwd()
log_dir_path = os.path.join(PROJECT_ROOT, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)

log_file_path = os.path.join(log_dir_path, LOG_FILE)

# Configure logger ON IMPORT
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

file_handler = RotatingFileHandler(
    log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Add handlers ONLY ONCE
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Expose logging module
logging = logger
