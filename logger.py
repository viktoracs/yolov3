import logging
import os
from datetime import datetime

# Create log directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Set log file path
if not hasattr(logging, "run_log_file"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"train_log_{timestamp}.txt")
    logging.run_log_file = log_file_path

# Log file size limit: 50 MB
MAX_LOG_SIZE = 0.05 * 1024 * 1024 * 1024  # bytes

class SizeCappedFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            if os.path.exists(self.baseFilename):
                if os.path.getsize(self.baseFilename) > MAX_LOG_SIZE:
                    return  # Skip logging if file is too large
        except Exception as e:
            print(f"[Logger Warning] Failed to check file size: {e}")
            return

        super().emit(record)

# Set up the logger
logger = logging.getLogger("yolov3_logger")

if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = SizeCappedFileHandler(
        filename=logging.run_log_file,
        mode='a',
        encoding='utf-8'
    )

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)




