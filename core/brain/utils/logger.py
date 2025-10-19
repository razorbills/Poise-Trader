# core/brain/utils/logger.py
import logging
from logging.handlers import TimedRotatingFileHandler
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "../../../logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name="PoiseTrader"):
    log_file = os.path.join(LOG_DIR, "poise.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Rotates logs daily, keeps 7 days
    handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=7)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())  # Also print to console

    return logger
