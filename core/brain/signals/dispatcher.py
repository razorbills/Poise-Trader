# core/brain/signals/dispatcher.py
from utils.logger import get_logger
logger = get_logger("SignalDispatcher")

def send_signal(signal: dict):
    logger.info(f"[SIGNAL] {signal}")
