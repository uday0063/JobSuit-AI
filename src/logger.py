"""
Structured logging for the pipeline.
Outputs to both console and logs/pipeline.log.
"""
import os
import logging
from src.config import LOG_DIR, LOG_FILE, LOG_LEVEL


def get_logger(name: str = "pipeline") -> logging.Logger:
    """Get or create a named logger with file + console handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    os.makedirs(LOG_DIR, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
