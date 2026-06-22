"""Logging configuration using loguru."""

import os
import sys

from loguru import logger as _logger

logger = _logger


def configure_logging(log_level: str | None = None) -> None:
    """Configure application logging to console and rotating file.

    Args:
        log_level: Optional override for the log level. Defaults to ``LOG_LEVEL`` env var or INFO.
    """
    level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    data_dir = os.environ.get("VOICE_DATA_DIR", "data")
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    logger.add(
        os.path.join(log_dir, "app.log"),
        level=level,
        rotation="10 MB",
        retention="1 week",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
