"""
Centralized logging utility using Loguru.

Design Decision:
    - Loguru over standard logging because:
        * Structured JSON output for production log shipping
        * Automatic traceback capture with variables
        * Simple sink configuration (file rotation, compression)
    - Single setup_logging() call from main entry points.
    - get_logger() used in every module.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    rotation: str = "100 MB",
    retention: str = "7 days",
    serialize: bool = False,
) -> None:
    """
    Configure Loguru logger for the entire application.

    Args:
        log_level:  Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file:   Optional path to write rotating log file.
        rotation:   When to rotate the log file.
        retention:  How long to keep old log files.
        serialize:  If True, output JSON-formatted logs (for prod).
    """
    logger.remove()  # Remove default stderr handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=fmt if not serialize else "{message}",
        level=log_level,
        colorize=not serialize,
        serialize=serialize,
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            rotation=rotation,
            retention=retention,
            level=log_level,
            serialize=serialize,
            compression="gz",
        )

    logger.info(f"Logging configured | level={log_level} | file={log_file}")


def get_logger(name: str):
    """
    Return a bound Loguru logger with module name context.

    Usage:
        log = get_logger(__name__)
        log.info("Processing frame")
    """
    return logger.bind(module=name)
