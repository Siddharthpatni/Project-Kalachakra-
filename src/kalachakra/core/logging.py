"""
Project Kalachakra — Structured Logging

Uses loguru for rich, structured logging with context propagation.
"""

import sys
from typing import Any

from loguru import logger

# Remove default handler
logger.remove()

# Add structured handler with rich formatting
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# File handler for experiment logs
logger.add(
    "logs/kalachakra_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
)


def get_logger(name: str, **kwargs: Any) -> Any:
    """Get a contextual logger bound with module name.

    Args:
        name: Module name (typically __name__)
        **kwargs: Additional context fields to bind

    Returns:
        Bound logger instance with module context
    """
    return logger.bind(module=name, **kwargs)
