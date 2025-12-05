"""Logging helpers for the Google interview scheduling pipeline."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from ..config.settings import Settings

DEFAULT_LOG_FILE = "scheduler.log"


def _clear_existing_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def setup_logging(settings: Settings, *, console: bool = True, log_file: Optional[str] = None) -> None:
    """Configure root logging handlers according to the provided settings."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    _clear_existing_handlers(root_logger)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    target_file = settings.log_directory / (log_file or DEFAULT_LOG_FILE)
    file_handler = RotatingFileHandler(target_file, maxBytes=2_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root_logger.addHandler(stream_handler)

    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
