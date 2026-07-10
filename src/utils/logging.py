"""
src/utils/logging.py

Centralised logging configuration for the Adaptive-IPI project.

Provides a consistent logging setup with both console and file output.
All modules should call ``get_logger(__name__)`` to obtain their logger.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Configure the root logger with console and optional file handlers.

    This function is idempotent — calling it multiple times will not
    add duplicate handlers.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a log file. If provided, logs are
            written to both console and file.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Obtain a logger for the given module name.

    If ``setup_logging()`` has not been called yet, a basic console
    configuration is applied automatically.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A configured Logger instance.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
