"""
Centralized logging configuration for SpinnyBall.

Provides consistent logging format and configuration across all modules.
"""

import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    format_string: str | None = None,
) -> None:
    """
    Setup centralized logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Configure root logger (only if not already configured)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    # Add file handler if specified (only if not already added)
    if log_file:
        handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == log_file
            for h in root_logger.handlers
        )
        if not handler_exists:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_string))
            root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
