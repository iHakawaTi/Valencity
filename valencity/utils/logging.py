"""Structured logging utilities for valencity."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger for a valencity module.
    
    Args:
        name: Logger name (usually __name__ of the module)
        level: Optional logging level. Defaults to INFO.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"valencity.{name}")
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    
    return logger


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> None:
    """
    Configure global logging for valencity.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format_string: Custom format string for log messages
        date_format: Custom date format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Set level for all valencity loggers
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("valencity"):
            logging.getLogger(name).setLevel(level)
