"""
Logging utilities for cringegen.

This module provides standardized logging functionality for all components of the cringegen package.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Import model detection functions from the dedicated module
from .model_detection import (
    is_sdxl_model,
    is_sd15_model,
    is_sd2_model,
    is_sd35_model,
    is_flux_model,
    is_stable_cascade_model,
    is_ltx_model,
    is_lumina_model,
    detect_model_architecture,
    get_model_info,
    get_sd35_variant,
    test_model_detection,
    is_optimal_resolution,
    get_optimal_resolution,
    get_optimal_resolution_suggestions
)

# Module exports
__all__ = [
    "get_logger", 
    "configure_logging", 
    "set_log_level", 
    "print_colored_warning",
    "print_colored_info",
    "is_sdxl_model",
    "is_sd15_model",
    "is_sd2_model",
    "is_sd35_model",
    "is_flux_model",
    "is_stable_cascade_model",
    "is_ltx_model",
    "is_lumina_model",
    "detect_model_architecture",
    "test_model_detection",
    "is_optimal_resolution",
    "get_optimal_resolution",
    "get_optimal_resolution_suggestions",
    "get_model_info"
]

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_LOG_FORMAT = "%(levelname)s: %(message)s"

# Global logger dictionary to avoid creating multiple loggers for the same name
_LOGGERS: Dict[str, logging.Logger] = {}

# Environment variable to control default log level
LOG_LEVEL_ENV_VAR = "CRINGEGEN_LOG_LEVEL"

# Add these constants for ANSI colors
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def get_logger(name: str = "cringegen") -> logging.Logger:
    """Get a named logger.

    Args:
        name: Name for the logger. If not provided, uses 'cringegen'.

    Returns:
        Logger instance
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)

    # Set default level from environment variable if present
    env_level = os.environ.get(LOG_LEVEL_ENV_VAR)
    if env_level:
        try:
            logger.setLevel(getattr(logging, env_level.upper()))
        except (AttributeError, TypeError):
            logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.INFO)

    _LOGGERS[name] = logger
    return logger


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    console_level: Optional[Union[int, str]] = None,
    file_level: Optional[Union[int, str]] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    console_format: Optional[str] = None,
    file_format: Optional[str] = None,
    propagate: bool = False,
) -> None:
    """Configure logging for all cringegen loggers.

    Args:
        level: Default log level for all handlers
        log_file: Path to log file (if None, file logging is disabled)
        console: Whether to log to console
        console_level: Log level for console (if None, uses default level)
        file_level: Log level for file (if None, uses default level)
        log_format: Default log format for all handlers
        console_format: Log format for console (if None, uses default format)
        file_format: Log format for file (if None, uses default format)
        propagate: Whether to propagate messages to parent loggers
    """
    root_logger = logging.getLogger("cringegen")

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Clean up any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set the default level
    root_logger.setLevel(level)
    root_logger.propagate = propagate

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(console_format or log_format))
        if console_level:
            if isinstance(console_level, str):
                console_level = getattr(logging, console_level.upper())
            console_handler.setLevel(console_level)
        else:
            console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(file_format or log_format))
        if file_level:
            if isinstance(file_level, str):
                file_level = getattr(logging, file_level.upper())
            file_handler.setLevel(file_level)
        else:
            file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


def configure_cli_logging(args: Any) -> None:
    """Configure logging based on CLI arguments.

    This function configures logging based on CLI arguments.

    Args:
        args: Parsed command-line arguments
    """
    # If --log-level DEBUG is set, also set --debug
    if hasattr(args, "log_level") and str(args.log_level).upper() == "DEBUG":
        if hasattr(args, "debug"):
            args.debug = True  # --log-level DEBUG implies --debug

    # Determine log level based on args
    log_level = logging.INFO

    # If args has log_level attribute, use it to set the log level
    if hasattr(args, "log_level") and args.log_level:
        log_level = getattr(logging, args.log_level)
    # Otherwise, check debug and verbose flags
    elif hasattr(args, "debug") and args.debug:
        log_level = logging.DEBUG
    elif hasattr(args, "verbose") and args.verbose:
        log_level = logging.INFO

    # Configure with console output and optional file output
    log_file = os.environ.get("CRINGEGEN_LOG_FILE")
    if hasattr(args, "log_file") and args.log_file:
        log_file = args.log_file

    configure_logging(
        level=log_level,
        log_file=log_file,
        console=True,
        console_format=SIMPLE_LOG_FORMAT if log_level != logging.DEBUG else DEFAULT_LOG_FORMAT,
    )


def set_log_level(level: Union[int, str]) -> None:
    """Set the log level for all cringegen loggers.

    Args:
        level: Log level to set
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    root_logger = logging.getLogger("cringegen")
    root_logger.setLevel(level)

    # Also update all existing handlers
    for handler in root_logger.handlers:
        handler.setLevel(level)


def print_colored_warning(message: str, color: str = YELLOW) -> None:
    """Print a colored warning message to stderr.
    
    Args:
        message: The warning message to print
        color: ANSI color code to use
    """
    print(f"{color}{message}{RESET}", file=sys.stderr)


def print_colored_info(message: str, color: str = GREEN) -> None:
    """Print a colored info message to stdout.
    
    Args:
        message: The info message to print
        color: ANSI color code to use
    """
    print(f"{color}{message}{RESET}")
