"""
Logging utilities for cringegen.

This module provides standardized logging functionality for all components of the cringegen package.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import re

# Module exports
__all__ = [
    "get_logger", 
    "configure_logging", 
    "set_log_level", 
    "print_colored_warning",
    "is_sdxl_model",
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


def is_sdxl_model(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is an SDXL model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be an SDXL model
    """
    sdxl_patterns = [
        "sdxl", 
        "sd-xl", 
        "sd_xl",
        "stableDiffusionXL",
        "stable-diffusion-xl",
        "pixartsigma"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    # Special case for NoobAI models - specifically check for XL in the name
    if ("noob" in checkpoint_lower or "noobai" in checkpoint_lower) and ("xl" in checkpoint_lower or "XL" in checkpoint_name):
        return True
        
    return any(pattern in checkpoint_lower for pattern in sdxl_patterns)


def get_model_info(checkpoint_name: str) -> tuple[str, str]:
    """Detect model architecture and family from checkpoint name.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        Tuple of (architecture_type, model_family)
        Where architecture_type is one of: "sdxl", "sd15", "sd35", "flux", "ltx", "lumina", etc.
        And model_family is the specific model series like "noob", "yiffymix", "dreamshaper", etc.
    """
    checkpoint_lower = checkpoint_name.lower()
    
    # First detect architecture
    architecture = "unknown"
    
    # SDXL detection - more comprehensive patterns
    xl_pattern = re.compile(r'(xl|XL)')
    
    # Special case for NoobAI models - they are SDXL
    if ("noob" in checkpoint_lower or "noobai" in checkpoint_lower):
        if "xl" in checkpoint_lower or "XL" in checkpoint_name:
            architecture = "sdxl"
        else:
            # Assume SD1.5 for non-XL NoobAI models
            architecture = "sd15"
    
    # Other SDXL detection patterns
    elif any(pattern in checkpoint_lower for pattern in [
        "sdxl", "sd-xl", "sd_xl", "stablediffusionxl", "stable-diffusion-xl"
    ]) or xl_pattern.search(checkpoint_name):
        architecture = "sdxl"
    
    # SD3.5 detection
    elif any(pattern in checkpoint_lower for pattern in [
        "sd3.5", "sd-3.5", "sd_3.5", "sd35"
    ]):
        architecture = "sd35"
    
    # SD1.5 detection (baseline models)
    elif any(pattern in checkpoint_lower for pattern in [
        "sd1.5", "sd-1.5", "sd_1.5", "sd15"
    ]) or "yiffymix" in checkpoint_lower:
        architecture = "sd15"
    
    # Flux detection
    elif "chroma-unlocked" in checkpoint_lower:
        architecture = "flux"
    
    # LTX detection
    elif "ltx" in checkpoint_lower:
        architecture = "ltx"
    
    # Lumina detection
    elif "lumina" in checkpoint_lower:
        architecture = "lumina"
    
    # Now detect model family
    family = "unknown"
    
    # Common model families
    family_patterns = {
        "animagine": ["animagine"],
        "dreamshaper": ["dreamshaper"],
        "epicrealism": ["epicrealism"],
        "illustrious": ["illustrious"],
        "juggernaut": ["juggernaut"],
        "noob": ["noob", "realnoob", "noobai"],
        "pony": ["pony"],
        "zavychroma": ["zavychroma"],
        "chroma": ["chroma"],
        "yiffymix": ["yiffymix"],
        "pixartsigma": ["pixartsigma"]
    }
    
    for family_name, patterns in family_patterns.items():
        if any(pattern in checkpoint_lower for pattern in patterns):
            family = family_name
            break
    
    # Handle special cases for SD3.5
    if architecture == "sd35":
        if "large" in checkpoint_lower:
            family = "sd35l"
        elif "medium" in checkpoint_lower:
            if "turbo" in checkpoint_lower:
                family = "sd35mt"
            else:
                family = "sd35m"
    
    # Handle vanilla SDXL case
    if architecture == "sdxl" and family == "unknown":
        if any(pattern in checkpoint_lower for pattern in ["base", "1.0"]):
            family = "sdxl"
    
    # Handle LTX and Lumina where architecture = family
    if architecture in ["ltx", "lumina"] and family == "unknown":
        family = architecture
        
    # Fix for SD3.5 turbo model
    if checkpoint_lower == "sd3.5m_turbo.safetensors":
        family = "sd35mt"
    
    return architecture, family


def is_optimal_resolution(width: int, height: int, model_type: str, use_deepshrink: bool = False) -> bool:
    """Check if dimensions are optimal for the given model type.
    
    Args:
        width: Image width
        height: Image height
        model_type: "sdxl" or "sd15" or other model type
        use_deepshrink: Whether DeepShrink is enabled (bypasses check if True)
        
    Returns:
        True if the resolution is optimal for the model type or DeepShrink is enabled
    """
    # Skip resolution check when DeepShrink is enabled
    if use_deepshrink:
        return True
        
    # Calculate total pixels
    total_pixels = width * height
    
    if model_type.lower() == "sdxl":
        # SDXL optimal: 1024×1024 (1,048,576 pixels)
        # Allow 5% margin for different aspect ratios
        target_pixels = 1048576
        margin = target_pixels * 0.05
        return abs(total_pixels - target_pixels) <= margin
    elif model_type.lower() == "sd15":
        # SD 1.5 optimal: 512×512 (262,144 pixels)
        # Allow 5% margin for different aspect ratios
        target_pixels = 262144
        margin = target_pixels * 0.05
        return abs(total_pixels - target_pixels) <= margin
    else:
        # For other models, assume any resolution is fine
        return True


def get_optimal_resolution(aspect_ratio: float, model_type: str) -> tuple:
    """Calculate optimal resolution for a given aspect ratio and model type.
    
    Args:
        aspect_ratio: Width to height ratio (width/height)
        model_type: "sdxl" or "sd15" or other model type
        
    Returns:
        Tuple of (width, height) that is optimal for the model type and aspect ratio
    """
    # First, determine the target pixel count
    if model_type.lower() == "sdxl":
        target_pixels = 1048576  # 1024×1024
    elif model_type.lower() == "sd15":
        target_pixels = 262144   # 512×512
    else:
        # Default to SD1.5 for unknown models
        target_pixels = 262144
    
    # Calculate dimensions based on aspect ratio and target pixels
    width = int((target_pixels * aspect_ratio) ** 0.5)
    height = int(width / aspect_ratio)
    
    # Ensure dimensions are divisible by 8 (required by Stable Diffusion)
    width = width - (width % 8)
    height = height - (height % 8)
    
    return (width, height)


def get_optimal_resolution_suggestions(width: int, height: int, model_type: str) -> list:
    """Get optimal resolution suggestions for a given aspect ratio and model type.
    
    Args:
        width: Current width
        height: Current height
        model_type: "sdxl" or "sd15" or other model type
        
    Returns:
        List of tuple suggestions (width, height) that are optimal
    """
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Get the base optimal resolution for this aspect ratio
    base_optimal = get_optimal_resolution(aspect_ratio, model_type)
    
    # Get some common variations with the same aspect ratio
    suggestions = [base_optimal]
    
    # Add common predefined resolutions that maintain the approximate pixel count
    if model_type.lower() == "sdxl":
        if abs(aspect_ratio - 1.0) < 0.01:  # Square (1:1)
            suggestions = [(1024, 1024)]
        elif abs(aspect_ratio - (4/3)) < 0.01:  # 4:3
            suggestions = [(1184, 888)]
        elif abs(aspect_ratio - (3/4)) < 0.01:  # 3:4
            suggestions = [(888, 1184)]
        elif abs(aspect_ratio - (16/9)) < 0.01:  # 16:9
            suggestions = [(1360, 768)]
        elif abs(aspect_ratio - (9/16)) < 0.01:  # 9:16
            suggestions = [(768, 1360)]
        elif abs(aspect_ratio - (3/2)) < 0.01:  # 3:2
            suggestions = [(1248, 832)]
        elif abs(aspect_ratio - (2/3)) < 0.01:  # 2:3
            suggestions = [(832, 1248)]
        
        # Add common SDXL presets regardless of the input aspect ratio
        if len(suggestions) == 1:  # Only the base suggestion was added
            if aspect_ratio >= 1.0:  # Landscape
                suggestions.extend([
                    (1024, 1024),  # 1:1
                    (1152, 896),   # 9:7
                    (1216, 832),   # 19:13
                    (1344, 768),   # 7:4
                    (1536, 640),   # 12:5
                ])
            else:  # Portrait
                suggestions.extend([
                    (1024, 1024),  # 1:1
                    (896, 1152),   # 7:9
                    (832, 1216),   # 13:19
                    (768, 1344),   # 4:7
                    (640, 1536),   # 5:12
                ])
    
    elif model_type.lower() == "sd15":
        if abs(aspect_ratio - 1.0) < 0.01:  # Square (1:1)
            suggestions = [(512, 512)]
        elif abs(aspect_ratio - (4/3)) < 0.01:  # 4:3
            suggestions = [(592, 444)]
        elif abs(aspect_ratio - (3/4)) < 0.01:  # 3:4
            suggestions = [(444, 592)]
        elif abs(aspect_ratio - (16/9)) < 0.01:  # 16:9
            suggestions = [(680, 384)]
        elif abs(aspect_ratio - (9/16)) < 0.01:  # 9:16
            suggestions = [(384, 680)]
        elif abs(aspect_ratio - (3/2)) < 0.01:  # 3:2
            suggestions = [(624, 416)]
        elif abs(aspect_ratio - (2/3)) < 0.01:  # 2:3
            suggestions = [(416, 624)]
        
        # Add common SD1.5 presets regardless of the input aspect ratio
        if len(suggestions) == 1:  # Only the base suggestion was added
            if aspect_ratio >= 1.0:  # Landscape
                suggestions.extend([
                    (512, 512),    # 1:1
                    (576, 448),    # 9:7
                    (608, 416),    # 19:13
                    (672, 384),    # 7:4
                ])
            else:  # Portrait
                suggestions.extend([
                    (512, 512),    # 1:1
                    (448, 576),    # 7:9
                    (416, 608),    # 13:19
                    (384, 672),    # 4:7
                ])
    
    # Ensure all suggestions are unique
    unique_suggestions = []
    for w, h in suggestions:
        if (w, h) not in unique_suggestions:
            unique_suggestions.append((w, h))
    
    return unique_suggestions
