"""
Backgrounds command for cringegen CLI.

This module provides a command to generate backgrounds for character scenes.
"""

import logging
import random
from typing import Any, List

# Import centralized prompt components
from ..data.prompt_components import (
    get_backgrounds_by_type,
    BACKGROUNDS
)

# Set up logging
logger = logging.getLogger(__name__)

def add_backgrounds_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the backgrounds command to the CLI.
    
    Args:
        subparsers: The subparsers object to add the command to
        parent_parser: The parent parser for common arguments
    """
    backgrounds_parser = subparsers.add_parser(
        "backgrounds", 
        help="Generate backgrounds for character scenes",
        parents=[parent_parser]
    )
    
    # Command-specific arguments
    backgrounds_parser.add_argument(
        "--environment",
        type=str,
        choices=["all"] + list(BACKGROUNDS.keys()),
        default="all",
        help="The environment type to generate backgrounds for"
    )
    backgrounds_parser.add_argument(
        "--subtype",
        type=str,
        default="all",
        help="The environment subtype to generate backgrounds for (use 'all' for all subtypes)"
    )
    backgrounds_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of backgrounds to generate"
    )
    backgrounds_parser.add_argument(
        "--format",
        type=str,
        choices=["list", "comma", "json"],
        default="list",
        help="Output format for generated backgrounds"
    )
    
    # Set the handler function
    backgrounds_parser.set_defaults(func=handle_backgrounds_command)

def handle_backgrounds_command(args: Any) -> None:
    """Handle the backgrounds command.
    
    Args:
        args: The command arguments
    """
    # Generate backgrounds
    backgrounds = generate_backgrounds(
        environment_type=args.environment,
        subtype=args.subtype,
        count=args.count
    )
    
    # Output backgrounds in the requested format
    output_backgrounds(backgrounds, args.format, args.count)

def generate_backgrounds(
    environment_type: str = "all",
    subtype: str = "all",
    count: int = 5
) -> List[str]:
    """Generate backgrounds for a specific environment type and subtype.
    
    Args:
        environment_type: The environment type
        subtype: The environment subtype
        count: How many backgrounds to generate
        
    Returns:
        A list of generated backgrounds
    """
    # Validate environment type
    if environment_type != "all" and environment_type not in BACKGROUNDS:
        logger.warning(f"Unknown environment type: {environment_type}, using all")
        environment_type = "all"
    
    # Get all available backgrounds for this environment type and subtype
    available_backgrounds = get_backgrounds_by_type(environment_type, subtype)
    
    # If no backgrounds available, use a fallback
    if not available_backgrounds:
        logger.warning(f"No backgrounds found for {environment_type}/{subtype}, using defaults")
        available_backgrounds = get_backgrounds_by_type("natural", "all")
        
    # If we still have no backgrounds, use a hardcoded fallback
    if not available_backgrounds:
        logger.warning("No backgrounds available in defaults, using fallback")
        available_backgrounds = ["forest", "beach", "mountain", "field"]
    
    # Generate the requested number of backgrounds
    result_backgrounds = []
    for _ in range(count):
        if available_backgrounds:
            # Try to avoid duplicates when possible
            if len(available_backgrounds) > len(result_backgrounds):
                # Get backgrounds not already selected
                remaining_backgrounds = [bg for bg in available_backgrounds if bg not in result_backgrounds]
                background = random.choice(remaining_backgrounds)
            else:
                background = random.choice(available_backgrounds)
            result_backgrounds.append(background)
    
    # Ensure we have at least one background
    if not result_backgrounds:
        result_backgrounds = ["forest clearing"]
        
    return result_backgrounds

def output_backgrounds(backgrounds: List[str], format_type: str, count: int) -> None:
    """Output backgrounds in the requested format.
    
    Args:
        backgrounds: The backgrounds to output
        format_type: The output format
        count: The number of backgrounds requested
    """
    if format_type == "list":
        for i, bg in enumerate(backgrounds):
            logger.info(f"{i+1}. {bg}")
    elif format_type == "comma":
        logger.info(", ".join(backgrounds))
    elif format_type == "json":
        import json
        logger.info(json.dumps(backgrounds))
    else:
        # Default to list format
        for i, bg in enumerate(backgrounds):
            logger.info(f"{i+1}. {bg}")
    
    logger.info(f"\nGenerated {len(backgrounds)} of {count} requested backgrounds.") 