"""
Accessories command for cringegen CLI.

This module provides a command to generate accessories for characters.
"""

import logging
from typing import Any, List

# Import centralized prompt components
from ..data.prompt_components import (
    get_random_accessories,
    ACCESSORIES
)

# Set up logging
logger = logging.getLogger(__name__)

def add_accessories_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the accessories command to the CLI.
    
    Args:
        subparsers: The subparsers object to add the command to
        parent_parser: The parent parser for common arguments
    """
    accessories_parser = subparsers.add_parser(
        "accessories", 
        help="Generate accessories for characters",
        parents=[parent_parser]
    )
    
    # Command-specific arguments
    accessories_parser.add_argument(
        "--type",
        type=str,
        choices=["all"] + list(ACCESSORIES.keys()),
        default="all",
        help="The type of accessories to generate"
    )
    accessories_parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of accessories to generate"
    )
    accessories_parser.add_argument(
        "--format",
        type=str,
        choices=["list", "comma", "json"],
        default="list",
        help="Output format for generated accessories"
    )
    
    # Set the handler function
    accessories_parser.set_defaults(func=handle_accessories_command)

def handle_accessories_command(args: Any) -> None:
    """Handle the accessories command.
    
    Args:
        args: The command arguments
    """
    # Generate accessories
    accessories = get_random_accessories(
        accessory_type=args.type,
        count=args.count
    )
    
    # Output accessories in the requested format
    output_accessories(accessories, args.format, args.count)

def output_accessories(accessories: List[str], format_type: str, count: int) -> None:
    """Output accessories in the requested format.
    
    Args:
        accessories: The accessories to output
        format_type: The output format
        count: The number of items requested
    """
    if format_type == "list":
        for i, item in enumerate(accessories):
            logger.info(f"{i+1}. {item}")
    elif format_type == "comma":
        logger.info(", ".join(accessories))
    elif format_type == "json":
        import json
        logger.info(json.dumps(accessories))
    else:
        # Default to list format
        for i, item in enumerate(accessories):
            logger.info(f"{i+1}. {item}")
    
    logger.info(f"\nGenerated {len(accessories)} of {count} requested accessories.") 