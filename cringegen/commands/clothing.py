"""
Clothing command for cringegen CLI.

This module provides a command to generate clothing options for characters.
"""

import argparse
import logging
import random
from typing import Any, Dict, List, Optional

# Import centralized prompt components
from ..data.prompt_components import (
    get_clothing_by_type,
    get_random_clothing,
    CLOTHING
)

# Set up logging
logger = logging.getLogger(__name__)

def add_clothing_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the clothing command to the CLI.
    
    Args:
        subparsers: The subparsers object to add the command to
        parent_parser: The parent parser for common arguments
    """
    clothing_parser = subparsers.add_parser(
        "clothing", 
        help="Generate clothing options for characters",
        parents=[parent_parser]
    )
    
    # Command-specific arguments
    clothing_parser.add_argument(
        "--form",
        type=str,
        choices=["anthro", "feral"],
        default="anthro",
        help="Whether to generate anthro or feral clothing"
    )
    clothing_parser.add_argument(
        "--style",
        type=str,
        choices=["all"] + list(CLOTHING["anthro"].keys()),
        default="all",
        help="The style of clothing to generate"
    )
    clothing_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of clothing items to generate"
    )
    clothing_parser.add_argument(
        "--format",
        type=str,
        choices=["list", "comma", "json"],
        default="list",
        help="Output format for generated clothing"
    )
    
    # Set the handler function
    clothing_parser.set_defaults(func=handle_clothing_command)

def handle_clothing_command(args: Any) -> None:
    """Handle the clothing command.
    
    Args:
        args: The command arguments
    """
    # Generate clothing items
    clothing_items = get_random_clothing(
        form_type=args.form,
        style=args.style,
        count=args.count
    )
    
    # Output clothing in the requested format
    output_clothing(clothing_items, args.format, args.count)

def output_clothing(clothing_items: List[str], format_type: str, count: int) -> None:
    """Output clothing items in the requested format.
    
    Args:
        clothing_items: The clothing items to output
        format_type: The output format
        count: The number of items requested
    """
    if format_type == "list":
        for i, item in enumerate(clothing_items):
            logger.info(f"{i+1}. {item}")
    elif format_type == "comma":
        logger.info(", ".join(clothing_items))
    elif format_type == "json":
        import json
        logger.info(json.dumps(clothing_items))
    else:
        # Default to list format
        for i, item in enumerate(clothing_items):
            logger.info(f"{i+1}. {item}")
    
    logger.info(f"\nGenerated {len(clothing_items)} of {count} requested clothing items.") 