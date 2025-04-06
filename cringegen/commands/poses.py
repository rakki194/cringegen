"""
Poses command for cringegen CLI.

This module provides a command to generate poses for furry characters.
"""

import argparse
import logging
import random
from typing import Any, Dict, List, Optional

# Import centralized prompt components
from ..data.prompt_components import (
    get_pose_by_taxonomy,
    get_random_pose,
    POSES
)

# Set up logging
logger = logging.getLogger(__name__)

def add_poses_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the poses command to the CLI.
    
    Args:
        subparsers: The subparsers object to add the command to
        parent_parser: The parent parser for common arguments
    """
    poses_parser = subparsers.add_parser(
        "poses", 
        help="Generate poses for furry characters",
        parents=[parent_parser]
    )
    
    # Command-specific arguments
    poses_parser.add_argument(
        "--taxonomy",
        type=str,
        choices=list(POSES.keys()),
        default="default",
        help="The species taxonomy to generate poses for"
    )
    poses_parser.add_argument(
        "--form",
        type=str,
        choices=["anthro", "feral"],
        default="anthro",
        help="Whether to generate anthro or feral poses"
    )
    poses_parser.add_argument(
        "--category",
        type=str,
        choices=["all", "neutral", "intimate", "action", "specific"],
        default="all",
        help="The category of poses to generate"
    )
    poses_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of poses to generate"
    )
    poses_parser.add_argument(
        "--format",
        type=str,
        choices=["list", "comma", "json"],
        default="list",
        help="Output format for generated poses"
    )
    
    # Set the handler function
    poses_parser.set_defaults(func=handle_poses_command)


def handle_poses_command(args: Any) -> None:
    """Handle the poses command.
    
    Args:
        args: The command arguments
    """
    # Generate poses
    poses = generate_poses(
        taxonomy=args.taxonomy,
        form_type=args.form,
        category=args.category,
        count=args.count
    )
    
    # Output poses in the requested format
    output_poses(poses, args.format, args.count)


def generate_poses(
    taxonomy: str,
    form_type: str, 
    category: str = "all", 
    count: int = 1
) -> List[str]:
    """Generate poses for a specific species taxonomy and form type.
    
    Args:
        taxonomy: The species taxonomy ("canine", "feline", etc.)
        form_type: The form type ("anthro" or "feral")
        category: The pose category
        count: How many poses to generate
        
    Returns:
        A list of generated poses
    """
    # Validate taxonomy
    if taxonomy not in POSES:
        logger.warning(f"Unknown taxonomy: {taxonomy}, using default")
        taxonomy = "default"
    
    # Get all available poses for this taxonomy and form type
    available_poses = get_pose_by_taxonomy(taxonomy, form_type, category)
    
    # If no poses available, use a fallback
    if not available_poses:
        logger.warning(f"No poses found for {taxonomy}/{form_type}/{category}, using defaults")
        available_poses = get_pose_by_taxonomy("default", form_type, "neutral")
        
    # If we still have no poses, use a hardcoded fallback
    if not available_poses:
        logger.warning("No poses available in defaults, using fallback")
        available_poses = ["standing", "sitting", "walking"]
    
    # Generate the requested number of poses
    result_poses = []
    for _ in range(count):
        if available_poses:
            pose = random.choice(available_poses)
            result_poses.append(pose)
    
    # Ensure we have at least one pose
    if not result_poses:
        result_poses = ["standing"]
        
    return result_poses


def output_poses(poses: List[str], format_type: str, count: int) -> None:
    """Output poses in the requested format.
    
    Args:
        poses: The poses to output
        format_type: The output format
        count: The number of poses requested
    """
    if format_type == "list":
        for i, pose in enumerate(poses):
            logger.info(f"{i+1}. {pose}")
    elif format_type == "comma":
        logger.info(", ".join(poses))
    elif format_type == "json":
        import json
        logger.info(json.dumps(poses))
    else:
        # Default to list format
        for i, pose in enumerate(poses):
            logger.info(f"{i+1}. {pose}")
    
    logger.info(f"\nGenerated {len(poses)} of {count} requested poses.") 