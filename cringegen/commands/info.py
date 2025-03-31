"""
Information and listing commands for CringeGen
"""

import logging
import os
from typing import Dict, List, Optional

from ..utils.comfy_api import (
    check_comfy_server,
    get_available_checkpoints,
    get_available_loras,
    get_available_samplers,
    get_available_schedulers,
    get_compatible_checkpoints,
    get_compatible_loras,
    get_lora_directory
)
from ..utils.file_utils import (
    copy_latest_images_from_comfyui,
    rsync_latest_images_from_comfyui,
    open_images_with_imv
)

logger = logging.getLogger(__name__)


def add_info_commands(subparsers, parent_parser):
    """Add information and listing commands to the CLI"""
    # Add list-samplers command
    list_samplers_parser = subparsers.add_parser(
        "list-samplers", help="List available samplers", parents=[parent_parser]
    )
    list_samplers_parser.set_defaults(func=list_samplers)

    # Add list-schedulers command
    list_schedulers_parser = subparsers.add_parser(
        "list-schedulers", help="List available schedulers", parents=[parent_parser]
    )
    list_schedulers_parser.set_defaults(func=list_schedulers)

    # Add list-checkpoints command
    list_checkpoints_parser = subparsers.add_parser(
        "list-checkpoints", help="List available checkpoints", parents=[parent_parser]
    )
    list_checkpoints_parser.set_defaults(func=list_checkpoints)

    # Add list-loras command
    list_loras_parser = subparsers.add_parser(
        "list-loras", help="List available LoRAs", parents=[parent_parser]
    )
    list_loras_parser.add_argument("--pattern", type=str, help="Pattern to filter LoRAs by name")
    list_loras_parser.add_argument(
        "--show-categories", 
        action="store_true", 
        help="Show the category of each LoRA (style, character, concept, kink)"
    )
    list_loras_parser.add_argument(
        "--category", 
        type=str, 
        choices=["style", "character", "concept", "kink", "all"],
        help="Filter LoRAs by category"
    )
    list_loras_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable metadata caching (slower but avoids potential issues)"
    )
    list_loras_parser.set_defaults(func=list_loras)

    # Add list-compatible-loras command
    list_compatible_loras_parser = subparsers.add_parser(
        "list-compatible-loras",
        help="List LoRAs compatible with a checkpoint",
        parents=[parent_parser],
    )
    list_compatible_loras_parser.add_argument(
        "checkpoint", type=str, help="Checkpoint to check compatibility with"
    )
    list_compatible_loras_parser.set_defaults(func=list_compatible_loras)

    # Add list-compatible-checkpoints command
    list_compatible_checkpoints_parser = subparsers.add_parser(
        "list-compatible-checkpoints",
        help="List checkpoints compatible with a LoRA",
        parents=[parent_parser],
    )
    list_compatible_checkpoints_parser.add_argument(
        "lora", type=str, help="LoRA to check compatibility with"
    )
    list_compatible_checkpoints_parser.set_defaults(func=list_compatible_checkpoints)

    # Add suggest-checkpoint command
    suggest_checkpoint_parser = subparsers.add_parser(
        "suggest-checkpoint", help="Suggest a checkpoint for a LoRA", parents=[parent_parser]
    )
    suggest_checkpoint_parser.add_argument(
        "lora", type=str, help="LoRA to suggest a checkpoint for"
    )
    suggest_checkpoint_parser.set_defaults(func=suggest_checkpoint)

    # Add suggest-loras command
    suggest_loras_parser = subparsers.add_parser(
        "suggest-loras", help="Suggest LoRAs for a checkpoint", parents=[parent_parser]
    )
    suggest_loras_parser.add_argument(
        "checkpoint", type=str, help="Checkpoint to suggest LoRAs for"
    )
    suggest_loras_parser.add_argument(
        "--count", type=int, default=5, help="Number of LoRAs to suggest"
    )
    suggest_loras_parser.set_defaults(func=suggest_loras)

    # Add copy-images command
    copy_images_parser = subparsers.add_parser(
        "copy-images",
        help="Copy latest images from ComfyUI output directory",
        parents=[parent_parser],
    )
    copy_images_parser.add_argument("--count", type=int, default=1, help="Number of images to copy")
    copy_images_parser.add_argument(
        "--comfy-output-dir",
        type=str,
        default="/home/kade/toolkit/diffusion/comfy/ComfyUI/output",
        help="ComfyUI output directory",
    )
    copy_images_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for copied images",
    )
    # Remote ComfyUI options
    copy_images_parser.add_argument(
        "--remote",
        action="store_true",
        help="Use SSH to copy images from a remote ComfyUI instance",
    )
    copy_images_parser.add_argument(
        "--ssh-host",
        type=str,
        default="otter_den",
        help="SSH hostname or IP address for remote ComfyUI instance",
    )
    copy_images_parser.add_argument(
        "--ssh-port",
        type=int,
        default=1487,
        help="SSH port for remote ComfyUI instance",
    )
    copy_images_parser.add_argument(
        "--ssh-user",
        type=str,
        default="kade",
        help="SSH username for remote ComfyUI instance",
    )
    copy_images_parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key file for remote ComfyUI instance",
    )
    copy_images_parser.set_defaults(func=copy_images_from_comfyui_cmd)

    return subparsers


def list_samplers(args):
    """List available samplers"""
    samplers = get_available_samplers()
    if samplers:
        logger.info("Available samplers:")
        for sampler in sorted(samplers):
            logger.info(f"  - {sampler}")
    else:
        logger.warning("No samplers found or ComfyUI is not running.")


def list_schedulers(args):
    """List available schedulers"""
    schedulers = get_available_schedulers()
    if schedulers:
        logger.info("Available schedulers:")
        for scheduler in sorted(schedulers):
            logger.info(f"  - {scheduler}")
    else:
        logger.warning("No schedulers found or ComfyUI is not running.")


def list_checkpoints(args):
    """List available checkpoints"""
    checkpoints = get_available_checkpoints()
    if checkpoints:
        logger.info("Available checkpoints:")
        for checkpoint in sorted(checkpoints):
            logger.info(f"  - {checkpoint}")
    else:
        logger.warning("No checkpoints found or ComfyUI is not running.")


def list_loras(args):
    """List available LoRAs"""
    from ..utils.lora_metadata.analyzer import analyze_lora_type, analyze_multiple_loras
    from ..utils.comfy_api import get_lora_directory
    
    loras = get_available_loras()
    if loras:
        # Apply pattern filter if specified
        if args.pattern:
            # Filter LoRAs by pattern
            filtered_loras = [lora for lora in loras if args.pattern.lower() in lora.lower()]
        else:
            filtered_loras = loras
        
        # Show categories if requested
        if args.show_categories or args.category:
            # Get the LoRA directory
            lora_dir = get_lora_directory()
            
            # Create a dictionary to store LoRA categories
            categorized_loras = {}
            
            # Analyze each LoRA to determine its category
            for lora in filtered_loras:
                try:
                    # Get the full path to the LoRA
                    if not lora.endswith(".safetensors"):
                        lora_path = os.path.join(lora_dir, f"{lora}.safetensors")
                    else:
                        lora_path = os.path.join(lora_dir, lora)
                    
                    # Analyze the LoRA to determine its type
                    analysis = analyze_lora_type(lora_path, use_cache=not args.no_cache)
                    category = analysis.get("type", "unknown")
                    confidence = analysis.get("confidence", 0.0)
                    
                    # Store the category and confidence
                    categorized_loras[lora] = (category, confidence)
                except Exception as e:
                    logger.warning(f"Error analyzing LoRA {lora}: {e}")
                    categorized_loras[lora] = ("unknown", 0.0)
            
            # Filter by category if specified
            if args.category and args.category != "all":
                # Keep only LoRAs of the specified category
                filtered_loras = [
                    lora for lora in filtered_loras 
                    if lora in categorized_loras and categorized_loras[lora][0] == args.category
                ]
            
            # Group LoRAs by category
            loras_by_category = {
                "style": [],
                "character": [],
                "concept": [],
                "kink": [],
                "unknown": []
            }
            
            for lora in filtered_loras:
                if lora in categorized_loras:
                    category, _ = categorized_loras[lora]
                    loras_by_category[category].append(lora)
            
            # Display LoRAs grouped by category
            logger.info("Available LoRAs by category:")
            
            for category, category_loras in loras_by_category.items():
                if category_loras:
                    logger.info(f"\n{category.upper()} LoRAs ({len(category_loras)}):")
                    for lora in sorted(category_loras):
                        confidence = categorized_loras[lora][1]
                        logger.info(f"  - {lora} (confidence: {confidence:.2f})")
            
            total_count = len(filtered_loras)
            logger.info(f"\nFound {total_count} LoRAs matching criteria.")
        else:
            # Regular display without categories
            logger.info("Available LoRAs:")
            for lora in sorted(filtered_loras):
                logger.info(f"  - {lora}")
            logger.info(f"Found {len(filtered_loras)} LoRAs.")
    else:
        logger.warning("No LoRAs found or ComfyUI is not running.")


def list_compatible_loras(args):
    """List LoRAs compatible with a checkpoint"""
    compatible_loras = get_compatible_loras(args.checkpoint)
    if compatible_loras:
        logger.info(f"LoRAs compatible with checkpoint '{args.checkpoint}':")
        for lora in sorted(compatible_loras):
            logger.info(f"  - {lora}")
        logger.info(f"Found {len(compatible_loras)} compatible LoRAs.")
    else:
        logger.warning(
            f"No compatible LoRAs found for checkpoint '{args.checkpoint}' or ComfyUI is not running."
        )


def list_compatible_checkpoints(args):
    """List checkpoints compatible with a LoRA"""
    compatible_checkpoints = get_compatible_checkpoints(args.lora)
    if compatible_checkpoints:
        logger.info(f"Checkpoints compatible with LoRA '{args.lora}':")
        for checkpoint in sorted(compatible_checkpoints):
            logger.info(f"  - {checkpoint}")
        logger.info(f"Found {len(compatible_checkpoints)} compatible checkpoints.")
    else:
        logger.warning(
            f"No compatible checkpoints found for LoRA '{args.lora}' or ComfyUI is not running."
        )


def suggest_checkpoint(args):
    """Suggest a checkpoint for a LoRA"""
    checkpoint = get_checkpoint_suggestion(args.lora)
    if checkpoint:
        logger.info(f"Suggested checkpoint for LoRA '{args.lora}':")
        logger.info(f"  - {checkpoint}")
    else:
        logger.warning(
            f"No checkpoint suggestion available for LoRA '{args.lora}' or ComfyUI is not running."
        )


def suggest_loras(args):
    """Suggest LoRAs for a checkpoint"""
    loras = get_lora_suggestions(args.checkpoint, args.count)
    if loras:
        logger.info(f"Suggested LoRAs for checkpoint '{args.checkpoint}':")
        for lora in loras:
            logger.info(f"  - {lora}")
    else:
        logger.warning(
            f"No LoRA suggestions available for checkpoint '{args.checkpoint}' or ComfyUI is not running."
        )


def copy_images_from_comfyui_cmd(args):
    """Copy latest images from ComfyUI output directory"""
    if args.remote:
        # Check for required SSH parameters
        if not args.ssh_host:
            logger.error("SSH host is required when using --remote")
            return
        
        logger.info(f"Using rsync over SSH to copy images from {args.ssh_host}")
        copied = rsync_latest_images_from_comfyui(
            args.ssh_host,
            args.comfy_output_dir,
            args.output_dir,
            limit=args.count,
            ssh_port=args.ssh_port,
            ssh_user=args.ssh_user,
            ssh_key=args.ssh_key
        )
    else:
        # Local copy
        copied = copy_latest_images_from_comfyui(
            args.comfy_output_dir, 
            args.output_dir, 
            limit=args.count
        )
        
    if copied:
        logger.info(f"Copied {len(copied)} images to {args.output_dir}:")
        for image in copied:
            logger.info(f"  - {os.path.basename(image)}")
            
        # Open images with imv if requested
        if args.show:
            open_images_with_imv(copied)
    else:
        logger.warning("No images copied.")
