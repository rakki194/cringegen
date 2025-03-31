"""
Utility commands for CringeGen
"""

import argparse
import logging
import os
from typing import Any, Dict, List

from ..utils.comfy_api import get_lora_directory, resolve_model_path
from ..utils.lora_metadata.autocomplete import get_trigger_phrases

logger = logging.getLogger(__name__)


def add_utils_commands(subparsers, parent_parser):
    """Add utility commands to the CLI"""
    # Add resolve-path command
    resolve_path_parser = subparsers.add_parser(
        "resolve-path", help="Resolve a model path", parents=[parent_parser]
    )
    resolve_path_parser.add_argument("model_path", type=str, help="Model path to resolve")
    resolve_path_parser.set_defaults(func=resolve_path)

    # Add trigger-phrases command
    trigger_phrases_parser = subparsers.add_parser(
        "trigger-phrases", help="Get trigger phrases for a LoRA", parents=[parent_parser]
    )
    trigger_phrases_parser.add_argument("lora", type=str, help="Name or path of the LoRA file")
    trigger_phrases_parser.set_defaults(func=get_trigger_phrases_cmd)

    return subparsers


def resolve_path(args):
    """Resolve a model path"""
    try:
        resolved_path = resolve_model_path(args.model_path)
        if resolved_path:
            logger.info(f"Resolved path: {resolved_path}")
        else:
            logger.warning(f"Could not resolve path: {args.model_path}")
    except Exception as e:
        logger.error(f"Error resolving path: {str(e)}")


def get_trigger_phrases_cmd(args):
    """Get trigger phrases for a LoRA"""
    try:
        # Get the LoRA directory
        lora_dir = get_lora_directory()

        # Try to find the LoRA file
        lora_path = None
        if os.path.exists(args.lora):
            # Absolute path provided
            lora_path = args.lora
        elif os.path.exists(os.path.join(lora_dir, args.lora)):
            # Direct path in LoRA directory
            lora_path = os.path.join(lora_dir, args.lora)
        else:
            # Try to find in subdirectories
            for root, _, files in os.walk(lora_dir):
                for file in files:
                    if file == args.lora or file == f"{args.lora}.safetensors":
                        lora_path = os.path.join(root, file)
                        break
                if lora_path:
                    break

        if not lora_path:
            # Try to match partial name
            for root, _, files in os.walk(lora_dir):
                for file in files:
                    if args.lora.lower() in file.lower() and file.endswith(".safetensors"):
                        lora_path = os.path.join(root, file)
                        break
                if lora_path:
                    break

        if not lora_path:
            logger.error(f"LoRA file not found: {args.lora}")
            return

        # Get trigger phrases
        phrases = get_trigger_phrases(lora_path)

        if phrases:
            logger.info(f"Trigger phrases for {os.path.basename(lora_path)}:")
            for phrase in phrases:
                logger.info(f"  - {phrase}")
        else:
            logger.info(f"No trigger phrases found for {os.path.basename(lora_path)}")
    except Exception as e:
        logger.error(f"Error getting trigger phrases: {str(e)}")
