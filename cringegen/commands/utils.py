"""
Utility commands for cringegen
"""

import argparse
import logging
import os
import random
from typing import Any, Dict, List

from ..utils.comfy_api import get_lora_directory, resolve_model_path
from ..utils.lora_metadata.autocomplete import get_trigger_phrases
from ..prompt_generation.generators.furry_generator import FurryPromptGenerator, NsfwFurryPromptGenerator

logger = logging.getLogger(__name__)


def generate_random_prompt(nsfw: bool = False) -> str:
    """
    Generate a random prompt for image generation.
    
    Args:
        nsfw: Whether to generate an NSFW prompt
        
    Returns:
        A randomly generated prompt string
    """
    if nsfw:
        # Create NSFW furry prompt generator with random parameters
        generator = NsfwFurryPromptGenerator(
            explicit_level=random.randint(1, 3),
            use_anatomical_terms=True,
            use_nlp=True,
            use_art_style=True
        )
    else:
        # Create regular furry prompt generator
        generator = FurryPromptGenerator(
            use_nlp=True,
            use_art_style=True
        )
    
    # Generate the prompt
    return generator.generate()


def generate_prompt_cmd(args):
    """Command handler for generate-prompt"""
    # Set the seed if specified
    if args.seed != -1:
        random.seed(args.seed)
        logger.info(f"Using seed: {args.seed}")
        
    # Generate prompts
    for i in range(args.count):
        prompt = generate_random_prompt(nsfw=args.nsfw)
        if args.count > 1:
            logger.info(f"Prompt {i+1}/{args.count}:")
        logger.info(prompt)


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
    
    # Add generate-prompt command
    generate_prompt_parser = subparsers.add_parser(
        "generate-prompt", help="Generate random prompts", parents=[parent_parser]
    )
    generate_prompt_parser.add_argument("--nsfw", action="store_true", help="Generate NSFW prompts")
    generate_prompt_parser.add_argument("--count", type=int, default=1, help="Number of prompts to generate")
    generate_prompt_parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    generate_prompt_parser.set_defaults(func=generate_prompt_cmd)

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
