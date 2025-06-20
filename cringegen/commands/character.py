"""
Character-specific commands for cringegen CLI.

This module provides commands for generating prompts for specific canonical characters.
"""

import logging
from typing import Any
import sys
import typer
import rich

from ..utils.character_utils import generate_canon_character_prompt
from ..data.character_taxonomy import (
    FAMOUS_CHARACTER_TEMPLATES,
    FAMOUS_FURRY_CHARACTERS,
)
from ..data.characters import get_available_characters, get_individual_character

# Set up logging
logger = logging.getLogger(__name__)

app = typer.Typer(help="Generate prompts for specific characters")


def handle_character_command(args: Any) -> None:
    """Handle the character command."""
    # If --list flag is used or no name is provided, show the list of characters
    if args.list or not args.name:
        return handle_character_list(args)

    character_name = args.name.lower()

    # If --model-tags is provided, only print the model tags
    if hasattr(args, "model_tags") and args.model_tags:
        # Try to get character from individual files first
        try:
            character = get_individual_character(character_name)
        except Exception:
            # Fall back to the main character system
            from ..data.character_taxonomy import get_character_by_name

            character = get_character_by_name(character_name)

        if not character:
            logger.error(f"Unknown character: {character_name}")
            return

        # Print requested model tags
        model_tags = args.model_tags.split(",")
        print(f"Model tags for {character_name}:")
        for tag_name in model_tags:
            tag_name = tag_name.strip()
            if tag_name in character.model_tags:
                print(f"{tag_name}: {character.model_tags[tag_name]}")
            else:
                print(f"{tag_name}: Not available")
        return

    # Default behavior: just print the e621 tag
    try:
        character = get_individual_character(character_name)
    except Exception:
        # Fall back to the main character system
        from ..data.character_taxonomy import get_character_by_name

        character = get_character_by_name(character_name)

    if not character:
        logger.error(f"Unknown character: {character_name}")
        return

    if character.model_tags and "e621" in character.model_tags:
        print(character.model_tags["e621"])
        return

    # Fallback to generating the full prompt if no e621 tag exists
    # Generate the character prompt
    prompt = generate_canon_character_prompt(
        character_name=character_name,
        nsfw=args.nsfw,
        include_appearance=not args.no_appearance,
        include_accessories=not args.no_accessories,
        holding_sword=args.holding_sword,
    )

    # Output the prompt
    if prompt.startswith("Unknown character"):
        logger.error(prompt)
    else:
        print("Generated prompt:")
        print(prompt)

        # Output style recommendations
        if args.suggest_loras:
            from ..utils.lora_utils import find_loras_for_prompt

            loras = find_loras_for_prompt(prompt)
            if loras:
                print("\nRecommended LoRAs:")
                for lora_name, strength in loras.items():
                    print(f"  - {lora_name}: {strength}")


def add_character_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the character command to the CLI."""
    parser = subparsers.add_parser(
        "character",
        parents=[parent_parser],
        help="Generate a character-specific prompt",
        description="Generate a prompt for a specific canonical character like Blaidd",
    )

    # Character name argument
    parser.add_argument(
        "name",
        type=str,
        nargs="?",  # Make name optional to support --list
        help="Name of the character (e.g., blaidd)",
    )

    # Optional flags
    parser.add_argument("--nsfw", action="store_true", help="Include NSFW traits")
    parser.add_argument("--no-appearance", action="store_true", help="Exclude appearance traits")
    parser.add_argument("--no-accessories", action="store_true", help="Exclude accessories")
    parser.add_argument(
        "--holding-sword", action="store_true", help="For Blaidd: include 'holding sword'"
    )
    parser.add_argument(
        "--suggest-loras", action="store_true", help="Suggest appropriate LoRAs for this character"
    )

    # Add model tags option
    parser.add_argument(
        "--model-tags",
        type=str,
        help="Comma-separated list of model tags to display (e.g., e621,danbooru,gelbooru)",
    )

    # List available characters flag
    parser.add_argument(
        "--list", action="store_true", help="List all available character templates"
    )

    # Set the function to call when this command is invoked
    parser.set_defaults(func=handle_character_command)

    # Override if --list is used
    if "--list" in sys.argv:
        parser.set_defaults(func=handle_character_list)


def handle_character_list(args: Any) -> None:
    """Handle the character --list command."""
    print("Available character templates:")

    # Group characters by source if possible
    character_by_source = {}

    for name in sorted(FAMOUS_CHARACTER_TEMPLATES.keys()):
        character = FAMOUS_CHARACTER_TEMPLATES[name]

        # Get source (game, cartoon, etc.) if available
        source = "Unknown"
        for src_category, characters in FAMOUS_FURRY_CHARACTERS.items():
            if isinstance(characters, dict):
                for src, char_list in characters.items():
                    if name in [c.lower() for c in char_list]:
                        source = f"{src_category}/{src}"
                        break
            else:
                if name in [c.lower() for c in characters]:
                    source = src_category

            if source != "Unknown":
                break

        if source not in character_by_source:
            character_by_source[source] = []

        # Add character with extra detail for detailed templates
        if character.is_canon_character:
            character_by_source[source].append(f"{name} [detailed]")
        else:
            character_by_source[source].append(name)

    # Display characters grouped by source
    for source, characters in sorted(character_by_source.items()):
        print(f"\n{source}:")
        for character in sorted(characters):
            print(f"  - {character}")

    print("\nUse: cringegen character <name> [options]")
    print("Add --nsfw flag for NSFW prompt variants")

    # Special note for Blaidd
    if "blaidd" in FAMOUS_CHARACTER_TEMPLATES:
        print("\nSpecial options for Blaidd:")
        print("  --holding-sword : Include 'holding sword' in the prompt")


@app.callback()
def main(
    list_characters: bool = typer.Option(False, "--list", help="List available characters"),
):
    """Generate prompts for specific characters."""
    if list_characters:
        # Get characters from both systems - new individual files and old taxonomy
        individual_characters = get_available_characters()
        legacy_characters = sorted(FAMOUS_FURRY_CHARACTERS.keys())

        # Combine and deduplicate while preserving order
        all_characters = individual_characters + [
            c for c in legacy_characters if c not in individual_characters
        ]

        # Display available characters
        rich.print("\n[bold green]Available Characters:[/bold green]")
        for character in sorted(all_characters):
            rich.print(f"  • [cyan]{character}[/cyan]")
        rich.print(
            "\nUse 'cringegen character <name>' to generate a prompt for a specific character.\n"
        )
        raise typer.Exit()


# Need to import sys for the --list argument handling
import sys


def create_character_workflow(args):
    """Create a base character workflow template for XY plotting

    Args:
        args: Command line arguments

    Returns:
        A workflow dictionary
    """
    import random

    # Create a basic workflow
    from ..workflows.base import ComfyWorkflow

    workflow = ComfyWorkflow()

    # Add checkpoint loader
    checkpoint = (
        args.checkpoint
        if hasattr(args, "checkpoint") and args.checkpoint
        else "noobaiXLVpredv10.safetensors"
    )
    model_output = workflow.load_checkpoint(checkpoint)

    # Add LoRA if specified
    if hasattr(args, "lora") and args.lora:
        model_output, clip_output = workflow.load_lora(
            model_output,
            args.lora,
            args.lora_strength if hasattr(args, "lora_strength") else 1.0,
            args.lora_strength if hasattr(args, "lora_strength") else 1.0,
        )
    else:
        # Extract clip from model output if no LoRA
        clip_output = workflow.get_output(1, 1)

    # Create prompt - use character name if specified or a default prompt
    if hasattr(args, "prompt") and args.prompt:
        positive_prompt = args.prompt
    elif hasattr(args, "name") and args.name:
        # Generate character prompt
        positive_prompt = generate_canon_character_prompt(
            character_name=args.name.lower(),
            nsfw=hasattr(args, "nsfw") and args.nsfw,
            include_appearance=not (hasattr(args, "no_appearance") and args.no_appearance),
            include_accessories=not (hasattr(args, "no_accessories") and args.no_accessories),
            holding_sword=hasattr(args, "holding_sword") and args.holding_sword,
        )
    else:
        # Default prompt if no character name
        positive_prompt = "a detailed character portrait, high quality, best quality"

    # Create negative prompt
    if hasattr(args, "negative_prompt") and args.negative_prompt:
        negative_prompt = args.negative_prompt
    else:
        negative_prompt = "worst quality, low quality, medium quality, deleted, lowres, bad anatomy, bad hands, watermark"

    # Encode prompts
    positive_conditioning = workflow.clip_text_encode(clip_output, positive_prompt)
    negative_conditioning = workflow.clip_text_encode(clip_output, negative_prompt)

    # Create latent image
    width = args.width if hasattr(args, "width") else 1024
    height = args.height if hasattr(args, "height") else 1024
    latent = workflow.empty_latent(width, height)

    # Set up sampler
    seed = args.seed if hasattr(args, "seed") and args.seed != -1 else random.randint(0, 2**32 - 1)
    steps = args.steps if hasattr(args, "steps") else 40
    cfg = args.cfg if hasattr(args, "cfg") else 7.0
    sampler = args.sampler if hasattr(args, "sampler") else "euler_ancestral"
    scheduler = args.scheduler if hasattr(args, "scheduler") else "normal"

    # Add sampler
    latent_sampled = workflow.ksampler(
        model_output,
        positive_conditioning,
        negative_conditioning,
        latent,
        seed,
        steps,
        cfg,
        sampler,
        scheduler,
    )

    # Add VAE decoder
    image = workflow.vae_decode(workflow.get_output(1, 2), latent_sampled)

    # Add image saving
    workflow.save_image(image, "xyplot")

    return workflow.to_dict()
