"""
Accessories command for cringegen CLI.

This module provides a command to generate random accessories for furries.
"""

import argparse
import logging
import random
import sys
from typing import Any, List, Dict, Optional

from ..data.accessories import SPECIES_ACCESSORIES
from ..data.character_taxonomy import SPECIES_TAXONOMY
from ..prompt_generation.nlp.species_utils import get_species_accessories
from ..prompt_generation.nlp.data_utils import get_taxonomy_group

# Set up logging
logger = logging.getLogger(__name__)


def add_accessories_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the accessories command to the CLI."""
    parser = subparsers.add_parser(
        "accessories",
        parents=[parent_parser],
        help="Generate random accessories for furry characters",
        description="Generate random accessories appropriate for specific furry species",
    )

    # Species argument
    parser.add_argument(
        "--species",
        type=str,
        help="Species of the furry character (e.g., wolf, fox, dragon)",
    )

    # Form type arguments (mutually exclusive)
    form_group = parser.add_mutually_exclusive_group()
    form_group.add_argument(
        "--anthro", 
        action="store_true", 
        help="Generate accessories for anthropomorphic characters"
    )
    form_group.add_argument(
        "--feral", 
        action="store_true", 
        help="Generate accessories for feral characters"
    )

    # Gender argument
    parser.add_argument(
        "--gender",
        type=str,
        choices=["male", "female", "neutral"],
        default="neutral",
        help="Gender of the character (default: neutral)",
    )

    # Count argument
    parser.add_argument(
        "--count", 
        type=int, 
        default=3, 
        help="Number of accessories to generate (default: 3)"
    )

    # List available species flag
    parser.add_argument(
        "--list-species", 
        action="store_true", 
        help="List all available species with accessory data"
    )

    # List available accessories for a specific species
    parser.add_argument(
        "--list-accessories",
        action="store_true",
        help="List all available accessories for the specified species",
    )

    # Format options
    parser.add_argument(
        "--format",
        type=str,
        choices=["list", "comma", "prompt", "csv"],
        default="list",
        help="Output format (list, comma-separated, prompt-ready, or csv)",
    )

    # Set the function to call when this command is invoked
    parser.set_defaults(func=handle_accessories_command)


def handle_accessories_command(args: Any) -> None:
    """Handle the accessories command."""
    # Handle list species flag
    if args.list_species:
        list_species_with_accessories()
        return

    # Check if species is provided
    if not args.species and not args.list_species:
        logger.error("Species is required. Use --species or --list-species to see available options.")
        return

    # Handle list accessories flag
    if args.list_accessories:
        list_accessories_for_species(args.species, args.anthro, args.feral)
        return

    # Determine form type (default to anthro if neither is specified)
    is_anthro = not args.feral if args.feral else True

    # Generate accessories
    accessories = generate_random_accessories(
        species=args.species,
        gender=args.gender,
        count=args.count,
        is_anthro=is_anthro,
    )

    # Format and output the accessories
    output_accessories(accessories, args.format)


def generate_random_accessories(
    species: str, 
    gender: str = "neutral", 
    count: int = 3, 
    is_anthro: bool = True
) -> List[str]:
    """Generate random accessories for a furry character.

    Args:
        species: The species of the character
        gender: The gender of the character (male/female/neutral)
        count: Number of accessories to generate
        is_anthro: Whether the character is anthro (True) or feral (False)

    Returns:
        A list of random accessories
    """
    # Use the existing NLP utility function
    accessories = get_species_accessories(
        species=species,
        gender=gender,
        count=count,
        is_anthro=is_anthro
    )
    
    # If no accessories were found, provide a generic fallback
    if not accessories:
        form_type = "anthro" if is_anthro else "feral"
        logger.warning(f"No specific accessories found for {species} ({form_type}). Using defaults.")
        
        # Use default accessories
        default_accessories = SPECIES_ACCESSORIES.get("default", {})
        form_accessories = default_accessories.get(form_type, {})
        gender_accessories = form_accessories.get(gender, form_accessories.get("neutral", []))
        
        if gender_accessories:
            accessories = random.sample(gender_accessories, min(count, len(gender_accessories)))
        else:
            accessories = ["collar", "bandana"] if count >= 2 else ["collar"]
    
    return accessories


def list_species_with_accessories() -> None:
    """List all species that have accessory data."""
    print("Species with accessory data:")
    
    # Get all taxonomy groups with accessories
    taxonomy_groups = sorted(SPECIES_ACCESSORIES.keys())
    
    for group in taxonomy_groups:
        # Skip the default group for this listing
        if group == "default":
            continue
            
        # Find all species in this taxonomy group
        species_in_group = [
            species for species, taxonomy in SPECIES_TAXONOMY.items() 
            if taxonomy == group
        ]
        
        if species_in_group:
            print(f"\n{group.capitalize()} (taxonomy group):")
            for species in sorted(species_in_group):
                print(f"  - {species}")
    
    print("\nUse: cringegen accessories --species <species> [options]")


def list_accessories_for_species(species: str, anthro_flag: bool, feral_flag: bool) -> None:
    """List all available accessories for a specific species.

    Args:
        species: The species to list accessories for
        anthro_flag: Whether anthro accessories were requested
        feral_flag: Whether feral accessories were requested
    """
    # Get the taxonomy group for the species
    taxonomy = get_taxonomy_group(species)
    
    # Get the accessories for this taxonomy group
    taxonomy_accessories = SPECIES_ACCESSORIES.get(taxonomy, SPECIES_ACCESSORIES.get("default", {}))
    
    print(f"Available accessories for {species} (taxonomy: {taxonomy}):")
    
    # Determine which form types to display
    if not anthro_flag and not feral_flag:
        # Show both if neither is specified
        form_types = ["anthro", "feral"]
    elif anthro_flag:
        form_types = ["anthro"]
    else:
        form_types = ["feral"]
    
    for form_type in form_types:
        if form_type in taxonomy_accessories:
            print(f"\n{form_type.capitalize()}:")
            for gender, accs in sorted(taxonomy_accessories[form_type].items()):
                print(f"  {gender.capitalize()}: {', '.join(sorted(accs))}")
        else:
            print(f"\n{form_type.capitalize()}: No specific accessories defined")


def output_accessories(accessories: List[str], format_type: str) -> None:
    """Format and output the generated accessories.

    Args:
        accessories: List of accessories to output
        format_type: The format to output (list, comma, prompt, csv)
    """
    if not accessories:
        print("No accessories generated.")
        return
        
    if format_type == "list":
        print("Generated accessories:")
        for acc in accessories:
            print(f"- {acc}")
    elif format_type == "comma":
        print(", ".join(accessories))
    elif format_type == "csv":
        # Just output the comma-separated values with no other text
        print(",".join(accessories))
    elif format_type == "prompt":
        # Format for direct inclusion in prompts
        prefix = random.choice(["wearing", "with", "adorned with"])
        print(f"{prefix} {', '.join(accessories)}") 