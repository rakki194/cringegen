"""
Clothing command for cringegen CLI.

This module provides a command to generate clothing options for furry characters.
"""

import argparse
import logging
import random
from typing import Any, Dict, List, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Dictionary of clothing by taxonomy and gender
CLOTHING = {
    "default": {
        "male": {
            "casual": [
                "t-shirt",
                "jeans",
                "hoodie",
                "sweater",
                "cargo pants",
                "shorts",
                "tank top",
                "jacket",
                "button-up shirt",
                "flannel shirt",
                "khaki pants",
                "baseball cap",
                "beanie",
                "sneakers",
                "boots",
            ],
            "formal": [
                "suit",
                "tuxedo",
                "dress shirt",
                "tie",
                "bow tie",
                "vest",
                "slacks",
                "blazer",
                "dress shoes",
                "cufflinks",
                "pocket square",
            ],
            "fantasy": [
                "tunic",
                "leather armor",
                "cloak",
                "wizard robes",
                "chainmail",
                "cloth wrappings",
                "plate armor",
                "ranger outfit",
                "linen shirt",
                "travel cape",
                "leather boots",
                "hide garments",
            ],
        },
        "female": {
            "casual": [
                "t-shirt",
                "jeans",
                "blouse",
                "skirt",
                "dress",
                "leggings",
                "sweater",
                "shorts",
                "tank top",
                "hoodie",
                "cardigan",
                "jacket",
                "crop top",
                "capri pants",
                "sandals",
                "sneakers",
                "boots",
            ],
            "formal": [
                "dress",
                "gown",
                "cocktail dress",
                "blouse",
                "skirt suit",
                "pants suit",
                "evening gown",
                "heels",
                "dress shoes",
                "jewelry",
                "pearl necklace",
                "silk scarf",
            ],
            "fantasy": [
                "sorceress robes",
                "leather armor",
                "elven dress",
                "ranger outfit",
                "flowing gown",
                "witch's attire",
                "battle dress",
                "priestess garments",
                "druid robes",
                "medieval dress",
                "cloth wrappings",
                "huntress outfit",
            ],
        },
        "neutral": {
            "casual": [
                "t-shirt",
                "jeans",
                "hoodie",
                "sweater",
                "jacket",
                "shorts",
                "tank top",
                "button-up shirt",
                "flannel shirt",
                "cargo pants",
                "sneakers",
                "boots",
                "hat",
            ],
            "formal": [
                "suit",
                "dress shirt",
                "tie",
                "vest",
                "slacks",
                "blazer",
                "dress shoes",
            ],
            "fantasy": [
                "tunic",
                "leather armor",
                "cloak",
                "robes",
                "ranger outfit",
                "linen shirt",
                "chainmail",
                "adventurer's garb",
                "travel cape",
                "leather boots",
            ],
        },
    },
    "reptile": {
        "neutral": {
            "special": [
                "scale-accommodating tunic",
                "open-backed shirt",
                "tail-hole pants",
                "scale-friendly fabrics",
                "lightweight garments",
                "breathable materials",
                "sun-protective clothing",
                "heat-preserving cloak",
            ],
        },
    },
    "avian": {
        "neutral": {
            "special": [
                "wing-slot jacket",
                "feather-safe materials",
                "lightweight garments",
                "open-backed shirt",
                "specially tailored wing openings",
                "plumage-protective clothing",
                "harness with wing spaces",
            ],
        },
    },
    "canine": {
        "neutral": {
            "special": [
                "tail-hole pants",
                "fur-friendly materials",
                "jacket with hood for ears",
                "paw-friendly footwear",
                "scent-resistant fabrics",
            ],
        },
    },
    "feline": {
        "neutral": {
            "special": [
                "claw-resistant materials",
                "tail-hole pants",
                "fur-friendly fabrics",
                "quiet clothing (no bells)",
                "ear-accommodating hats",
                "stretchy materials for climbing",
            ],
        },
    },
}


def add_clothing_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the clothing command to the CLI."""
    parser = subparsers.add_parser(
        "clothing",
        parents=[parent_parser],
        help="Generate clothing options for furry characters",
        description="Generate clothing appropriate for specific furry species and gender",
    )

    # Species argument
    parser.add_argument(
        "--species",
        type=str,
        help="Species of the furry character (e.g., wolf, fox, dragon)",
    )

    # Gender argument
    parser.add_argument(
        "--gender",
        type=str,
        choices=["male", "female", "neutral"],
        default="neutral",
        help="Gender of the character (default: neutral)",
    )

    # Style argument
    parser.add_argument(
        "--style",
        type=str,
        choices=["casual", "formal", "fantasy", "special", "all"],
        default="casual",
        help="Style of clothing to generate (default: casual)",
    )

    # Count argument
    parser.add_argument(
        "--count", 
        type=int, 
        default=1, 
        help="Number of clothing items/outfits to generate (default: 1)"
    )

    # Full outfit flag
    parser.add_argument(
        "--outfit",
        action="store_true",
        help="Generate a complete outfit rather than individual clothing items",
    )

    # Format options
    parser.add_argument(
        "--format",
        type=str,
        choices=["simple", "detailed", "prompt", "csv"],
        default="simple",
        help="Output format (simple, detailed, prompt-ready, or csv)",
    )

    # Set the function to call when this command is invoked
    parser.set_defaults(func=handle_clothing_command)


def handle_clothing_command(args: Any) -> None:
    """Handle the clothing command."""
    # Check if species is provided
    if not args.species:
        logger.error("Species is required. Use --species to specify a species.")
        return

    # Get the taxonomy group for the species
    from ..data.character_taxonomy import SPECIES_TAXONOMY
    taxonomy = SPECIES_TAXONOMY.get(args.species.lower(), "default")

    # Generate clothing
    if args.outfit:
        # Generate a complete outfit
        outfits = generate_outfits(
            taxonomy=taxonomy,
            gender=args.gender,
            style=args.style,
            count=args.count,
        )
        output_outfits(outfits, args.format)
    else:
        # Generate individual clothing items
        clothing = generate_clothing(
            taxonomy=taxonomy,
            gender=args.gender,
            style=args.style,
            count=args.count,
        )
        output_clothing(clothing, args.format)


def generate_clothing(
    taxonomy: str,
    gender: str = "neutral", 
    style: str = "casual", 
    count: int = 1
) -> List[str]:
    """Generate clothing items for a furry character.

    Args:
        taxonomy: The taxonomy group of the species
        gender: The gender of the character (male/female/neutral)
        style: The style of clothing to generate
        count: Number of clothing items to generate

    Returns:
        A list of clothing items
    """
    all_clothing = []
    
    # Get default clothing for the gender
    default_clothing = CLOTHING.get("default", {}).get(gender, CLOTHING["default"]["neutral"])
    
    # Get species-specific clothing
    species_clothing = CLOTHING.get(taxonomy, {}).get("neutral", {})
    
    # Combine clothing based on style
    if style == "all":
        # Add all clothing from default styles
        for s, items in default_clothing.items():
            all_clothing.extend(items)
        
        # Add all special clothing for the species if available
        if "special" in species_clothing:
            all_clothing.extend(species_clothing["special"])
    elif style == "special":
        # For special style, check species-specific clothing first
        if "special" in species_clothing:
            all_clothing.extend(species_clothing["special"])
        else:
            # If no special clothing, use casual as fallback
            logger.warning(f"No special clothing found for {taxonomy}. Using casual clothing instead.")
            if "casual" in default_clothing:
                all_clothing.extend(default_clothing["casual"])
    else:
        # Add clothing from specific style
        if style in default_clothing:
            all_clothing.extend(default_clothing[style])
        else:
            # Fallback to casual if requested style isn't available
            logger.warning(f"No {style} clothing found for {gender}. Using casual clothing instead.")
            if "casual" in default_clothing:
                all_clothing.extend(default_clothing["casual"])
    
    # If no clothing was found, use generic items
    if not all_clothing:
        logger.warning(f"No clothing found for {taxonomy} ({gender}, {style}). Using generic items.")
        all_clothing = ["shirt", "pants", "shoes"]
    
    # Select random clothing items
    if count >= len(all_clothing):
        return all_clothing
    else:
        return random.sample(all_clothing, count)


def generate_outfits(
    taxonomy: str,
    gender: str = "neutral", 
    style: str = "casual", 
    count: int = 1
) -> List[Dict[str, List[str]]]:
    """Generate complete outfits for a furry character.

    Args:
        taxonomy: The taxonomy group of the species
        gender: The gender of the character (male/female/neutral)
        style: The style of clothing to generate
        count: Number of outfits to generate

    Returns:
        A list of outfits, where each outfit is a dictionary of clothing categories
    """
    outfits = []
    
    for _ in range(count):
        outfit = {
            "top": [],
            "bottom": [],
            "footwear": [],
            "outerwear": [],
            "accessories": [],
        }
        
        # Get default clothing for the gender
        default_clothing = CLOTHING.get("default", {}).get(gender, CLOTHING["default"]["neutral"])
        
        # Get species-specific clothing
        species_clothing = CLOTHING.get(taxonomy, {}).get("neutral", {})
        
        # Determine which style to use
        actual_style = style
        if style == "special":
            # For special style, check if it exists, otherwise use casual
            if "special" not in species_clothing:
                actual_style = "casual"
        elif style not in default_clothing:
            # Fallback to casual if requested style isn't available
            actual_style = "casual"
        
        # Add tops
        tops = []
        if actual_style == "casual":
            tops = ["t-shirt", "button-up shirt", "sweater", "tank top", "hoodie"]
        elif actual_style == "formal":
            if gender == "male":
                tops = ["dress shirt", "tuxedo shirt", "suit jacket"]
            elif gender == "female":
                tops = ["blouse", "silk top", "dress"]
            else:
                tops = ["dress shirt", "blouse", "button-up shirt"]
        elif actual_style == "fantasy":
            tops = ["tunic", "robe", "leather vest", "linen shirt"]
        
        if tops:
            outfit["top"].append(random.choice(tops))
        
        # Add bottoms
        bottoms = []
        if actual_style == "casual":
            if gender == "male":
                bottoms = ["jeans", "shorts", "cargo pants", "khaki pants"]
            elif gender == "female":
                bottoms = ["jeans", "skirt", "shorts", "leggings"]
            else:
                bottoms = ["jeans", "shorts", "pants"]
        elif actual_style == "formal":
            if gender == "male":
                bottoms = ["slacks", "suit pants", "dress pants"]
            elif gender == "female":
                if "dress" not in outfit["top"]:  # Don't add bottoms if we have a dress
                    bottoms = ["dress pants", "pencil skirt", "a-line skirt"]
            else:
                bottoms = ["slacks", "dress pants"]
        elif actual_style == "fantasy":
            bottoms = ["cloth pants", "leather pants", "hide leggings"]
        
        if bottoms and "dress" not in outfit["top"][0]:  # Don't add bottoms if we have a dress
            outfit["bottom"].append(random.choice(bottoms))
        
        # Add footwear
        footwear = []
        if actual_style == "casual":
            footwear = ["sneakers", "boots", "sandals"]
        elif actual_style == "formal":
            if gender == "male":
                footwear = ["dress shoes", "loafers", "oxfords"]
            elif gender == "female":
                footwear = ["heels", "flats", "dress shoes"]
            else:
                footwear = ["dress shoes", "loafers"]
        elif actual_style == "fantasy":
            footwear = ["leather boots", "cloth shoes", "sandals"]
        
        if footwear:
            outfit["footwear"].append(random.choice(footwear))
        
        # Add outerwear
        if random.random() < 0.5:  # 50% chance to add outerwear
            outerwear = []
            if actual_style == "casual":
                outerwear = ["jacket", "hoodie", "cardigan"]
            elif actual_style == "formal":
                if gender == "male":
                    outerwear = ["blazer", "suit jacket", "overcoat"]
                elif gender == "female":
                    outerwear = ["blazer", "cardigan", "shawl"]
                else:
                    outerwear = ["blazer", "jacket", "overcoat"]
            elif actual_style == "fantasy":
                outerwear = ["cloak", "cape", "robe"]
            
            if outerwear:
                outfit["outerwear"].append(random.choice(outerwear))
        
        # Add accessories
        if random.random() < 0.7:  # 70% chance to add accessories
            accessories = []
            if actual_style == "casual":
                if gender == "male":
                    accessories = ["watch", "beanie", "cap", "sunglasses"]
                elif gender == "female":
                    accessories = ["jewelry", "hairpin", "scarf", "sunglasses"]
                else:
                    accessories = ["watch", "cap", "sunglasses"]
            elif actual_style == "formal":
                if gender == "male":
                    accessories = ["tie", "bow tie", "cufflinks", "pocket square"]
                elif gender == "female":
                    accessories = ["necklace", "earrings", "bracelet", "clutch"]
                else:
                    accessories = ["tie", "watch", "cufflinks"]
            elif actual_style == "fantasy":
                accessories = ["belt pouch", "amulet", "cloak clasp", "bracers"]
            
            if accessories:
                outfit["accessories"].append(random.choice(accessories))
        
        # Add species-specific item if available and random chance
        if "special" in species_clothing and random.random() < 0.5:
            special_items = species_clothing["special"]
            if special_items:
                outfit["accessories"].append(random.choice(special_items))
        
        outfits.append(outfit)
    
    return outfits


def output_clothing(clothing: List[str], format_type: str) -> None:
    """Format and output the generated clothing items.

    Args:
        clothing: List of clothing items to output
        format_type: The format to output (simple, detailed, prompt, csv)
    """
    if not clothing:
        print("No clothing generated.")
        return
        
    if format_type == "simple":
        if len(clothing) == 1:
            print(clothing[0])
        else:
            for item in clothing:
                print(f"- {item}")
    elif format_type == "detailed":
        if len(clothing) == 1:
            item = clothing[0]
            print(f"Clothing: {item}")
            print(f"Description: A {item}")
        else:
            for i, item in enumerate(clothing, 1):
                print(f"Item {i}: {item}")
                print(f"Description: A {item}")
                if i < len(clothing):
                    print()  # Add blank line between items
    elif format_type == "csv":
        # Just output the comma-separated values with no other text
        print(",".join(clothing))
    elif format_type == "prompt":
        # Format for direct inclusion in prompts
        if len(clothing) == 1:
            print(f"wearing a {clothing[0]}")
        else:
            clothing_phrases = [f"a {item}" for item in clothing]
            print(f"wearing {', '.join(clothing_phrases)}")


def output_outfits(outfits: List[Dict[str, List[str]]], format_type: str) -> None:
    """Format and output the generated outfits.

    Args:
        outfits: List of outfits to output
        format_type: The format to output (simple, detailed, prompt, csv)
    """
    if not outfits:
        print("No outfits generated.")
        return
    
    for i, outfit in enumerate(outfits):
        if len(outfits) > 1 and format_type != "csv":
            print(f"\nOutfit {i+1}:")
        
        if format_type == "simple":
            for category, items in outfit.items():
                if items:
                    cat_name = category.capitalize()
                    print(f"{cat_name}: {', '.join(items)}")
        
        elif format_type == "detailed":
            for category, items in outfit.items():
                if items:
                    cat_name = category.capitalize()
                    print(f"{cat_name}: {', '.join(items)}")
                    
                    # Add description for each category
                    if category == "top":
                        print(f"Description: A {items[0]} for the upper body")
                    elif category == "bottom":
                        print(f"Description: {items[0].capitalize()} for the lower body")
                    elif category == "footwear":
                        print(f"Description: {items[0].capitalize()} for the feet")
                    elif category == "outerwear":
                        print(f"Description: A {items[0]} worn over other clothing")
                    elif category == "accessories":
                        print(f"Description: Accessorized with {', '.join(items)}")
                    print()  # Add blank line between categories
        
        elif format_type == "csv":
            # Collect all clothing items from all categories
            all_items = []
            for category, items in outfit.items():
                all_items.extend(items)
            
            # Output as comma-separated list with no other text
            if all_items:
                print(",".join(all_items))
        
        elif format_type == "prompt":
            # Format for direct inclusion in prompts
            outfit_pieces = []
            
            # Start with top
            if outfit["top"]:
                outfit_pieces.append(f"a {outfit['top'][0]}")
            
            # Add bottom if present
            if outfit["bottom"]:
                outfit_pieces.append(f"{outfit['bottom'][0]}")
            
            # Add outerwear if present
            if outfit["outerwear"]:
                outfit_pieces.append(f"a {outfit['outerwear'][0]}")
            
            # Add footwear if present
            if outfit["footwear"]:
                outfit_pieces.append(f"{outfit['footwear'][0]}")
            
            # Add accessories if present
            if outfit["accessories"]:
                if len(outfit["accessories"]) == 1:
                    outfit_pieces.append(f"a {outfit['accessories'][0]}")
                else:
                    acc_list = [f"a {acc}" for acc in outfit['accessories']]
                    outfit_pieces.append(f"{', '.join(acc_list)}")
            
            # Output the complete outfit
            print(f"wearing {', '.join(outfit_pieces)}") 