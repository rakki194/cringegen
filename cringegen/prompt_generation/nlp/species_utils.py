"""
Utility functions for species-related natural language processing.
"""

import random
from typing import List, Optional

# Update imports to use new centralized data module
from ...data import anatomy, accessories, colors, taxonomy

# Original import for reference:
# from cringegen.data.species_data import (
#     FEMALE_ANATOMY,
#     MALE_ANATOMY,
#     SPECIES_ACCESSORIES,
#     SPECIES_COLORS,
#     SPECIES_TAXONOMY,
# )

# Map old variables to new module structure
FEMALE_ANATOMY = anatomy.FEMALE_ANATOMY
MALE_ANATOMY = anatomy.MALE_ANATOMY
SPECIES_ACCESSORIES = accessories.SPECIES_ACCESSORIES
SPECIES_COLORS = colors.SPECIES_COLORS
BASIC_COLORS = colors.BASIC_COLORS
SPECIES_TAXONOMY = taxonomy.SPECIES_TAXONOMY


def get_anatomical_terms(species: str, gender: str, explicit_level: int = 1) -> List[str]:
    """Get anatomical terms for a specific species and gender

    Args:
        species: The species of the character
        gender: The gender of the character (male/female)
        explicit_level: Level of explicitness (1-3)

    Returns:
        A list of anatomical terms appropriate for the species/gender
    """
    # Default to general terms if species isn't recognized
    taxonomy = SPECIES_TAXONOMY.get(species.lower(), "default")

    # Select terms based on gender
    if gender.lower() == "male":
        # Get terms for the specific taxonomy, or default if not found
        available_terms = MALE_ANATOMY.get(taxonomy, MALE_ANATOMY["default"])
    elif gender.lower() == "female":
        available_terms = FEMALE_ANATOMY.get(taxonomy, FEMALE_ANATOMY["default"])
    else:
        # For non-binary or unspecified gender, use default terms
        return []

    # Select terms based on explicit level
    # Level 1: Just return the taxonomy-specific term
    if explicit_level == 1:
        return [available_terms[0]] if available_terms else []

    # Level 2: Return more explicit but still general terms
    elif explicit_level == 2:
        return available_terms[: min(2, len(available_terms))]

    # Level 3: Return only NSFW terms (skip questionable ones)
    else:
        if len(available_terms) <= 2:
            return available_terms
        else:
            return available_terms[2:]


def enhance_prompt_with_anatomy(
    prompt: str, species: str, gender: str, explicit_level: int = 1
) -> str:
    """Enhance a prompt with anatomical terms based on species and gender

    Args:
        prompt: Original prompt text
        species: The species of the character
        gender: The gender of the character
        explicit_level: Level of explicitness (1-3)

    Returns:
        Enhanced prompt with anatomical terms
    """
    # Special case for sergals at explicit level 3
    if species.lower() == "sergal" and gender.lower() == "male" and explicit_level == 3:
        return f"{prompt}, animal genitalia"

    # Get appropriate anatomical terms
    terms = get_anatomical_terms(species, gender, explicit_level)

    # If no terms were found or this is not meant to be explicit, return original prompt
    if not terms:
        return prompt

    # Add terms to the prompt in a natural way
    # Choose a random conjunction
    conjunction = random.choice(["with", "showing", "displaying", "presenting"])

    # Add the terms to the prompt with a comma
    enhanced_prompt = f"{prompt}, {conjunction} {', '.join(terms)}"

    return enhanced_prompt


def get_species_accessories(
    species: str, gender: str, count: int = 1, is_anthro: bool = True
) -> List[str]:
    """Get accessories appropriate for a specific species and gender

    Args:
        species: The species of the character
        gender: The gender of the character
        count: Number of accessories to return
        is_anthro: Whether the character is anthro (True) or feral (False)

    Returns:
        A list of accessory terms
    """
    # Get the taxonomy for the species
    taxonomy = SPECIES_TAXONOMY.get(species.lower(), "default")

    # Get the accessories for this taxonomy group
    taxonomy_accessories = SPECIES_ACCESSORIES.get(taxonomy, SPECIES_ACCESSORIES["default"])

    # Determine if we're using anthro or feral accessories
    form_type = "anthro" if is_anthro else "feral"

    # Get the accessories for the appropriate form type, defaulting to anthro if not found
    if form_type in taxonomy_accessories:
        form_accessories = taxonomy_accessories[form_type]
    else:
        # Fallback to the anthro form if the specific form isn't found
        form_accessories = taxonomy_accessories.get("anthro", taxonomy_accessories)

    # Get gender-specific accessories, defaulting to neutral if gender not found
    if gender.lower() in ["male", "female"]:
        accessories = form_accessories.get(gender.lower(), form_accessories.get("neutral", []))
    else:
        accessories = form_accessories.get("neutral", [])

    # Return random selection of count accessories
    if not accessories:
        return []
    return random.sample(accessories, min(count, len(accessories)))


def get_species_colors(species: str, count: int = 1) -> List[str]:
    """Get colors commonly associated with a specific species

    Args:
        species: The species of the character
        count: Number of colors to return

    Returns:
        A list of color terms
    """
    # Get species-specific colors if available
    if species.lower() in SPECIES_COLORS:
        colors_list = SPECIES_COLORS[species.lower()]
    # If no species-specific colors found, use basic colors
    else:
        colors_list = BASIC_COLORS

    # Return random selection of count colors
    return random.sample(colors_list, min(count, len(colors_list)))


def generate_species_description(species: str, gender: str) -> str:
    """Generate a natural language description of a species

    Args:
        species: The species of the character
        gender: The gender of the character

    Returns:
        A descriptive phrase for the species
    """
    # Get the taxonomy for the species
    taxonomy = SPECIES_TAXONOMY.get(species.lower(), "default")

    # Ensure gender is a recognized value
    if gender.lower() not in ["male", "female"]:
        gender = random.choice(["male", "female"])

    # Capitalize gender for better presentation
    gender_display = gender.lower()

    # Templates for different taxonomies
    templates = {
        "canine": [
            f"{gender_display} {species} with a fluffy tail and pointed ears",
            f"{gender_display} {species} with alert eyes and a sleek coat",
            f"{gender_display} {species} with a powerful build",
        ],
        "feline": [
            f"{gender_display} {species} with sleek fur and alert eyes",
            f"{gender_display} {species} with graceful movements and sharp claws",
            f"{gender_display} {species} with a lithe body and a long tail",
        ],
        "equine": [
            f"{gender_display} {species} with a flowing mane and strong build",
            f"{gender_display} {species} with powerful legs and a proud stance",
            f"{gender_display} {species} with a muscular body and expressive eyes",
        ],
        "bovine": [
            f"{gender_display} {species} with a sturdy frame and gentle eyes",
            f"{gender_display} {species} with impressive horns and a powerful build",
            f"{gender_display} {species} with a broad chest and strong limbs",
        ],
        "rodent": [
            f"{gender_display} {species} with bright eyes and twitching whiskers",
            f"{gender_display} {species} with nimble paws and soft fur",
            f"{gender_display} {species} with round ears and a twitchy nose",
        ],
        "lagomorph": [
            f"{gender_display} {species} with long ears and a fluffy tail",
            f"{gender_display} {species} with a twitching nose and soft fur",
            f"{gender_display} {species} with powerful hind legs and alert eyes",
        ],
        "reptile": [
            f"{gender_display} {species} with shimmering scales and striking eyes",
            f"{gender_display} {species} with a powerful tail and imposing presence",
            f"{gender_display} {species} with sharp claws",
        ],
        "avian": [
            f"{gender_display} {species} with colorful feathers and a sharp beak",
            f"{gender_display} {species} with graceful wings and keen eyes",
            f"{gender_display} {species} with a proud stance and elegant movements",
        ],
    }

    # Get species-specific templates or default templates
    species_templates = templates.get(
        taxonomy,
        [
            f"{gender_display} {species}",
        ],
    )

    # Choose a random template
    description = random.choice(species_templates)

    # Add a color detail from the species colors
    colors = get_species_colors(species, 1)
    if colors:
        color_phrases = [
            f"with {colors[0]} highlights",
            f"with {colors[0]} accents",
            f"with {colors[0]} markings",
            f"in shades of {colors[0]}",
        ]
        description += " " + random.choice(color_phrases)

    return description
