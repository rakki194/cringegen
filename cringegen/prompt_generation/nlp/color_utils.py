"""
Utility functions for color generation and manipulation.
"""

import random
from typing import List

# Update imports to use new centralized data module
from ...data import colors, taxonomy

# Original import for reference:
# from cringegen.data.color_data import (
#     BASIC_COLORS,
#     BODY_COVERING_BY_TAXONOMY,
#     COLOR_DESCRIPTION_TEMPLATES,
#     COLOR_DISTRIBUTIONS,
#     COLOR_PATTERNS,
#     COMMON_MARKINGS,
#     SPECIES_SPECIFIC_PATTERNS,
# )
# from cringegen.data.species_data import SPECIES_TAXONOMY

# Map old variables to new module structure
BASIC_COLORS = colors.BASIC_COLORS
BODY_COVERING_BY_TAXONOMY = taxonomy.BODY_COVERING_BY_TAXONOMY
COLOR_DESCRIPTION_TEMPLATES = colors.COLOR_DESCRIPTION_TEMPLATES
COLOR_DISTRIBUTIONS = colors.COLOR_DISTRIBUTIONS
COLOR_PATTERNS = colors.COLOR_PATTERNS
COMMON_MARKINGS = colors.COMMON_MARKINGS
SPECIES_SPECIFIC_PATTERNS = colors.SPECIES_SPECIFIC_PATTERNS
SPECIES_TAXONOMY = taxonomy.SPECIES_TAXONOMY


def get_body_covering_type(species: str) -> str:
    """Get the appropriate body covering type for a species.

    Args:
        species: The species name

    Returns:
        The body covering type (fur, scales, feathers, etc.)
    """
    # Get taxonomy for the species
    taxonomy = SPECIES_TAXONOMY.get(species.lower(), "default")

    # Get body covering type for this taxonomy
    covering_type = BODY_COVERING_BY_TAXONOMY.get(taxonomy, "fur")

    # If covering_type is a list, return the first element
    if isinstance(covering_type, list):
        return covering_type[0]
    return covering_type


def get_random_colors(count: int = 1, exclude: List[str] = None) -> List[str]:
    """Get random colors from the basic color list.

    Args:
        count: Number of colors to return
        exclude: Colors to exclude from selection

    Returns:
        List of color names
    """
    exclude = exclude or []
    available_colors = [c for c in BASIC_COLORS if c not in exclude]

    if count > len(available_colors):
        count = len(available_colors)

    return random.sample(available_colors, count)


def get_complementary_colors(base_color: str, count: int = 1) -> List[str]:
    """Get complementary colors that work well with the base color.

    Args:
        base_color: The base color to complement
        count: Number of complementary colors to return

    Returns:
        List of complementary color names
    """
    # Simple complementary color pairs
    complementary_pairs = {
        "red": ["green", "white", "black", "gold"],
        "orange": ["blue", "brown", "cream", "white"],
        "yellow": ["purple", "blue", "black", "gray"],
        "green": ["red", "brown", "white", "gold"],
        "blue": ["orange", "yellow", "white", "silver"],
        "purple": ["yellow", "white", "silver", "pink"],
        "pink": ["gray", "white", "black", "purple"],
        "brown": ["green", "blue", "cream", "white"],
        "gray": ["pink", "red", "blue", "black"],
        "black": ["red", "white", "yellow", "silver"],
        "white": ["black", "red", "blue", "pink"],
        "tan": ["brown", "black", "white", "red"],
        "cream": ["brown", "orange", "blue", "black"],
        "gold": ["black", "red", "green", "purple"],
        "silver": ["black", "blue", "purple", "red"],
    }

    # Get complementary colors or random colors if not found
    complements = complementary_pairs.get(base_color.lower(), [])

    if not complements or len(complements) < count:
        # Exclude the base color from random selection
        return get_random_colors(count, exclude=[base_color])

    # Return random subset of complementary colors
    return random.sample(complements, min(count, len(complements)))


def get_random_pattern(covering_type: str) -> str:
    """Get a random pattern appropriate for the covering type.

    Args:
        covering_type: Type of body covering (fur, scales, etc.)

    Returns:
        A pattern description
    """
    patterns = COLOR_PATTERNS.get(covering_type, COLOR_PATTERNS["fur"])
    return random.choice(patterns)


def get_random_marking(species: str) -> str:
    """Get a random marking appropriate for the species.

    Args:
        species: The species name

    Returns:
        A marking description
    """
    # Get taxonomy for the species
    taxonomy = SPECIES_TAXONOMY.get(species.lower(), "default")

    # Get markings for this taxonomy or use default
    markings = COMMON_MARKINGS.get(taxonomy, COMMON_MARKINGS["default"])

    return random.choice(markings)


def generate_color_description(
    species: str, colors: List[str] = None, pattern: str = None, marking: str = None
) -> str:
    """Generate a color description for a species.

    Args:
        species: Species name
        colors: List of colors (1-3 max)
        pattern: Pattern type (spotted, striped, etc.)
        marking: Specific marking type

    Returns:
        A natural language color description
    """
    # Default to random colors if none provided
    if not colors:
        colors = get_random_colors(random.randint(1, 3))

    # Limit to max 3 colors
    colors = colors[:3]

    # Get body covering type
    covering_type = get_body_covering_type(species)

    # Get pattern if not specified
    if not pattern:
        pattern = get_random_pattern(covering_type)

    # Get marking if not specified
    if not marking:
        marking = get_random_marking(species)

    # Check for species-specific patterns
    if species.lower() in SPECIES_SPECIFIC_PATTERNS and random.random() < 0.3:
        species_patterns = SPECIES_SPECIFIC_PATTERNS[species.lower()]
        return random.choice(species_patterns)

    # Select template based on number of colors
    if len(colors) == 1:
        # Single color (monotone)
        templates = COLOR_DESCRIPTION_TEMPLATES.get(
            covering_type, COLOR_DESCRIPTION_TEMPLATES["fur"]
        )
        template = random.choice(
            [t for t in templates if "{color1}" not in t and "{color2}" not in t]
        )
        return template.format(color=colors[0], pattern=pattern, marking=marking)

    elif len(colors) == 2:
        # Two colors (bicolor)
        template = random.choice(COLOR_DISTRIBUTIONS["bicolor"])
        color_desc = template.format(color1=colors[0], color2=colors[1])

        # Add the covering type
        return f"{color_desc} {covering_type}"

    else:
        # Three colors (tricolor)
        template = random.choice(COLOR_DISTRIBUTIONS["tricolor"])
        color_desc = template.format(color1=colors[0], color2=colors[1], color3=colors[2])

        # Add the covering type
        return f"{color_desc} {covering_type}"


def parse_color_input(color_input: str) -> List[str]:
    """Parse a color input string into a list of colors.

    Args:
        color_input: Comma-separated list of colors

    Returns:
        List of color names
    """
    if not color_input:
        return []

    colors = [c.strip().lower() for c in color_input.split(",")]

    # Validate colors against known colors
    valid_colors = []
    for color in colors:
        if color in BASIC_COLORS:
            valid_colors.append(color)

    return valid_colors
