"""
Utilities for integrating data structures with NLP/NLTK tools.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import from centralized taxonomy module
from ...data.taxonomy import (
    FEMALE_ANATOMY,
    MALE_ANATOMY,
    SPECIES_TAXONOMY,
)

# Import from centralized characters module
from ...data.character_taxonomy import (
    ANIME_CHARACTER_TYPES,
    ANIME_CHARACTER_FEATURES,
)

# Import from centralized colors module
from ...data.colors import (
    SPECIES_COLORS,
)

# Import from centralized styles module
from ...data.styles import (
    ART_STYLES,
    FILM_TV_STYLES,
    ANIME_ART_STYLES,
    GAME_ART_STYLES,
    STYLE_DESCRIPTORS,
)

# These items might not exist in the current refactored structure
# They should be added to the appropriate modules in future updates
SPECIES_ACCESSORIES = {}  # To be moved to taxonomy.py
ANIME_ATMOSPHERE = {}
ANIME_CHARACTER_TO_DESCRIPTIVE = {}
ANIME_CLOTHING = {}
ANIME_SETTINGS = {}
ANIME_VISUAL_EFFECTS = {}
ANIME_GENRES = []
GAME_CHARACTER_FEATURES = {}
GAME_CHARACTER_TO_DESCRIPTIVE = {}
GAME_FRANCHISES = []
GAME_GENRES = {}
GAME_SETTINGS = {}
GAME_STUDIOS = []
GAME_UI_ELEMENTS = []
GAME_VISUAL_STYLES = {}
ART_STYLE_TO_DESCRIPTIVE = {}
COMICS_STYLES = {}
CULTURAL_REFERENCES = {}
DESIGN_STYLES = {}
FASHION_STYLES = {}
LITERATURE_STYLES = {}
MUSIC_STYLES = {}

# Temporarily define GAME_CHARACTER_TYPES here until imports are fixed
# This duplicates data from characters.py
GAME_CHARACTER_TYPES: Dict[str, List[str]] = {
    "protagonists": [
        "silent protagonist",
        "voiced protagonist",
        "player avatar",
        "chosen one",
    ],
    "player_classes": [
        "warrior",
        "mage",
        "rogue",
        "ranger",
    ],
    "npcs": [
        "quest giver",
        "merchant",
        "shopkeeper",
    ],
    "anthro_archetypes": [
        "animal warrior",
        "beast master",
        "shapeshifter",
    ],
}

# Original imports are commented out for reference during migration
# from cringegen.data.species_data import (
#     FEMALE_ANATOMY,
#     MALE_ANATOMY,
#     SPECIES_ACCESSORIES,
#     SPECIES_COLORS,
#     SPECIES_TAXONOMY,
# )
#
# try:
#     from cringegen.data.anime_data import (
#         ANIME_ART_STYLES,
#         ANIME_ATMOSPHERE,
#         ANIME_CHARACTER_FEATURES,
#         ANIME_CHARACTER_TO_DESCRIPTIVE,
#         ANIME_CHARACTER_TYPES,
#         ANIME_CLOTHING,
#         ANIME_GENRES,
#         ANIME_SETTINGS,
#         ANIME_VISUAL_EFFECTS,
#     )
# except ImportError:
#     # Default empty if module doesn't exist yet
#     ANIME_CHARACTER_TYPES = {}
#     ANIME_CHARACTER_FEATURES = {}
#     ANIME_CLOTHING = {}
#     ANIME_ART_STYLES = {}
#     ANIME_SETTINGS = {}
#     ANIME_ATMOSPHERE = {}
#     ANIME_VISUAL_EFFECTS = {}
#     ANIME_GENRES = []
#     ANIME_CHARACTER_TO_DESCRIPTIVE = {}
#
# try:
#     from cringegen.data.game_data import (
#         GAME_CHARACTER_FEATURES,
#         GAME_CHARACTER_TO_DESCRIPTIVE,
#         GAME_CHARACTER_TYPES,
#         GAME_FRANCHISES,
#         GAME_GENRES,
#         GAME_SETTINGS,
#         GAME_STUDIOS,
#         GAME_UI_ELEMENTS,
#         GAME_VISUAL_STYLES,
#     )
# except ImportError:
#     # Default empty if module doesn't exist yet
#     GAME_GENRES = {}
#     GAME_CHARACTER_TYPES = {}
#     GAME_CHARACTER_FEATURES = {}
#     GAME_SETTINGS = {}
#     GAME_VISUAL_STYLES = {}
#     GAME_UI_ELEMENTS = []
#     GAME_STUDIOS = []
#     GAME_FRANCHISES = []
#     GAME_CHARACTER_TO_DESCRIPTIVE = {}
#
# try:
#     from cringegen.data.media_data import (
#         ART_STYLE_TO_DESCRIPTIVE,
#         ART_STYLES,
#         COMICS_STYLES,
#         CULTURAL_REFERENCES,
#         DESIGN_STYLES,
#         FASHION_STYLES,
#         FILM_TV_STYLES,
#         LITERATURE_STYLES,
#         MUSIC_STYLES,
#     )
# except ImportError:
#     # Default empty if module doesn't exist yet
#     ART_STYLES = {}
#     FILM_TV_STYLES = {}
#     COMICS_STYLES = {}
#     DESIGN_STYLES = {}
#     MUSIC_STYLES = {}
#     FASHION_STYLES = {}
#     LITERATURE_STYLES = {}
#     ART_STYLE_TO_DESCRIPTIVE = {}
#     CULTURAL_REFERENCES = {}


def get_all_species() -> List[str]:
    """
    Get a list of all species from taxonomy data.

    Returns:
        List of species names
    """
    return list(SPECIES_TAXONOMY.keys())


def get_all_anime_character_types() -> List[str]:
    """
    Get a flattened list of all anime character types.

    Returns:
        List of anime character type names
    """
    all_types = []
    for category, types in ANIME_CHARACTER_TYPES.items():
        all_types.extend(types)
    return all_types


def get_all_game_character_types() -> List[str]:
    """
    Get a flattened list of all game character types.

    Returns:
        List of game character type names
    """
    all_types = []
    for category, types in GAME_CHARACTER_TYPES.items():
        all_types.extend(types)
    return all_types


def get_all_art_styles() -> List[str]:
    """
    Get a flattened list of all art styles.

    Returns:
        List of art style names
    """
    all_styles = []
    for category, styles in ART_STYLES.items():
        all_styles.extend(styles)
    return all_styles


def get_taxonomy_group(species: str) -> str:
    """
    Get the taxonomy group for a species.

    Args:
        species: Name of the species

    Returns:
        Taxonomy group name or "default" if not found
    """
    return SPECIES_TAXONOMY.get(species.lower(), "default")


def get_anatomical_terms(species: str, gender: str) -> List[str]:
    """
    Get anatomical terms for a species and gender.

    Args:
        species: Name of the species
        gender: Gender (male/female)

    Returns:
        List of anatomical terms
    """
    taxonomy_group = get_taxonomy_group(species)

    if gender.lower() == "male":
        return MALE_ANATOMY.get(taxonomy_group, MALE_ANATOMY["default"])
    elif gender.lower() == "female":
        return FEMALE_ANATOMY.get(taxonomy_group, FEMALE_ANATOMY["default"])
    else:
        # For non-binary or unspecified, combine both but exclude explicit terms
        male_terms = [
            term
            for term in MALE_ANATOMY.get(taxonomy_group, [])
            if "penis" not in term and "balls" not in term and "testicles" not in term
        ]
        female_terms = [
            term
            for term in FEMALE_ANATOMY.get(taxonomy_group, [])
            if "pussy" not in term and "pussy" not in term
        ]
        return list(set(male_terms + female_terms))


def get_species_accessories(species: str, gender: str) -> List[str]:
    """
    Get accessories for a species and gender.

    Args:
        species: Name of the species
        gender: Gender (male/female/neutral)

    Returns:
        List of accessory suggestions
    """
    taxonomy_group = get_taxonomy_group(species)

    gender = gender.lower()
    if gender not in ["male", "female", "neutral"]:
        gender = "neutral"

    return SPECIES_ACCESSORIES.get(taxonomy_group, SPECIES_ACCESSORIES["default"]).get(gender, [])


def get_species_colors(species: str) -> List[str]:
    """
    Get suggested colors for a species.

    Args:
        species: Name of the species

    Returns:
        List of color suggestions
    """
    taxonomy_group = get_taxonomy_group(species)

    species_colors = SPECIES_COLORS.get(taxonomy_group, {})
    if species.lower() in species_colors:
        return species_colors[species.lower()]

    return SPECIES_COLORS.get("default", [])


def get_anime_character_details(character_type: str) -> Dict[str, Any]:
    """
    Get detailed information about an anime character type.

    Args:
        character_type: The type of anime character

    Returns:
        Dictionary with details about the character type
    """
    # Find the category containing this character type
    category = None
    for cat, types in ANIME_CHARACTER_TYPES.items():
        if character_type.lower() in [t.lower() for t in types]:
            category = cat
            break

    # Get the description if available
    description = ANIME_CHARACTER_TO_DESCRIPTIVE.get(character_type.lower(), "")

    # Get suggested clothing
    clothing = []
    if "cat" in character_type.lower() or "neko" in character_type.lower():
        clothing = ANIME_CLOTHING.get("school_uniforms", [])
    elif "fox" in character_type.lower() or "kitsune" in character_type.lower():
        clothing = ANIME_CLOTHING.get("traditional_clothing", [])
    elif "magical" in character_type.lower():
        clothing = ANIME_CLOTHING.get("special_outfits", [])
    else:
        clothing = ANIME_CLOTHING.get("casual_clothing", [])

    # Get suggested settings
    settings = []
    if "school" in character_type.lower() or "student" in character_type.lower():
        settings = ANIME_SETTINGS.get("school_settings", [])
    elif "magical" in character_type.lower() or "fantasy" in character_type.lower():
        settings = ANIME_SETTINGS.get("fantasy_settings", [])
    else:
        settings = ANIME_SETTINGS.get("urban_settings", [])

    return {
        "type": character_type,
        "category": category,
        "description": description,
        "suggested_clothing": clothing[:5],
        "suggested_settings": settings[:5],
        "suggested_hair_colors": ANIME_CHARACTER_FEATURES.get("hair_colors", [])[:5],
        "suggested_eye_colors": ANIME_CHARACTER_FEATURES.get("eye_colors", [])[:5],
    }


def get_game_character_details(character_type: str) -> Dict[str, Any]:
    """
    Get detailed information about a game character type.

    Args:
        character_type: The type of game character

    Returns:
        Dictionary with details about the character type
    """
    # Find the category containing this character type
    category = None
    for cat, types in GAME_CHARACTER_TYPES.items():
        if character_type.lower() in [t.lower() for t in types]:
            category = cat
            break

    # Get the description if available
    description = GAME_CHARACTER_TO_DESCRIPTIVE.get(character_type.lower(), "")

    # Get suggested weapons based on character type
    weapons = []
    if any(word in character_type.lower() for word in ["mage", "wizard", "sorcerer", "warlock"]):
        weapons = [
            w
            for w in GAME_CHARACTER_FEATURES.get("weapons", [])
            if any(item in w for item in ["staff", "wand", "orb", "tome", "grimoire"])
        ]
    elif any(
        word in character_type.lower() for word in ["warrior", "knight", "paladin", "fighter"]
    ):
        weapons = [
            w
            for w in GAME_CHARACTER_FEATURES.get("weapons", [])
            if any(item in w for item in ["sword", "axe", "hammer", "mace", "shield"])
        ]
    elif any(word in character_type.lower() for word in ["rogue", "thief", "assassin"]):
        weapons = [
            w
            for w in GAME_CHARACTER_FEATURES.get("weapons", [])
            if any(item in w for item in ["dagger", "knife", "bow", "crossbow", "poison"])
        ]
    elif any(word in character_type.lower() for word in ["ranger", "hunter", "archer"]):
        weapons = [
            w
            for w in GAME_CHARACTER_FEATURES.get("weapons", [])
            if any(item in w for item in ["bow", "arrow", "longbow", "shortbow", "crossbow"])
        ]
    else:
        weapons = GAME_CHARACTER_FEATURES.get("weapons", [])[:5]

    # Get suggested armor based on character type
    armor = []
    if any(word in character_type.lower() for word in ["mage", "wizard", "sorcerer", "warlock"]):
        armor = [
            a
            for a in GAME_CHARACTER_FEATURES.get("armors", [])
            if any(item in a for item in ["robe", "cloth", "light"])
        ]
    elif any(word in character_type.lower() for word in ["warrior", "knight", "paladin"]):
        armor = [
            a
            for a in GAME_CHARACTER_FEATURES.get("armors", [])
            if any(item in a for item in ["plate", "heavy", "chainmail", "breastplate"])
        ]
    elif any(word in character_type.lower() for word in ["rogue", "thief", "assassin"]):
        armor = [
            a
            for a in GAME_CHARACTER_FEATURES.get("armors", [])
            if any(item in a for item in ["leather", "stealth", "light", "medium"])
        ]
    else:
        armor = GAME_CHARACTER_FEATURES.get("armors", [])[:5]

    return {
        "type": character_type,
        "category": category,
        "description": description,
        "suggested_weapons": weapons[:5],
        "suggested_armor": armor[:5],
        "suggested_accessories": GAME_CHARACTER_FEATURES.get("accessories", [])[:5],
        "suggested_body_types": GAME_CHARACTER_FEATURES.get("body_types", [])[:5],
    }


def get_art_style_details(style: str) -> Dict[str, Any]:
    """
    Get detailed information about an art style.

    Args:
        style: The name of the art style

    Returns:
        Dictionary with details about the art style
    """
    # Find the category containing this art style
    category = None
    for cat, styles in ART_STYLES.items():
        if style.lower() in [s.lower() for s in styles]:
            category = cat
            break

    # Get the description if available
    description = ART_STYLE_TO_DESCRIPTIVE.get(style.lower(), "")

    return {"style": style, "category": category, "description": description}


def classify_tag(tag: str) -> Dict[str, Any]:
    """
    Classify a tag into different categories based on our data structures.

    Args:
        tag: Tag to classify

    Returns:
        Dictionary with classification information
    """
    tag = tag.lower().strip()
    result = {
        "tag": tag,
        "is_species": False,
        "is_anime_character": False,
        "is_game_character": False,
        "is_art_style": False,
        "category": "other",
        "taxonomy_group": None,
        "details": {},
    }

    # Check if it's a species
    all_species = get_all_species()
    if tag in all_species:
        result["is_species"] = True
        result["category"] = "species"
        result["taxonomy_group"] = get_taxonomy_group(tag)
        return result

    # Check if it's an anime character type
    all_anime_characters = get_all_anime_character_types()
    for character in all_anime_characters:
        if tag == character.lower() or tag in character.lower():
            result["is_anime_character"] = True
            result["category"] = "anime_character"
            result["details"] = get_anime_character_details(character)
            return result

    # Check if it's a game character type
    all_game_characters = get_all_game_character_types()
    for character in all_game_characters:
        if tag == character.lower() or tag in character.lower():
            result["is_game_character"] = True
            result["category"] = "game_character"
            result["details"] = get_game_character_details(character)
            return result

    # Check if it's an art style
    all_art_styles = get_all_art_styles()
    for style in all_art_styles:
        if tag == style.lower() or tag in style.lower():
            result["is_art_style"] = True
            result["category"] = "art_style"
            result["details"] = get_art_style_details(style)
            return result

    # Check for broader categories
    if any(word in tag for word in ["anthro", "furry", "kemono"]):
        result["category"] = "anthro"
    elif any(word in tag for word in ["feral", "wild", "animal"]):
        result["category"] = "feral"
    elif any(word in tag for word in ["anime", "manga", "japanese"]):
        result["category"] = "anime_style"
    elif any(word in tag for word in ["game", "gaming", "video game"]):
        result["category"] = "game_style"
    elif any(word in tag for word in ["digital", "illustration", "drawing"]):
        result["category"] = "digital_art"
    elif any(word in tag for word in ["painting", "traditional", "oil", "watercolor"]):
        result["category"] = "traditional_art"

    return result


def enhance_tag_description(tag: str, gender: Optional[str] = None) -> str:
    """
    Enhance a tag with detailed description based on our data structures.

    Args:
        tag: Tag to enhance
        gender: Optional gender for species-specific enhancements

    Returns:
        Enhanced description of the tag
    """
    classification = classify_tag(tag)

    if classification["is_species"]:
        species = tag
        taxonomy = classification["taxonomy_group"]
        colors = get_species_colors(species)

        if gender:
            accessories = get_species_accessories(species, gender)
            result = f"a {species} ({taxonomy} species)"
            if colors:
                result += f", typically with {', '.join(colors[:3])} coloration"
            if accessories:
                result += f", commonly accessorized with {', '.join(accessories[:3])}"
            return result
        else:
            return f"a {species} ({taxonomy} species)"

    elif classification["is_anime_character"]:
        details = classification["details"]
        character_type = details.get("type", tag)
        description = details.get("description", "")

        if description:
            return f"{character_type}: {description}"
        else:
            return f"{character_type} (anime character type)"

    elif classification["is_game_character"]:
        details = classification["details"]
        character_type = details.get("type", tag)
        description = details.get("description", "")

        if description:
            return f"{character_type}: {description}"
        else:
            return f"{character_type} (game character type)"

    elif classification["is_art_style"]:
        details = classification["details"]
        style = details.get("style", tag)
        description = details.get("description", "")

        if description:
            return f"{style}: {description}"
        else:
            return f"{style} (art style)"

    else:
        # Return the tag as is if no enhancement is available
        return tag


def get_compatible_accessories(
    species: str, gender: str, character_type: Optional[str] = None
) -> List[str]:
    """
    Get accessories compatible with both species and character type.

    Args:
        species: Species name
        gender: Gender (male/female/neutral)
        character_type: Optional character type (anime or game)

    Returns:
        List of compatible accessories
    """
    # Get species-specific accessories
    species_acc = get_species_accessories(species, gender)

    # If no character type specified, return species accessories
    if not character_type:
        return species_acc

    # Check if it's an anime character
    anime_classification = classify_tag(character_type)
    if anime_classification["is_anime_character"]:
        anime_acc = anime_classification["details"].get("suggested_accessories", [])
        # Combine, prioritizing species accessories
        combined = species_acc + [a for a in anime_acc if a not in species_acc]
        return combined[:10]  # Limit to 10 suggestions

    # Check if it's a game character
    game_classification = classify_tag(character_type)
    if game_classification["is_game_character"]:
        game_acc = game_classification["details"].get("suggested_accessories", [])
        # Combine, prioritizing species accessories
        combined = species_acc + [a for a in game_acc if a not in species_acc]
        return combined[:10]  # Limit to 10 suggestions

    # Default to species accessories
    return species_acc


def get_tag_suggestions(base_tag: str, count: int = 5) -> List[str]:
    """
    Get related tag suggestions based on a base tag.

    Args:
        base_tag: The tag to get suggestions for
        count: Number of suggestions to return

    Returns:
        List of related tag suggestions
    """
    classification = classify_tag(base_tag)
    category = classification["category"]

    suggestions = []

    if category == "species":
        # Get same taxonomy group species
        species = base_tag
        taxonomy = classification["taxonomy_group"]
        related_species = [s for s, t in SPECIES_TAXONOMY.items() if t == taxonomy and s != species]
        suggestions.extend(related_species[:count])

        # Add accessories
        accessories = get_species_accessories(species, "neutral")
        suggestions.extend(accessories[: count - len(suggestions)])

        # Add colors
        colors = get_species_colors(species)
        suggestions.extend([f"{color} fur" for color in colors[: count - len(suggestions)]])

    elif category == "anime_character":
        # Get related anime character types
        character_details = classification["details"]
        character_category = character_details.get("category")
        if character_category:
            related_characters = [
                c
                for c in ANIME_CHARACTER_TYPES.get(character_category, [])
                if c.lower() != base_tag
            ]
            suggestions.extend(related_characters[:count])

        # Add clothing suggestions
        suggestions.extend(
            character_details.get("suggested_clothing", [])[: count - len(suggestions)]
        )

        # Add setting suggestions
        suggestions.extend(
            character_details.get("suggested_settings", [])[: count - len(suggestions)]
        )

        # Add hair and eye colors
        suggestions.extend(
            character_details.get("suggested_hair_colors", [])[: count - len(suggestions)]
        )
        suggestions.extend(
            character_details.get("suggested_eye_colors", [])[: count - len(suggestions)]
        )

    elif category == "game_character":
        # Get related game character types
        character_details = classification["details"]
        character_category = character_details.get("category")
        if character_category:
            related_characters = [
                c for c in GAME_CHARACTER_TYPES.get(character_category, []) if c.lower() != base_tag
            ]
            suggestions.extend(related_characters[:count])

        # Add weapon suggestions
        suggestions.extend(
            character_details.get("suggested_weapons", [])[: count - len(suggestions)]
        )

        # Add armor suggestions
        suggestions.extend(character_details.get("suggested_armor", [])[: count - len(suggestions)])

        # Add accessory suggestions
        suggestions.extend(
            character_details.get("suggested_accessories", [])[: count - len(suggestions)]
        )

    elif category == "art_style":
        # Get related art styles from the same category
        style_details = classification["details"]
        style_category = style_details.get("category")
        if style_category:
            related_styles = [
                s for s in ART_STYLES.get(style_category, []) if s.lower() != base_tag
            ]
            suggestions.extend(related_styles[:count])

    elif category == "anthro":
        # Suggest anthro-related tags
        suggestions.extend(
            [
                "anthro",
                "furry",
                "anthropomorphic",
                "furry art",
                "kemono",
                "talking animal",
                "bipedal",
                "humanoid animal",
                "furry character",
            ][:count]
        )

        # Add some species
        suggestions.extend(
            ["anthro fox", "anthro wolf", "anthro cat", "anthro dragon"][: count - len(suggestions)]
        )

    elif category == "anime_style":
        # Suggest anime-related tags
        suggestions.extend(
            [
                "anime style",
                "manga style",
                "japanese animation",
                "anime aesthetic",
                "anime character",
                "anime background",
                "detailed anime",
                "high quality anime",
                "anime illustration",
            ][:count]
        )

        # Add some character types
        suggestions.extend(
            ["cat girl", "magical girl", "anime girl", "anime boy"][: count - len(suggestions)]
        )

    # If we still need more suggestions, add quality-related tags
    if len(suggestions) < count:
        quality_tags = [
            "masterpiece",
            "best quality",
            "high quality",
            "detailed",
            "highly detailed",
            "ultra detailed",
            "intricate details",
            "professional",
            "award-winning",
            "trending",
            "8k",
        ]
        suggestions.extend(quality_tags[: count - len(suggestions)])

    return suggestions[:count]


def analyze_tags_with_data_structures(tags: List[str]) -> Dict[str, Any]:
    """
    Analyze a list of tags using our data structures.

    Args:
        tags: List of tags to analyze

    Returns:
        Dictionary with analysis results
    """
    results = {
        "species": [],
        "anime_characters": [],
        "game_characters": [],
        "art_styles": [],
        "other": [],
        "taxonomy_groups": set(),
        "has_anthro": False,
        "has_feral": False,
        "has_anime": False,
        "has_game": False,
        "suggested_additions": [],
        "dominant_category": None,
    }

    category_counts = defaultdict(int)

    for tag in tags:
        classification = classify_tag(tag)
        category = classification["category"]
        category_counts[category] += 1

        if classification["is_species"]:
            results["species"].append(tag)
            if classification["taxonomy_group"]:
                results["taxonomy_groups"].add(classification["taxonomy_group"])
        elif classification["is_anime_character"]:
            results["anime_characters"].append(tag)
            results["has_anime"] = True
        elif classification["is_game_character"]:
            results["game_characters"].append(tag)
            results["has_game"] = True
        elif classification["is_art_style"]:
            results["art_styles"].append(tag)
        else:
            results["other"].append(tag)

            # Check for broader categories
            if category == "anthro":
                results["has_anthro"] = True
            elif category == "feral":
                results["has_feral"] = True
            elif category == "anime_style":
                results["has_anime"] = True
            elif category == "game_style":
                results["has_game"] = True

    # Determine dominant category
    if category_counts:
        results["dominant_category"] = max(category_counts.items(), key=lambda x: x[1])[0]

    # Generate suggested additions based on dominant category
    if results["dominant_category"]:
        if results["species"] and not results["has_anthro"] and not results["has_feral"]:
            # Suggest anthro/feral clarification
            results["suggested_additions"].append("anthro" if "anthro" not in tags else "feral")

        if results["dominant_category"] == "species" or results["dominant_category"] == "anthro":
            # Suggest a common accessory for the first species
            if results["species"]:
                species_acc = get_species_accessories(results["species"][0], "neutral")
                if species_acc:
                    results["suggested_additions"].append(species_acc[0])

        if results["dominant_category"] == "anime_character" or results["has_anime"]:
            # Suggest a common anime setting or clothing
            if not any(
                item in " ".join(tags).lower() for item in ["school", "classroom", "uniform"]
            ):
                results["suggested_additions"].append("school uniform")
            if not any(item in " ".join(tags).lower() for item in ["cherry blossom", "sakura"]):
                results["suggested_additions"].append("cherry blossoms")

        if results["dominant_category"] == "game_character" or results["has_game"]:
            # Suggest a weapon or armor if not present
            if not any(
                item in " ".join(tags).lower() for item in ["sword", "weapon", "staff", "bow"]
            ):
                results["suggested_additions"].append("sword")
            if not any(item in " ".join(tags).lower() for item in ["armor", "outfit", "costume"]):
                results["suggested_additions"].append("armor")

        if results["dominant_category"] == "art_style":
            # Suggest quality tags if not present
            if not any(
                item in " ".join(tags).lower()
                for item in ["masterpiece", "high quality", "detailed"]
            ):
                results["suggested_additions"].append("masterpiece")

    # Convert taxonomy_groups to list for JSON serialization
    results["taxonomy_groups"] = list(results["taxonomy_groups"])

    return results
