"""
Character Taxonomy System for cringegen

This module provides a comprehensive, hierarchical organization for all character-related data.
It consolidates and structures character information previously distributed across multiple files.

Key components:
- Character class hierarchies
- Species classification and relationships
- Clothing and accessory taxonomies
- Character features and attributes
- Compatibility and relationship data
- Helper functions for character generation and manipulation
"""

from typing import Dict, List, Set, Tuple, Optional, Union, Any
from enum import Enum, auto
from dataclasses import dataclass, field
import random

# These imports need to be updated based on the actual location of these variables
# in your codebase. The following is a placeholder - adjust as needed.
from .taxonomy import ANTHRO_SPECIES, FANTASY_SPECIES
from .anatomy import MALE_ANATOMY, FEMALE_ANATOMY

# Define these directly instead of importing from .characters to avoid circular imports
FAMOUS_FURRY_CHARACTERS = {}
CHARACTER_TO_FULL_NAME = {}
ALL_ANTHRO_CHARACTERS = []
ANIME_CHARACTER_TYPES = {
    "kemonomimi": [],
    "monster_people": [],
    "fantasy_races": [],
    "character_archetypes": [],
}
ANIME_CHARACTER_FEATURES = {
    "hair_colors": [],
    "hair_styles": [],
    "eye_colors": [],
    "expressions": [],
}

# Define this since it's used below but might be imported from elsewhere
SPECIES_ACCESSORIES = {"default": {}}

# -------------------------------------------------------------------------
# Core Character Taxonomy Classes
# -------------------------------------------------------------------------


class CharacterType(Enum):
    """Enumeration of high-level character types."""

    ANTHRO = auto()  # Anthropomorphic animal characters
    HUMAN = auto()  # Human characters
    KEMONOMIMI = auto()  # Humans with animal features
    MONSTER = auto()  # Monster or creature characters
    FANTASY = auto()  # Fantasy race characters
    ROBOT = auto()  # Robotic or synthetic characters
    HYBRID = auto()  # Hybrid characters (mix of multiple types)


class FormType(Enum):
    """Enumeration of character form types."""

    ANTHRO = auto()  # Anthropomorphic (bipedal, humanoid proportions)
    FERAL = auto()  # Feral (quadrupedal, animal proportions)
    SEMI_ANTHRO = auto()  # Mix of anthro and feral features
    TAUR = auto()  # Centaur-like (upper body anthro, lower body feral)
    HUMANOID = auto()  # Primarily human shape with non-human elements


class Gender(Enum):
    """Enumeration of character genders."""

    MALE = auto()
    FEMALE = auto()
    NONBINARY = auto()
    AMBIGUOUS = auto()
    FLUID = auto()


@dataclass
class CharacterFeature:
    """A specific character feature with category and variants."""

    name: str
    category: str
    variants: List[str] = field(default_factory=list)
    description: str = ""
    compatibility: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SpeciesInfo:
    """Comprehensive information about a species."""

    name: str  # Species name (e.g., "wolf")
    taxonomy_group: str  # Taxonomy group (e.g., "canine")
    body_covering: str  # Primary body covering (e.g., "fur")
    is_fantasy: bool = False  # Whether it's a fantasy species
    is_popular: bool = False  # Whether it's a commonly used species
    anatomical_terms: Dict[str, List[str]] = field(
        default_factory=dict
    )  # Gender-specific anatomical terms
    common_features: List[str] = field(default_factory=list)  # Common physical features
    common_accessories: Dict[str, List[str]] = field(
        default_factory=dict
    )  # Common accessories by form type and gender
    common_colors: List[str] = field(default_factory=list)  # Common coloration patterns
    related_species: List[str] = field(default_factory=list)  # Related or similar species


@dataclass
class ClothingItem:
    """A clothing item with metadata."""

    name: str
    category: str
    formality_level: int = 0  # 0-10 scale (0=casual, 10=formal)
    gender_association: Optional[Gender] = None  # None means unisex/neutral
    season_appropriate: List[str] = field(default_factory=lambda: ["any"])
    compatible_species_types: List[str] = field(default_factory=lambda: ["any"])
    incompatible_species_types: List[str] = field(default_factory=list)
    required_body_parts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class AccessoryItem:
    """An accessory item with metadata."""

    name: str
    category: str
    placement: str  # Where on the body it's worn
    gender_association: Optional[Gender] = None  # None means unisex/neutral
    compatible_species_types: List[str] = field(default_factory=lambda: ["any"])
    incompatible_species_types: List[str] = field(default_factory=list)
    required_body_parts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class CharacterArchetype:
    """A character archetype with typical traits and features."""

    name: str
    category: str
    typical_personality: List[str] = field(default_factory=list)
    typical_appearance: List[str] = field(default_factory=list)
    typical_accessories: List[str] = field(default_factory=list)
    typical_clothing: List[str] = field(default_factory=list)
    common_species: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class CharacterTemplate:
    """A template for generating a complete character."""

    character_type: CharacterType
    form_type: FormType
    gender: Gender
    species: Optional[str] = None
    archetype: Optional[str] = None
    features: List[str] = field(default_factory=list)
    clothing: List[str] = field(default_factory=list)
    accessories: List[str] = field(default_factory=list)
    colors: Dict[str, str] = field(default_factory=dict)
    personality: List[str] = field(default_factory=list)
    # New fields for character-specific attributes
    model_tags: Dict[str, str] = field(
        default_factory=dict
    )  # Model-specific tags (e621, danbooru, etc.)
    appearance_traits: List[str] = field(default_factory=list)  # Specific appearance traits
    nsfw_traits: List[str] = field(default_factory=list)  # NSFW-specific traits
    is_canon_character: bool = False  # Whether this is a specific canon character
    hardcore_tags: List[str] = field(default_factory=list)  # Species-appropriate explicit tags for 'hardcore' intensity


# -------------------------------------------------------------------------
# Data Dictionaries
# -------------------------------------------------------------------------

# Simple mapping of species to taxonomy groups
# This is a placeholder - you should adapt this to your actual data
SPECIES_TAXONOMY = {
    "wolf": "canine",
    "fox": "canine",
    "dog": "canine",
    "cat": "feline",
    "tiger": "feline",
    "lion": "feline",
    "dragon": "reptile",
    "rabbit": "lagomorph",
    "deer": "cervid",
}

# Simple mapping of taxonomy to body covering
BODY_COVERING_BY_TAXONOMY = {
    "canine": "fur",
    "feline": "fur",
    "reptile": "scales",
    "lagomorph": "fur",
    "cervid": "fur",
}

# Mapping of character names to full names
CHARACTER_NAME_MAP = CHARACTER_TO_FULL_NAME

# Map kemonomimi types from existing data
KEMONOMIMI_TYPES = ANIME_CHARACTER_TYPES.get("kemonomimi", [])
MONSTER_PEOPLE = ANIME_CHARACTER_TYPES.get("monster_people", [])
FANTASY_RACES = ANIME_CHARACTER_TYPES.get("fantasy_races", [])

# Map character feature data
HAIR_COLORS = ANIME_CHARACTER_FEATURES.get("hair_colors", [])
HAIR_STYLES = ANIME_CHARACTER_FEATURES.get("hair_styles", [])
EYE_COLORS = ANIME_CHARACTER_FEATURES.get("eye_colors", [])
CHARACTER_EXPRESSIONS = ANIME_CHARACTER_FEATURES.get("expressions", [])
CHARACTER_FEATURES = ANIME_CHARACTER_FEATURES

# Map anime archetypes
COMMON_ANIME_ARCHETYPES = ANIME_CHARACTER_TYPES.get("character_archetypes", {})

# -------------------------------------------------------------------------
# Species information dictionary
# -------------------------------------------------------------------------
SPECIES_INFO: Dict[str, SpeciesInfo] = {}

# Initialize with existing data
for species, taxonomy in SPECIES_TAXONOMY.items():
    body_covering = BODY_COVERING_BY_TAXONOMY.get(taxonomy, "fur")
    is_fantasy = species in FANTASY_SPECIES
    is_popular = species in ANTHRO_SPECIES

    # Get anatomical terms if available
    anatomical_terms = {
        "male": MALE_ANATOMY.get(taxonomy, []),
        "female": FEMALE_ANATOMY.get(taxonomy, []),
    }

    # Get accessories if available
    accessories_dict = {}
    if taxonomy in SPECIES_ACCESSORIES:
        accessories_dict = SPECIES_ACCESSORIES[taxonomy]
    elif "default" in SPECIES_ACCESSORIES:
        accessories_dict = SPECIES_ACCESSORIES["default"]

    SPECIES_INFO[species] = SpeciesInfo(
        name=species,
        taxonomy_group=taxonomy,
        body_covering=body_covering,
        is_fantasy=is_fantasy,
        is_popular=is_popular,
        anatomical_terms=anatomical_terms,
        common_accessories=accessories_dict,
    )

# -------------------------------------------------------------------------
# Clothing and Accessory Categories
# -------------------------------------------------------------------------

# Clothing categories
CLOTHING_CATEGORIES = [
    "tops",
    "bottoms",
    "dresses",
    "outerwear",
    "footwear",
    "underwear",
    "swimwear",
    "formal",
    "uniform",
    "athletic",
]

# Initialize clothing items dictionary (to be expanded)
CLOTHING_ITEMS: Dict[str, ClothingItem] = {}

# Accessory categories
ACCESSORY_CATEGORIES = [
    "jewelry",
    "headwear",
    "eyewear",
    "neckwear",
    "handwear",
    "footwear",
    "tech",
    "weapons",
    "bags",
    "decorative",
]

# Initialize accessory items dictionary (to be expanded)
ACCESSORY_ITEMS: Dict[str, AccessoryItem] = {}

# -------------------------------------------------------------------------
# Character Archetypes
# -------------------------------------------------------------------------

# Character archetypes from existing data
CHARACTER_ARCHETYPES: Dict[str, CharacterArchetype] = {}

# Initialize with existing data
for category, archetypes in ANIME_CHARACTER_TYPES.items():
    if category == "character_archetypes":
        for archetype in archetypes:
            CHARACTER_ARCHETYPES[archetype.lower()] = CharacterArchetype(
                name=archetype, category="anime", description=f"An anime character archetype"
            )

# -------------------------------------------------------------------------
# Famous Character Templates
# -------------------------------------------------------------------------

# Famous character templates
FAMOUS_CHARACTER_TEMPLATES: Dict[str, CharacterTemplate] = {}

# Initialize with existing famous characters data
for source, characters in FAMOUS_FURRY_CHARACTERS.items():
    if isinstance(characters, dict):
        # Handle nested dictionaries like games -> game_names -> characters
        for subsource, char_list in characters.items():
            for character in char_list:
                # Use CHARACTER_NAME_MAP to get expanded name if available
                full_name = CHARACTER_NAME_MAP.get(character.lower(), character)

                # Determine species (simplified - would need enhancement)
                species = None
                for s in ANTHRO_SPECIES:
                    if s in full_name.lower():
                        species = s
                        break

                # Create basic template (to be expanded with more data)
                FAMOUS_CHARACTER_TEMPLATES[character.lower()] = CharacterTemplate(
                    character_type=CharacterType.ANTHRO,
                    form_type=FormType.ANTHRO,  # Assume anthro by default
                    gender=(
                        Gender.MALE
                        if "he" in full_name.lower()
                        else Gender.FEMALE if "she" in full_name.lower() else Gender.AMBIGUOUS
                    ),
                    species=species,
                    archetype=None,
                )
    else:
        # Handle flat lists
        for character in characters:
            # Skip if already added
            if character.lower() in FAMOUS_CHARACTER_TEMPLATES:
                continue

            # Use CHARACTER_NAME_MAP to get expanded name if available
            full_name = CHARACTER_NAME_MAP.get(character.lower(), character)

            # Determine species (simplified - would need enhancement)
            species = None
            for s in ANTHRO_SPECIES:
                if s in full_name.lower():
                    species = s
                    break

            # Create basic template (to be expanded with more data)
            FAMOUS_CHARACTER_TEMPLATES[character.lower()] = CharacterTemplate(
                character_type=CharacterType.ANTHRO,
                form_type=FormType.ANTHRO,  # Assume anthro by default
                gender=(
                    Gender.MALE
                    if "he" in full_name.lower()
                    else Gender.FEMALE if "she" in full_name.lower() else Gender.AMBIGUOUS
                ),
                species=species,
                archetype=None,
            )

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


def get_species_info(species_name: str) -> Optional[SpeciesInfo]:
    """Get comprehensive information about a species."""
    species_lower = species_name.lower()
    return SPECIES_INFO.get(species_lower)


def get_appropriate_clothing(
    species: str, gender: Gender, form_type: FormType, formality: int = 5, season: str = "any"
) -> List[ClothingItem]:
    """Get appropriate clothing for a character based on their attributes."""
    # Implementation would filter CLOTHING_ITEMS based on parameters
    # For now, return empty list (to be implemented)
    return []


def get_appropriate_accessories(
    species: str, gender: Gender, form_type: FormType, outfit: List[ClothingItem] = None
) -> List[AccessoryItem]:
    """Get appropriate accessories for a character based on their attributes."""
    # Implementation would filter ACCESSORY_ITEMS based on parameters
    # For now, return empty list (to be implemented)
    return []


def generate_character_template(
    character_type: CharacterType = None,
    species: str = None,
    gender: Gender = None,
    form_type: FormType = None,
    archetype: str = None,
) -> CharacterTemplate:
    """Generate a complete character template based on provided parameters."""
    # Fill in defaults for missing parameters
    if character_type is None:
        character_type = CharacterType.ANTHRO
    if species is None and character_type == CharacterType.ANTHRO:
        # Pick a random popular species
        species = random.choice(ANTHRO_SPECIES)
    if gender is None:
        gender = random.choice([Gender.MALE, Gender.FEMALE])
    if form_type is None:
        form_type = FormType.ANTHRO

    # Create base template
    template = CharacterTemplate(
        character_type=character_type,
        form_type=form_type,
        gender=gender,
        species=species,
        archetype=archetype,
    )

    # Add appropriate clothing and accessories
    # (To be implemented with appropriate logic)

    return template


def get_character_by_name(name: str) -> Optional[CharacterTemplate]:
    """Get a famous character template by name."""
    name_lower = name.lower()

    # Try direct lookup
    if name_lower in FAMOUS_CHARACTER_TEMPLATES:
        return FAMOUS_CHARACTER_TEMPLATES[name_lower]

    # Try using the name map
    if name_lower in CHARACTER_NAME_MAP:
        mapped_name = CHARACTER_NAME_MAP[name_lower].lower()
        return FAMOUS_CHARACTER_TEMPLATES.get(mapped_name)

    return None


def get_anatomical_terms(species: str, gender: Gender) -> List[str]:
    """Get appropriate anatomical terms for a species and gender."""
    species_info = get_species_info(species)
    if not species_info:
        return []

    if gender == Gender.MALE:
        return species_info.anatomical_terms.get("male", [])
    elif gender == Gender.FEMALE:
        return species_info.anatomical_terms.get("female", [])
    else:
        # For non-binary or ambiguous, combine both sets
        male_terms = species_info.anatomical_terms.get("male", [])
        female_terms = species_info.anatomical_terms.get("female", [])
        return list(set(male_terms + female_terms))


def suggest_character_features(species: str, gender: Gender) -> List[str]:
    """Suggest appropriate physical features for a character."""
    suggested_features = []

    # Add species-specific features
    species_info = get_species_info(species)
    if species_info and species_info.common_features:
        suggested_features.extend(species_info.common_features)

    # Add gender-appropriate features
    if gender in [Gender.MALE, Gender.FEMALE]:
        gender_str = gender.name.lower()
        # Check if CHARACTER_FEATURES has gender-specific features
        for feature_category, features in CHARACTER_FEATURES.items():
            if isinstance(features, dict) and gender_str in features:
                suggested_features.extend(features[gender_str])

    return suggested_features
