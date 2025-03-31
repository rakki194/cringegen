"""
CringeGen data module

This module contains structured data used for generating image prompts.
It includes taxonomies, character data, environments, LoRA datasets,
and art style classifications.
"""

# Taxonomy and species data
from .taxonomy import (
    SPECIES_TAXONOMY,
    BODY_COVERING_BY_TAXONOMY,
    ANTHRO_SPECIES,
    FANTASY_SPECIES,
    POPULAR_ANTHRO_SPECIES,
    TAXONOMY_GROUPS,
    ANTHRO_DESCRIPTOR_SET,
)
from .anatomy import (
    MALE_ANATOMY,
    FEMALE_ANATOMY,
)

# Color system data
from .colors import (
    BASIC_COLORS,
    EXTENDED_COLORS,
    COLOR_PATTERNS,
    MATERIAL_COLORS,
    COLOR_DESCRIPTORS,
    COLOR_TO_RGB,
)

# Character data - directly define or import from character_taxonomy instead
from .character_taxonomy import (
    FAMOUS_FURRY_CHARACTERS,
    CHARACTER_TO_FULL_NAME,
    ALL_ANTHRO_CHARACTERS,
    ANIME_CHARACTER_TYPES,
    ANIME_CHARACTER_FEATURES,
)
from .characters import (
    INDIVIDUAL_CHARACTER_TEMPLATES,
    load_all_characters,
    get_individual_character,
    get_available_characters,
)

# New Character Taxonomy System
from .character_taxonomy import (
    CharacterType,
    FormType,
    Gender,
    CharacterFeature,
    SpeciesInfo,
    ClothingItem,
    AccessoryItem,
    CharacterArchetype,
    CharacterTemplate,
    SPECIES_INFO,
    CHARACTER_ARCHETYPES,
    FAMOUS_CHARACTER_TEMPLATES,
    get_species_info,
    get_appropriate_clothing,
    get_appropriate_accessories,
    generate_character_template,
    get_character_by_name,
    get_anatomical_terms,
    suggest_character_features,
)

# Environment and settings data
from .environments import (
    BACKGROUND_SETTINGS,
    TIME_OF_DAY,
    WEATHER_CONDITIONS,
    SEASONS,
    MOOD_DESCRIPTORS,
    SPECIES_HABITATS,
)

# Legacy LoRA data
from .lora import (
    get_available_loras,
    LORA_DATASETS,
    LORA_STYLES,
    LORA_ARTISTS,
    LORA_CHARACTERS,
    ALL_LORAS as LEGACY_LORAS,
)

# New LoRA taxonomy
from .lora_taxonomy import (
    LoRACategory,
    LoRASubcategory,
    LoRACompatibility,
    LoRAMetadata,
    LoRAVersion,
    ALL_LORAS,
    STYLE_LORAS,
    CHARACTER_LORAS,
    CONCEPT_LORAS,
    KINK_LORAS,
    POSE_LORAS,
    categorize_lora_by_filename,
    get_loras_by_category,
    get_loras_by_subcategory,
    get_loras_by_tags,
    get_compatible_loras,
    suggest_lora_strength,
)

# Art style data
from .styles import (
    ART_STYLES,
    FILM_TV_STYLES,
    GAME_ART_STYLES,
    VISUAL_AESTHETICS,
)

# New hierarchical style taxonomy
from .style_taxonomy import (
    STYLE_MEDIUM,
    STYLE_MOVEMENTS,
    STYLE_GENRES,
    STYLE_ANIMATION,
    STYLE_AESTHETICS,
    STYLE_RELATIONSHIPS,
    STYLE_ATTRIBUTES,
    get_style_by_medium,
    get_style_by_movement,
    get_related_styles,
    get_style_attributes,
    get_styles_by_attribute,
)

# For backward compatibility
ALL_LORAS.update(LEGACY_LORAS)

# Map legacy character templates to new format when needed
# This will be gradually populated as templates are requested
LEGACY_TO_NEW_CHARACTER_MAP = {}

__all__ = [
    # Taxonomy and species data
    "SPECIES_TAXONOMY",
    "BODY_COVERING_BY_TAXONOMY",
    "ANTHRO_SPECIES",
    "FANTASY_SPECIES",
    "POPULAR_ANTHRO_SPECIES",
    "TAXONOMY_GROUPS",
    "ANTHRO_DESCRIPTOR_SET",
    "MALE_ANATOMY",
    "FEMALE_ANATOMY",
    # Color system data
    "BASIC_COLORS",
    "EXTENDED_COLORS",
    "COLOR_PATTERNS",
    "MATERIAL_COLORS",
    "COLOR_DESCRIPTORS",
    "COLOR_TO_RGB",
    # Character data
    "FAMOUS_FURRY_CHARACTERS",
    "CHARACTER_TO_FULL_NAME",
    "ALL_ANTHRO_CHARACTERS",
    "ANIME_CHARACTER_TYPES",
    "ANIME_CHARACTER_FEATURES",
    "INDIVIDUAL_CHARACTER_TEMPLATES",
    "load_all_characters",
    "get_individual_character",
    "get_available_characters",
    # New Character Taxonomy
    "CharacterType",
    "FormType",
    "Gender",
    "CharacterFeature",
    "SpeciesInfo",
    "ClothingItem",
    "AccessoryItem",
    "CharacterArchetype",
    "CharacterTemplate",
    "SPECIES_INFO",
    "CHARACTER_ARCHETYPES",
    "FAMOUS_CHARACTER_TEMPLATES",
    "get_species_info",
    "get_appropriate_clothing",
    "get_appropriate_accessories",
    "generate_character_template",
    "get_character_by_name",
    "get_anatomical_terms",
    "suggest_character_features",
    # Environment and settings data
    "BACKGROUND_SETTINGS",
    "TIME_OF_DAY",
    "WEATHER_CONDITIONS",
    "SEASONS",
    "MOOD_DESCRIPTORS",
    "SPECIES_HABITATS",
    # Legacy LoRA data
    "get_available_loras",
    "LORA_DATASETS",
    "LORA_STYLES",
    "LORA_ARTISTS",
    "LORA_CHARACTERS",
    "LEGACY_LORAS",
    # New LoRA taxonomy
    "LoRACategory",
    "LoRASubcategory",
    "LoRACompatibility",
    "LoRAMetadata",
    "LoRAVersion",
    "ALL_LORAS",
    "STYLE_LORAS",
    "CHARACTER_LORAS",
    "CONCEPT_LORAS",
    "KINK_LORAS",
    "POSE_LORAS",
    "categorize_lora_by_filename",
    "get_loras_by_category",
    "get_loras_by_subcategory",
    "get_loras_by_tags",
    "get_compatible_loras",
    "suggest_lora_strength",
    # Art style data
    "ART_STYLES",
    "FILM_TV_STYLES",
    "GAME_ART_STYLES",
    "VISUAL_AESTHETICS",
    # New style taxonomy
    "STYLE_MEDIUM",
    "STYLE_MOVEMENTS",
    "STYLE_GENRES",
    "STYLE_ANIMATION",
    "STYLE_AESTHETICS",
    "STYLE_RELATIONSHIPS",
    "STYLE_ATTRIBUTES",
    "get_style_by_medium",
    "get_style_by_movement",
    "get_related_styles",
    "get_style_attributes",
    "get_styles_by_attribute",
]
