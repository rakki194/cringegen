"""
Hierarchical LoRA Taxonomy System for cringegen

This module provides a hierarchical taxonomy of LoRAs (Low-Rank Adaptations) used in image generation.
It organizes LoRAs by category, purpose, and compatibility with a rich metadata system
to enable more sophisticated LoRA selection and combination.

Key components:
- LoRA categories (style, character, concept, kink, etc.)
- Detailed metadata for each LoRA
- Version tracking and compatibility information
- Helper functions for LoRA selection
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Union
import re
from enum import Enum, auto
from datetime import datetime

# =============================================================================
# LoRA Taxonomy Core Types
# =============================================================================


class LoRACategory(Enum):
    """Main categories of LoRAs"""

    STYLE = auto()
    CHARACTER = auto()
    CONCEPT = auto()
    KINK = auto()
    POSE = auto()
    MIXED = auto()
    UNKNOWN = auto()


class LoRASubcategory(Enum):
    """Subcategories for better organization"""

    # Style subcategories
    ARTIST = auto()
    ART_MOVEMENT = auto()
    AESTHETIC = auto()
    MEDIUM = auto()

    # Character subcategories
    SPECIFIC_CHARACTER = auto()
    CHARACTER_TYPE = auto()
    SPECIES = auto()

    # Concept subcategories
    OBJECT = auto()
    ENVIRONMENT = auto()
    CLOTHING = auto()
    ACCESSORY = auto()
    ACTION = auto()

    # Kink subcategories
    FETISH = auto()
    BODY_MODIFICATION = auto()
    TRANSFORMATION = auto()

    # Pose subcategories
    POSITION = auto()
    EXPRESSION = auto()
    COMPOSITION = auto()

    # Other
    GENERAL = auto()
    MIXED = auto()
    UNKNOWN = auto()


class LoRACompatibility(Enum):
    """Compatibility with model versions"""

    SD1_5 = auto()
    SD2_1 = auto()
    SDXL = auto()
    UNIVERSAL = auto()  # Works with multiple model versions


# =============================================================================
# LoRA Metadata Structure
# =============================================================================


class LoRAVersion:
    """Version information for a LoRA"""

    def __init__(
        self,
        version_num: str,
        release_date: Optional[datetime] = None,
        changes: Optional[List[str]] = None,
        training_settings: Optional[Dict[str, Any]] = None,
        revision: Optional[str] = None,
    ):
        self.version_num = version_num
        self.release_date = release_date
        self.changes = changes or []
        self.training_settings = training_settings or {}
        self.revision = revision

    def __str__(self) -> str:
        return f"v{self.version_num}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "version_num": self.version_num,
            "release_date": self.release_date.isoformat() if self.release_date else None,
            "changes": self.changes,
            "training_settings": self.training_settings,
            "revision": self.revision,
        }


class LoRAMetadata:
    """Rich metadata for a LoRA"""

    def __init__(
        self,
        name: str,
        filename: str,
        category: LoRACategory,
        subcategory: Optional[LoRASubcategory] = None,
        creator: Optional[str] = None,
        version: Optional[LoRAVersion] = None,
        description: Optional[str] = None,
        strength_range: Optional[Tuple[float, float]] = None,
        trigger_terms: Optional[List[str]] = None,
        compatibility: Optional[List[LoRACompatibility]] = None,
        tags: Optional[List[str]] = None,
        related_loras: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        preview_image: Optional[str] = None,
        nsfw: bool = False,
        excluded_from_random: bool = False,
    ):
        self.name = name
        self.filename = filename
        self.category = category
        self.subcategory = subcategory or LoRASubcategory.UNKNOWN
        self.creator = creator
        self.version = version or LoRAVersion("1.0")
        self.description = description or ""
        self.strength_range = strength_range or (0.3, 0.7)
        self.trigger_terms = trigger_terms or []
        self.compatibility = compatibility or [LoRACompatibility.SDXL]
        self.tags = tags or []
        self.related_loras = related_loras or []
        self.source_url = source_url
        self.preview_image = preview_image
        self.nsfw = nsfw
        self.excluded_from_random = excluded_from_random

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "filename": self.filename,
            "category": self.category.name,
            "subcategory": self.subcategory.name,
            "creator": self.creator,
            "version": self.version.to_dict(),
            "description": self.description,
            "strength_range": self.strength_range,
            "trigger_terms": self.trigger_terms,
            "compatibility": [c.name for c in self.compatibility],
            "tags": self.tags,
            "related_loras": self.related_loras,
            "source_url": self.source_url,
            "preview_image": self.preview_image,
            "nsfw": self.nsfw,
            "excluded_from_random": self.excluded_from_random,
        }

    def optimal_strength(self) -> float:
        """Return the optimal strength for this LoRA"""
        return (self.strength_range[0] + self.strength_range[1]) / 2

    def __str__(self) -> str:
        return f"{self.name} {self.version} [{self.category.name}]"


# =============================================================================
# LoRA Collections
# =============================================================================


# Helper for parsing version information from filename
def _parse_version(filename: str) -> str:
    """Extract version information from filename"""
    version_match = re.search(r"v(\d+[a-z]?)(?:[\._-]?s\d+)?(?:\.safetensors)?$", filename)
    if version_match:
        return version_match.group(1)
    return "1.0"  # Default version


# Style LoRAs - focused on artistic styles
STYLE_LORAS: Dict[str, LoRAMetadata] = {
    "pixel_art": LoRAMetadata(
        name="Pixel Art",
        filename="pixel_art-v1.safetensors",
        category=LoRACategory.STYLE,
        subcategory=LoRASubcategory.MEDIUM,
        creator="Unknown",
        version=LoRAVersion("1.0"),
        description="Creates pixel art style graphics with visible pixels and limited color palette",
        strength_range=(0.4, 0.7),
        trigger_terms=["pixel art", "pixelated", "8-bit", "16-bit"],
        tags=["retro", "gaming", "digital", "blocky", "limited palette"],
        compatibility=[LoRACompatibility.SDXL],
    ),
    "watercolor": LoRAMetadata(
        name="Watercolor Painting",
        filename="watercolor-v1.safetensors",
        category=LoRACategory.STYLE,
        subcategory=LoRASubcategory.MEDIUM,
        creator="Unknown",
        version=LoRAVersion("1.0"),
        description="Creates soft, flowing watercolor painting effects with transparent colors",
        strength_range=(0.3, 0.6),
        trigger_terms=["watercolor", "watercolor painting", "wet media"],
        tags=["painting", "traditional", "flowing", "transparent", "soft"],
        compatibility=[LoRACompatibility.SDXL],
    ),
    "impressionism": LoRAMetadata(
        name="Impressionism",
        filename="impressionism-v1.safetensors",
        category=LoRACategory.STYLE,
        subcategory=LoRASubcategory.ART_MOVEMENT,
        creator="ArtisticAI",
        version=LoRAVersion("1.0"),
        description="Creates impressionist painting style with visible brushstrokes and light effects",
        strength_range=(0.4, 0.7),
        trigger_terms=["impressionism", "impressionist painting", "impressionist style"],
        tags=["painting", "traditional", "light", "outdoor", "brushstrokes"],
        compatibility=[LoRACompatibility.SDXL],
    ),
    "pop_art": LoRAMetadata(
        name="Pop Art",
        filename="pop_art-v1.safetensors",
        category=LoRACategory.STYLE,
        subcategory=LoRASubcategory.ART_MOVEMENT,
        creator="StyleLab",
        version=LoRAVersion("1.0"),
        description="Creates bold, colorful pop art inspired by artists like Warhol and Lichtenstein",
        strength_range=(0.5, 0.8),
        trigger_terms=["pop art", "pop art style", "comic art"],
        tags=["bold", "colorful", "modern", "graphic", "commercial"],
        compatibility=[LoRACompatibility.SDXL],
    ),
    "anime": LoRAMetadata(
        name="Anime Style",
        filename="anime-v2.safetensors",
        category=LoRACategory.STYLE,
        subcategory=LoRASubcategory.AESTHETIC,
        creator="AnimeAI",
        version=LoRAVersion("2.0"),
        description="Creates Japanese anime-style artwork with characteristic features",
        strength_range=(0.4, 0.7),
        trigger_terms=["anime", "anime style", "manga style"],
        tags=["animation", "japanese", "cartoon", "2D", "stylized"],
        compatibility=[LoRACompatibility.SDXL],
    ),
    "cyberpunk": LoRAMetadata(
        name="Cyberpunk Aesthetic",
        filename="cyberpunk-v1.safetensors",
        category=LoRACategory.STYLE,
        subcategory=LoRASubcategory.AESTHETIC,
        creator="FutureVisions",
        version=LoRAVersion("1.0"),
        description="Creates cyberpunk aesthetic with neon colors, high-tech, and dystopian elements",
        strength_range=(0.4, 0.7),
        trigger_terms=["cyberpunk", "cyberpunk style", "neon", "dystopian"],
        tags=["futuristic", "neon", "tech", "urban", "dystopian"],
        compatibility=[LoRACompatibility.SDXL],
    ),
}

# Character LoRAs - focused on specific characters or species
CHARACTER_LORAS: Dict[str, LoRAMetadata] = {
    "foxparks": LoRAMetadata(
        name="Fox Parks",
        filename="noob/foxparks-v1s1600.safetensors",
        category=LoRACategory.CHARACTER,
        subcategory=LoRASubcategory.SPECIFIC_CHARACTER,
        creator="noob",
        version=LoRAVersion("1.0", training_settings={"steps": 1600}),
        description="Creates Fox Parks character, an anthropomorphic fox with specific features",
        strength_range=(0.5, 0.8),
        trigger_terms=["Fox Parks", "anthropomorphic fox"],
        tags=["furry", "anthro", "fox", "character", "animal"],
        nsfw=False,
        compatibility=[LoRACompatibility.SDXL],
    ),
    "renamon": LoRAMetadata(
        name="Renamon",
        filename="renamon-v1.safetensors",
        category=LoRACategory.CHARACTER,
        subcategory=LoRASubcategory.SPECIFIC_CHARACTER,
        creator="DigimonFan",
        version=LoRAVersion("1.0"),
        description="Creates Renamon character from Digimon series",
        strength_range=(0.6, 0.9),
        trigger_terms=["Renamon", "Digimon", "yellow fox digimon"],
        tags=["furry", "digimon", "fox", "character", "yellow", "anime"],
        nsfw=False,
        compatibility=[LoRACompatibility.SDXL],
    ),
    "wolf_anthro": LoRAMetadata(
        name="Anthropomorphic Wolf",
        filename="wolf_anthro-v2.safetensors",
        category=LoRACategory.CHARACTER,
        subcategory=LoRASubcategory.SPECIES,
        creator="FurryArtist",
        version=LoRAVersion("2.0"),
        description="Creates anthropomorphic wolf characters with realistic features",
        strength_range=(0.4, 0.7),
        trigger_terms=["anthropomorphic wolf", "wolf anthro", "wolf furry"],
        tags=["furry", "wolf", "anthro", "canine", "animal"],
        nsfw=False,
        compatibility=[LoRACompatibility.SDXL],
    ),
    "dragon_anthro": LoRAMetadata(
        name="Anthropomorphic Dragon",
        filename="dragon_anthro-v1.safetensors",
        category=LoRACategory.CHARACTER,
        subcategory=LoRASubcategory.SPECIES,
        creator="ScaleFan",
        version=LoRAVersion("1.0"),
        description="Creates anthropomorphic dragon characters with scales and other draconic features",
        strength_range=(0.5, 0.8),
        trigger_terms=["anthropomorphic dragon", "dragon anthro", "dragon furry"],
        tags=["furry", "dragon", "anthro", "scales", "fantasy"],
        nsfw=False,
        compatibility=[LoRACompatibility.SDXL],
    ),
}

# Concept LoRAs - focused on objects, environments, and other concepts
CONCEPT_LORAS: Dict[str, LoRAMetadata] = {
    "cyberpunk_city": LoRAMetadata(
        name="Cyberpunk City",
        filename="cyberpunk_city-v1.safetensors",
        category=LoRACategory.CONCEPT,
        subcategory=LoRASubcategory.ENVIRONMENT,
        creator="FutureScapes",
        version=LoRAVersion("1.0"),
        description="Creates detailed cyberpunk city environments with futuristic architecture",
        strength_range=(0.3, 0.6),
        trigger_terms=["cyberpunk city", "futuristic city", "neon city"],
        tags=["urban", "future", "night", "neon", "tech", "dystopian"],
        compatibility=[LoRACompatibility.SDXL],
    ),
    "fantasy_armor": LoRAMetadata(
        name="Fantasy Armor",
        filename="fantasy_armor-v1.safetensors",
        category=LoRACategory.CONCEPT,
        subcategory=LoRASubcategory.CLOTHING,
        creator="ArmorDesigns",
        version=LoRAVersion("1.0"),
        description="Creates detailed fantasy armor with intricate design elements",
        strength_range=(0.4, 0.7),
        trigger_terms=["fantasy armor", "plate armor", "ornate armor"],
        tags=["medieval", "fantasy", "metal", "protection", "warrior"],
        compatibility=[LoRACompatibility.SDXL],
    ),
    "magical_effects": LoRAMetadata(
        name="Magical Effects",
        filename="magical_effects-v1.safetensors",
        category=LoRACategory.CONCEPT,
        subcategory=LoRASubcategory.ACTION,
        creator="SpellCaster",
        version=LoRAVersion("1.0"),
        description="Creates glowing magical effects like spells, auras, and energy",
        strength_range=(0.4, 0.8),
        trigger_terms=["magic", "magical effects", "spell effects", "energy effects"],
        tags=["glowing", "particles", "energy", "fantasy", "spells"],
        compatibility=[LoRACompatibility.SDXL],
    ),
}

# Kink LoRAs - focused on adult/NSFW content
KINK_LORAS: Dict[str, LoRAMetadata] = {
    "fart_fetish": LoRAMetadata(
        name="Fart Fetish",
        filename="noob/fart_fetish-v2s3000.safetensors",
        category=LoRACategory.KINK,
        subcategory=LoRASubcategory.FETISH,
        creator="noob",
        version=LoRAVersion("2.0", training_settings={"steps": 3000}),
        description="Creates flatulence effects and related content",
        strength_range=(0.5, 0.9),
        trigger_terms=["fart", "farting", "flatulence", "gas"],
        tags=["nsfw", "fetish", "adult"],
        nsfw=True,
        excluded_from_random=True,
        compatibility=[LoRACompatibility.SDXL],
    ),
    "bondage": LoRAMetadata(
        name="Bondage",
        filename="bondage-v1.safetensors",
        category=LoRACategory.KINK,
        subcategory=LoRASubcategory.FETISH,
        creator="Unknown",
        version=LoRAVersion("1.0"),
        description="Creates bondage scenes with ropes, restraints, and other elements",
        strength_range=(0.4, 0.7),
        trigger_terms=["bondage", "restraints", "tied up", "restraint"],
        tags=["nsfw", "fetish", "adult", "bdsm", "rope"],
        nsfw=True,
        excluded_from_random=True,
        compatibility=[LoRACompatibility.SDXL],
    ),
}

# Pose LoRAs - focused on specific poses or compositions
POSE_LORAS: Dict[str, LoRAMetadata] = {
    "action_pose": LoRAMetadata(
        name="Action Pose",
        filename="action_pose-v1.safetensors",
        category=LoRACategory.POSE,
        subcategory=LoRASubcategory.POSITION,
        creator="PoseExpert",
        version=LoRAVersion("1.0"),
        description="Creates dynamic action poses for characters in motion",
        strength_range=(0.4, 0.7),
        trigger_terms=["action pose", "dynamic pose", "movement"],
        tags=["action", "dynamic", "movement", "character", "motion"],
        compatibility=[LoRACompatibility.SDXL],
    ),
    "portrait_composition": LoRAMetadata(
        name="Portrait Composition",
        filename="portrait_composition-v1.safetensors",
        category=LoRACategory.POSE,
        subcategory=LoRASubcategory.COMPOSITION,
        creator="CompositionPro",
        version=LoRAVersion("1.0"),
        description="Creates well-composed portrait shots with pleasing framing",
        strength_range=(0.3, 0.6),
        trigger_terms=["portrait", "headshot", "portrait composition"],
        tags=["portrait", "framing", "composition", "photography", "face"],
        compatibility=[LoRACompatibility.SDXL],
    ),
}

# Combine all LoRAs in a master dictionary
ALL_LORAS: Dict[str, LoRAMetadata] = {
    **STYLE_LORAS,
    **CHARACTER_LORAS,
    **CONCEPT_LORAS,
    **KINK_LORAS,
    **POSE_LORAS,
}

# =============================================================================
# LoRA Selection Configuration
# =============================================================================

# Configuration for optimal LoRA combinations based on types
LORA_COMBINATION_WEIGHTS: Dict[Tuple[LoRACategory, LoRACategory], float] = {
    (LoRACategory.STYLE, LoRACategory.STYLE): 0.2,  # Two styles often conflict
    (LoRACategory.STYLE, LoRACategory.CHARACTER): 1.0,  # Style + character works well
    (LoRACategory.STYLE, LoRACategory.CONCEPT): 0.8,  # Style + concept usually good
    (LoRACategory.STYLE, LoRACategory.POSE): 0.9,  # Style + pose works well
    (LoRACategory.CHARACTER, LoRACategory.CHARACTER): 0.3,  # Two characters can conflict
    (LoRACategory.CHARACTER, LoRACategory.CONCEPT): 1.0,  # Character + concept great
    (LoRACategory.CHARACTER, LoRACategory.POSE): 1.0,  # Character + pose excellent
    (LoRACategory.CONCEPT, LoRACategory.CONCEPT): 0.7,  # Concepts may overlap
    (LoRACategory.CONCEPT, LoRACategory.POSE): 0.9,  # Concept + pose good
    (LoRACategory.POSE, LoRACategory.POSE): 0.4,  # Poses often conflict
}

# Default strength ranges by category
DEFAULT_STRENGTH_RANGES: Dict[LoRACategory, Tuple[float, float]] = {
    LoRACategory.STYLE: (0.3, 0.6),
    LoRACategory.CHARACTER: (0.6, 0.9),
    LoRACategory.CONCEPT: (0.4, 0.7),
    LoRACategory.KINK: (0.5, 0.9),
    LoRACategory.POSE: (0.4, 0.7),
    LoRACategory.MIXED: (0.4, 0.7),
    LoRACategory.UNKNOWN: (0.3, 0.6),
}

# =============================================================================
# Helper Functions
# =============================================================================


def categorize_lora_by_filename(filename: str) -> LoRACategory:
    """
    Determine the likely category of a LoRA based on its filename.

    Args:
        filename: The filename to analyze

    Returns:
        The most likely LoRA category
    """
    # First check against known LoRAs
    for lora in ALL_LORAS.values():
        if lora.filename == filename or lora.filename.endswith(filename):
            return lora.category

    # Character patterns
    character_patterns = [
        r"^character_",
        r"^char_",
        r"^chara_",
        r".*_character$",
        r".*_chara$",
        r".*_char$",
        r"^persona_",
        r".*_persona$",
        r"^avatar_",
        r".*_avatar$",
        r"^hero_",
        r".*_hero$",
        r"^villain_",
        r".*_villain$",
        r"^anthro_",
        r".*_anthro$",
        r"^furry_",
        r".*_furry$",
    ]

    for pattern in character_patterns:
        if re.match(pattern, filename, re.IGNORECASE):
            return LoRACategory.CHARACTER

    # Style patterns
    style_patterns = [
        r".*_style$",
        r".*_art$",
        r".*style$",
        r".*art$",
        r".*-style$",
        r".*-art$",
        r".*artstyle$",
        r".*_artstyle$",
        r".*-artstyle$",
        r".*painting$",
        r".*_painting$",
        r".*-painting$",
        r".*render$",
        r".*_render$",
        r".*-render$",
        r".*aesthetic$",
    ]

    for pattern in style_patterns:
        if re.match(pattern, filename, re.IGNORECASE):
            return LoRACategory.STYLE

    # Concept patterns
    concept_patterns = [
        r"^concept_",
        r".*_concept$",
        r"^theme_",
        r".*_theme$",
        r"^idea_",
        r".*_idea$",
        r"^design_",
        r".*_design$",
        r"^item_",
        r".*_item$",
        r"^object_",
        r".*_object$",
        r"^prop_",
        r".*_prop$",
        r"^weapon_",
        r".*_weapon$",
        r"^clothing_",
        r".*_clothing$",
    ]

    for pattern in concept_patterns:
        if re.match(pattern, filename, re.IGNORECASE):
            return LoRACategory.CONCEPT

    # Kink patterns
    kink_terms = [
        "kink",
        "fetish",
        "bdsm",
        "bondage",
        "fart",
        "latex",
        "leather",
        "rubber",
        "inflation",
        "vore",
        "macro",
        "micro",
        "giantess",
        "giant",
        "diaper",
        "watersports",
        "tickling",
    ]

    for term in kink_terms:
        if term.lower() in filename.lower():
            return LoRACategory.KINK

    # Pose patterns
    pose_terms = ["pose", "position", "composition", "expression", "framing"]

    for term in pose_terms:
        if term.lower() in filename.lower():
            return LoRACategory.POSE

    return LoRACategory.UNKNOWN


def get_loras_by_category(category: LoRACategory) -> Dict[str, LoRAMetadata]:
    """
    Get all LoRAs of a specific category.

    Args:
        category: The category to filter by

    Returns:
        A dictionary of LoRAs in that category
    """
    return {name: lora for name, lora in ALL_LORAS.items() if lora.category == category}


def get_loras_by_subcategory(subcategory: LoRASubcategory) -> Dict[str, LoRAMetadata]:
    """
    Get all LoRAs of a specific subcategory.

    Args:
        subcategory: The subcategory to filter by

    Returns:
        A dictionary of LoRAs in that subcategory
    """
    return {name: lora for name, lora in ALL_LORAS.items() if lora.subcategory == subcategory}


def get_loras_by_tags(tags: List[str], match_all: bool = False) -> Dict[str, LoRAMetadata]:
    """
    Get LoRAs that match specific tags.

    Args:
        tags: The tags to search for
        match_all: If True, all tags must match; if False, any tag match is sufficient

    Returns:
        A dictionary of matching LoRAs
    """
    result = {}

    for name, lora in ALL_LORAS.items():
        if match_all:
            if all(tag.lower() in [t.lower() for t in lora.tags] for tag in tags):
                result[name] = lora
        else:
            if any(tag.lower() in [t.lower() for t in lora.tags] for tag in tags):
                result[name] = lora

    return result


def get_compatible_loras(
    base_category: LoRACategory,
    existing_loras: List[str] = None,
    count: int = 1,
    exclude_nsfw: bool = True,
) -> List[str]:
    """
    Get LoRAs that would work well with the given category and existing LoRAs.

    Args:
        base_category: The primary category to find compatible LoRAs for
        existing_loras: Names of LoRAs already selected (to avoid conflicts)
        count: How many compatible LoRAs to return
        exclude_nsfw: Whether to exclude NSFW LoRAs

    Returns:
        A list of compatible LoRA names
    """
    existing_loras = existing_loras or []

    # Get existing LoRA categories
    existing_categories = []
    for lora_name in existing_loras:
        if lora_name in ALL_LORAS:
            existing_categories.append(ALL_LORAS[lora_name].category)

    # Calculate compatibility scores for all LoRAs
    scores = {}
    for name, lora in ALL_LORAS.items():
        # Skip if already selected
        if name in existing_loras:
            continue

        # Skip if NSFW and we're excluding those
        if exclude_nsfw and lora.nsfw:
            continue

        # Skip if excluded from random selection
        if lora.excluded_from_random:
            continue

        # Base score is 1.0
        score = 1.0

        # Apply category compatibility weights
        for existing_category in existing_categories:
            key = (existing_category, lora.category)
            # Try both orders for the key
            if key in LORA_COMBINATION_WEIGHTS:
                score *= LORA_COMBINATION_WEIGHTS[key]
            elif (lora.category, existing_category) in LORA_COMBINATION_WEIGHTS:
                score *= LORA_COMBINATION_WEIGHTS[(lora.category, existing_category)]

        # Boost score if this is the requested base category
        if lora.category == base_category:
            score *= 1.5

        scores[name] = score

    # Sort by score and return the top results
    sorted_loras = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_loras[:count]]


def suggest_lora_strength(lora_name: str) -> float:
    """
    Suggest an appropriate strength value for a LoRA.

    Args:
        lora_name: The name of the LoRA

    Returns:
        A suggested strength value
    """
    if lora_name in ALL_LORAS:
        lora = ALL_LORAS[lora_name]
        return lora.optimal_strength()

    # If not found, use default by guessing category from name
    category = categorize_lora_by_filename(lora_name)
    range_min, range_max = DEFAULT_STRENGTH_RANGES.get(category, (0.3, 0.7))
    return (range_min + range_max) / 2


def get_available_loras() -> List[str]:
    """
    Get a list of all available LoRA names.

    Returns:
        List of LoRA names
    """
    return list(ALL_LORAS.keys())
