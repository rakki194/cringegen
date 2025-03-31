"""
Character Template Example

Copy this file and rename it to your character's name (all lowercase, use underscores for spaces).
For example, for "Nick Wilde", name the file "nick_wilde.py".

NOTE: Escape all parentheses with \\ (backslash) when using them in the model_tags.
NOTE: Replace all underscores with spaces in the model_tags!

This is a template file and is not loaded as an actual character.
"""

from ..character_taxonomy import (
    CharacterTemplate,
    CharacterType,
    FormType,
    Gender
)

# Character metadata for documentation and reference
CHARACTER_INFO = {
    "name": "Character Full Name",           # The character's full name
    "source": "Source Media",                # Game, movie, TV show, etc.
    "description": "Brief description",      # 1-2 sentence description
    "species": "Species",                    # Primary species
    "tags": ["tag1", "tag2"],                # General tags for categorization
    "canon_design": "URL to reference",      # Optional reference URL
}

# The actual character template that will be used by the system
CHARACTER_TEMPLATE = CharacterTemplate(
    character_type=CharacterType.ANTHRO,     # ANTHRO, HUMAN, KEMONOMIMI, etc.
    form_type=FormType.ANTHRO,               # ANTHRO, FERAL, SEMI_ANTHRO, etc.
    gender=Gender.MALE,                      # MALE, FEMALE, NONBINARY, etc.
    species="species",                       # Species name (lowercase)
    archetype=None,                          # Optional character archetype
    features=[                               # List of physical features
        "feature1", 
        "feature2", 
        "feature3"
    ],
    clothing=[                               # List of clothing items
        "clothing1", 
        "clothing2", 
        "clothing3"
    ],
    accessories=[                            # List of accessories
        "accessory1", 
        "accessory2"
    ],
    personality=[                            # List of personality traits
        "trait1", 
        "trait2", 
        "trait3"
    ],
    model_tags={                             # Tags for specific image platforms
        "e621": "character_name",
        "danbooru": "character_name",
        "gelbooru": "character_name",
        "rule34": "character_name",
    },
    appearance_traits=[                      # Specific appearance details
        "trait1", 
        "trait2", 
        "trait3",
        "trait4",
    ],
    nsfw_traits=[                            # NSFW-specific traits (optional)
        "trait1", 
        "trait2",
    ],
    is_canon_character=True                  # Whether this is a known canon character
)

# Optional pose variations that can be used with the character
CHARACTER_POSES = {
    "pose_category1": ["pose1", "pose2", "pose3"],
    "pose_category2": ["pose4", "pose5", "pose6"],
}

# Optional clothing variations that can be used
CHARACTER_OUTFITS = {
    "default": ["clothing1", "clothing2", "clothing3"],
    "outfit1": ["clothing4", "clothing5", "clothing6"],
    "outfit2": ["clothing7", "clothing8", "clothing9"],
}

# Recommended prompt additions when using this character
RECOMMENDED_ADDITIONS = [
    "addition1",
    "addition2",
    "addition3",
    "addition4",
]

# Recommended LoRAs to use with this character
RECOMMENDED_LORAS = {
    "lora1": 0.7,
    "lora2": 0.6,
    "lora3": 0.5,
}

# Don't change this section
__all__ = [
    'CHARACTER_TEMPLATE',
    'CHARACTER_INFO',
    'CHARACTER_POSES',
    'CHARACTER_OUTFITS',
    'RECOMMENDED_ADDITIONS',
    'RECOMMENDED_LORAS',
] 