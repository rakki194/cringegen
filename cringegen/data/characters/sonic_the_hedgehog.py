"""
Sonic the Hedgehog Character Template

This file contains the character template for Sonic the Hedgehog from the Sonic game series.
"""

from ..character_taxonomy import (
    CharacterTemplate,
    CharacterType,
    FormType,
    Gender
)

# Character metadata for documentation and reference
CHARACTER_INFO = {
    "name": "Sonic the Hedgehog",
    "source": "Sonic the Hedgehog (SEGA)",
    "description": "A blue anthropomorphic hedgehog with superhuman speed.",
    "species": "Hedgehog",
    "tags": ["sega", "video game", "sonic", "mascot"],
    "canon_design": "https://sonic.fandom.com/wiki/Sonic_the_Hedgehog",
}

# The actual character template that will be used by the system
CHARACTER_TEMPLATE = CharacterTemplate(
    character_type=CharacterType.ANTHRO,
    form_type=FormType.ANTHRO,
    gender=Gender.MALE,
    species="hedgehog",
    archetype=None,
    features=["short", "athletic", "confident expression"],
    clothing=["white gloves", "red sneakers", "gold buckle shoes"],
    accessories=["gold ring"],
    personality=["cocky", "heroic", "impatient", "freedom-loving"],
    model_tags={
        "e621": "sonic the hedgehog",
        "danbooru": "sonic the hedgehog",
        "gelbooru": "sonic the hedgehog",
        "rule34": "sonic the hedgehog",
    },
    appearance_traits=[
        "blue fur", 
        "green eyes", 
        "blue spikes",
        "peach muzzle",
        "peach arms",
        "pointy ears",
        "white gloves",
        "red shoes",
    ],
    nsfw_traits=[
        "athletic",
        "toned",
    ],
    is_canon_character=True
)

# Optional pose variations that can be used with the character
CHARACTER_POSES = {
    "action": ["running", "spin dash", "jumping", "thumbs up"],
    "casual": ["standing", "smirking", "winking", "arms crossed"],
    "super": ["super sonic", "glowing", "yellow fur", "floating"],
}

# Optional clothing variations that can be used
CHARACTER_OUTFITS = {
    "default": ["white gloves", "red sneakers", "gold buckle shoes"],
    "winter": ["scarf", "white gloves", "red sneakers", "gold buckle shoes"],
    "boom": ["sports tape", "bandana", "white gloves", "blue sneakers"],
} 