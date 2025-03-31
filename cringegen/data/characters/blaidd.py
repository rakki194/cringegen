"""
Blaidd Character Template

This file contains the character template for Blaidd the Half-Wolf from Elden Ring.
"""

from ..character_taxonomy import CharacterTemplate, CharacterType, FormType, Gender

# Character metadata for documentation and reference
CHARACTER_INFO = {
    "name": "Blaidd the Half-Wolf",
    "source": "Elden Ring",
    "description": "A loyal half-wolf warrior who serves Ranni the Witch.",
    "species": "Wolf",
    "tags": ["elden ring", "fromsoft", "soulsborne", "game character"],
    "canon_design": "https://eldenring.wiki.fextralife.com/Blaidd",
}

# The actual character template that will be used by the system
CHARACTER_TEMPLATE = CharacterTemplate(
    character_type=CharacterType.ANTHRO,
    form_type=FormType.ANTHRO,
    gender=Gender.MALE,
    species="wolf",
    archetype=None,
    features=["tall", "muscular", "imposing"],
    clothing=["blue cape", "armor", "greaves"],
    accessories=["large sword", "gauntlets"],
    personality=["loyal", "determined", "honorable"],
    model_tags={
        "e621": "blaidd \\(elden ring\\)",
        "danbooru": "blaidd the half-wolf",  # TODO: check if this is correct
        "gelbooru": "blaidd \\(elden ring\\)",  # TODO: check if this is correct
        "rule34": "blaidd the half-wolf",  # TODO: check if this is correct
    },
    appearance_traits=[
        "black fur",
        "blue cape",
        "chest scar",
        "facial scar",
        "fangs",
        "mane",
        "arm scar",
        "yellow eyes",
    ],
    nsfw_traits=[
        "muscular",
        "well-endowed",
        "large penis",
    ],
    is_canon_character=True,
)

# Optional pose variations that can be used with the character
CHARACTER_POSES = {
    "sword": ["holding sword", "battle stance", "sword drawn"],
    "casual": ["standing", "arms crossed", "leaning against wall"],
    "combat": ["mid-swing", "lunging forward", "defensive stance"],
}

# Optional clothing variations that can be used
CHARACTER_OUTFITS = {
    "default": ["blue cape", "armor", "greaves", "gauntlets"],
    "casual": ["loose shirt", "pants", "boots"],
    "shirtless": ["blue cape", "greaves", "gauntlets"],
}

# Recommended prompt additions when using this character
RECOMMENDED_ADDITIONS = [
    "detailed fur",
    "full body",
    "action shot",
    "dark fantasy",
    "elden ring style",
]

# Recommended LoRAs to use with this character
RECOMMENDED_LORAS = {
    "elden_ring_style": 0.7,
    "fur_detail": 0.6,
    "fantasy_armor": 0.5,
}

__all__ = [
    "CHARACTER_TEMPLATE",
    "CHARACTER_INFO",
    "CHARACTER_POSES",
    "CHARACTER_OUTFITS",
    "RECOMMENDED_ADDITIONS",
    "RECOMMENDED_LORAS",
]
