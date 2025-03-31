"""
Nick Wilde Character Template

This file contains the character template for Nick Wilde from Zootopia.
"""

from ..character_taxonomy import CharacterTemplate, CharacterType, FormType, Gender

# Character metadata for documentation and reference
CHARACTER_INFO = {
    "name": "Nick Wilde",
    "source": "Zootopia",
    "description": "A sly and cynical fox who becomes a police officer in Zootopia.",
    "species": "Fox",
    "tags": ["disney", "zootopia", "animated", "movie character"],
    "canon_design": "https://disney.fandom.com/wiki/Nick_Wilde",
}

# The actual character template that will be used by the system
CHARACTER_TEMPLATE = CharacterTemplate(
    character_type=CharacterType.ANTHRO,
    form_type=FormType.ANTHRO,
    gender=Gender.MALE,
    species="fox",
    archetype=None,
    features=["slender", "sly", "smug expression"],
    clothing=["green hawaiian shirt", "loosened tie", "khaki pants"],
    accessories=["sunglasses"],
    personality=["sarcastic", "cunning", "street-smart", "loyal"],
    model_tags={
        "e621": "nick wilde",
        "danbooru": "nick wilde",
        "gelbooru": "nick wilde \\(zootopia\\)",
        "rule34": "nick wilde",
    },
    appearance_traits=[
        "red fur",
        "green eyes",
        "cream muzzle",
        "slim build",
        "pointed ears",
        "bushy tail",
    ],
    nsfw_traits=[
        "slender",
        "toned",
    ],
    is_canon_character=True,
)

# Optional pose variations that can be used with the character
CHARACTER_POSES = {
    "casual": ["hands in pockets", "leaning against wall", "arms crossed"],
    "smug": ["half-lidded eyes", "smirk", "eyebrow raised"],
    "police": ["police uniform", "saluting", "badge visible"],
}

# Optional clothing variations that can be used
CHARACTER_OUTFITS = {
    "default": ["green hawaiian shirt", "loosened tie", "khaki pants"],
    "police": ["police uniform", "tie", "badge"],
    "casual": ["t-shirt", "jeans"],
}

# Recommended prompt additions when using this character
RECOMMENDED_ADDITIONS = [
    "detailed fur",
    "facial expression",
    "vibrant colors",
    "disney style",
]

# Recommended LoRAs to use with this character
RECOMMENDED_LORAS = {
    "disney_style": 0.7,
    "anthro_fox": 0.6,
    "cartoon_animals": 0.5,
}

__all__ = [
    "CHARACTER_TEMPLATE",
    "CHARACTER_INFO",
    "CHARACTER_POSES",
    "CHARACTER_OUTFITS",
    "RECOMMENDED_ADDITIONS",
    "RECOMMENDED_LORAS",
]
