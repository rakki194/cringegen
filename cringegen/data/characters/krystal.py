"""
Krystal Character Template

This file contains the character template for Krystal from the Star Fox series.
"""

from ..character_taxonomy import (
    CharacterTemplate,
    CharacterType,
    FormType,
    Gender
)

# Character metadata for documentation and reference
CHARACTER_INFO = {
    "name": "Krystal",
    "source": "Star Fox",
    "description": "A blue telepathic vixen who joins the Star Fox team.",
    "species": "Fox",
    "tags": ["nintendo", "star fox", "video game", "telepath"],
    "canon_design": "https://starfox.fandom.com/wiki/Krystal",
}

# The actual character template that will be used by the system
CHARACTER_TEMPLATE = CharacterTemplate(
    character_type=CharacterType.ANTHRO,
    form_type=FormType.ANTHRO,
    gender=Gender.FEMALE,
    species="fox",
    archetype=None,
    features=["slim", "athletic", "graceful", "telepathic"],
    clothing=["blue body armor", "gold jewelry", "tribal outfit", "blue boots"],
    accessories=["staff", "diadem", "bracers", "ear piercings"],
    personality=["kind", "adventurous", "loyal", "sensitive", "intelligent"],
    model_tags={
        "e621": "krystal_(star_fox)",
        "danbooru": "krystal_(star_fox)",
        "gelbooru": "krystal_(star_fox)",
        "rule34": "krystal_(star_fox)",
    },
    appearance_traits=[
        "blue fur", 
        "white muzzle",
        "teal eyes", 
        "pointy ears",
        "white tipped tail",
        "athletic build",
        "tribal markings",
    ],
    nsfw_traits=[
        "curvy",
        "athletic",
        "toned",
        "large breasts",
    ],
    is_canon_character=True
)

# Optional pose variations that can be used with the character
CHARACTER_POSES = {
    "action": ["wielding staff", "fighting stance", "firing blaster", "acrobatic"],
    "casual": ["standing", "meditating", "contemplative", "glowing eyes"],
    "pilot": ["in arwing", "flight suit", "pilot gear", "space background"],
}

# Optional clothing variations that can be used
CHARACTER_OUTFITS = {
    "default": ["blue body armor", "gold jewelry", "tribal outfit", "blue boots"],
    "assault": ["blue flight suit", "white vest", "combat boots"],
    "adventures": ["tribal outfit", "gold jewelry", "minimal clothing", "jeweled staff"],
} 