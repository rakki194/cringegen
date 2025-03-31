"""
Robin Hood Character Template

This file contains the character template for Robin Hood from Disney's animated film.
"""

from ..character_taxonomy import (
    CharacterTemplate,
    CharacterType,
    FormType,
    Gender
)

# Character metadata for documentation and reference
CHARACTER_INFO = {
    "name": "Robin Hood",
    "source": "Robin Hood (Disney)",
    "description": "A heroic anthropomorphic fox who steals from the rich to give to the poor.",
    "species": "Fox",
    "tags": ["disney", "animated", "movie character", "archer"],
    "canon_design": "https://disney.fandom.com/wiki/Robin_Hood_(character)",
}

# The actual character template that will be used by the system
CHARACTER_TEMPLATE = CharacterTemplate(
    character_type=CharacterType.ANTHRO,
    form_type=FormType.ANTHRO,
    gender=Gender.MALE,
    species="fox",
    archetype=None,
    features=["slim", "agile", "smug smile", "confident posture"],
    clothing=["green tunic", "green hat", "yellow scarf", "brown belt"],
    accessories=["bow and arrow", "quiver", "feather in hat"],
    personality=["heroic", "charming", "witty", "compassionate", "daring"],
    model_tags={
        "e621": "robin_hood_(disney)",
        "danbooru": "robin_hood_(disney)",
        "gelbooru": "robin_hood_(disney)",
        "rule34": "robin_hood_(disney)",
    },
    appearance_traits=[
        "red fur", 
        "cream muzzle",
        "pointy ears",
        "bushy tail",
        "slim build",
        "green eyes",
        "green hat with feather",
    ],
    nsfw_traits=[
        "slim",
        "athletic",
        "toned",
    ],
    is_canon_character=True
)

# Optional pose variations that can be used with the character
CHARACTER_POSES = {
    "action": ["aiming bow", "shooting arrow", "climbing tree", "running"],
    "heroic": ["standing heroically", "hands on hips", "drawing bow", "smirking"],
    "stealthy": ["sneaking", "hiding", "peeking around corner", "crouching"],
}

# Optional clothing variations that can be used
CHARACTER_OUTFITS = {
    "default": ["green tunic", "green hat with feather", "yellow scarf", "brown belt"],
    "tournament": ["peasant disguise", "blindfold", "torn clothes"],
    "noble": ["fancy clothes", "royal outfit", "prince attire"],
} 