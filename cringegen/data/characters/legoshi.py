"""
Legoshi Character Template

This file contains the character template for Legoshi from Beastars.
"""

from ..character_taxonomy import CharacterTemplate, CharacterType, FormType, Gender

# Character metadata for documentation and reference
CHARACTER_INFO = {
    "name": "Legoshi",
    "source": "Beastars",
    "description": "A large but gentle gray wolf who struggles with his predatory instincts.",
    "species": "Wolf",
    "tags": ["beastars", "anime", "manga", "student"],
    "canon_design": "https://beastars.fandom.com/wiki/Legoshi",
}

# The actual character template that will be used by the system
CHARACTER_TEMPLATE = CharacterTemplate(
    character_type=CharacterType.ANTHRO,
    form_type=FormType.ANTHRO,
    gender=Gender.MALE,
    species="wolf",
    archetype=None,
    features=["tall", "lanky", "hunched posture", "shy expression"],
    clothing=["school uniform", "white shirt", "blue pants", "black sneakers"],
    accessories=["school bag", "scars"],
    personality=["shy", "gentle", "introspective", "protective", "conflicted"],
    model_tags={
        "e621": "legoshi \\(beastars\\)",
        "danbooru": "legosi \\(beastars\\)",
        "gelbooru": "legoshi \\(beastars\\)",
        "rule34": "legoshi \\(beastars\\)",
    },
    appearance_traits=[
        "gray fur",
        "cream muzzle",
        "gray face markings",
        "thin build",
        "tall",
        "dark eyes",
        "scar on face",
        "scar on back",
        "large paws",
    ],
    nsfw_traits=[
        "muscular",
        "toned",
        "lanky",
        "large",
    ],
    is_canon_character=True,
    hardcore_tags=[
        "canine genitalia",
        "knot",
        #"sheath", # NOTE: This adds a sword or a hilt to the character on NoobAI!
        "cum",
        "cum drip",
        "ejaculation",
        "cumshot",
        "cum on body",
        "cum inside",
        "penetration",
    ],
)

# Optional pose variations that can be used with the character
CHARACTER_POSES = {
    "school": ["in school uniform", "with school bag", "standing awkwardly", "hunched posture"],
    "action": ["fighting stance", "running", "crouching", "baring fangs"],
    "casual": ["reading", "looking to side", "hands in pockets", "holding rabbit"],
}

# Optional clothing variations that can be used
CHARACTER_OUTFITS = {
    "default": ["school uniform", "white shirt", "blue pants", "black sneakers"],
    "casual": ["loose t-shirt", "pants", "simple clothes"],
    "drama club": ["stage outfit", "costume", "mask"],
}
