"""
Miles "Tails" Prower Character Template

This file contains the character template for Tails from the Sonic game series.
"""

from ..character_taxonomy import CharacterTemplate, CharacterType, FormType, Gender

# Character metadata for documentation and reference
CHARACTER_INFO = {
    "name": 'Miles "Tails" Prower',
    "source": "Sonic the Hedgehog (SEGA)",
    "description": "A young two-tailed fox who is Sonic's best friend and sidekick.",
    "species": "Fox",
    "tags": ["sega", "video game", "sonic", "inventor"],
    "canon_design": "https://sonic.fandom.com/wiki/Miles_%22Tails%22_Prower",
}

# The actual character template that will be used by the system
CHARACTER_TEMPLATE = CharacterTemplate(
    character_type=CharacterType.ANTHRO,
    form_type=FormType.ANTHRO,
    gender=Gender.MALE,
    species="fox",
    archetype=None,
    features=["small", "young", "cute", "two tails"],
    clothing=["white gloves", "red and white sneakers"],
    accessories=["tool belt", "wrench"],
    personality=["intelligent", "loyal", "shy", "inventive"],
    model_tags={
        "e621": 'miles "tails" prower',
        "danbooru": "miles prower",
        "gelbooru": "tails \\(sonic\\)",
        "rule34": 'miles "tails" prower',
    },
    appearance_traits=[
        "yellow fur",
        "white muzzle",
        "white chest fur",
        "blue eyes",
        "two tails",
        "pointy ears",
        "white gloves",
    ],
    nsfw_traits=[
        "young",
        "small",
    ],
    is_canon_character=True,
    hardcore_tags=[
        "canine genitalia",
        "knot",
        #"sheath",  # See Legoshi note if needed
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
    "action": ["flying with tails", "flying", "tails spinning", "running"],
    "casual": ["standing", "smiling", "waving", "holding wrench"],
    "mechanic": ["fixing plane", "working on machine", "using wrench"],
}

# Optional clothing variations that can be used
CHARACTER_OUTFITS = {
    "default": ["white gloves", "red and white sneakers"],
    "adventure": ["goggles", "brown belt", "white gloves", "red and white sneakers"],
    "workshop": ["goggles", "tool belt", "white gloves", "red and white sneakers"],
}
