"""
Centralized prompt components for cringegen.

This module contains data structures and utility functions for prompt components
like poses, clothing, accessories, and backgrounds that can be used both by
individual commands and the main generator.
"""

from typing import Dict, List, Optional, Any

# Dictionary of poses by species taxonomy and form type
POSES = {
    "default": {
        "anthro": {
            "neutral": [
                "standing",
                "sitting",
                "walking",
                "running",
                "jumping",
                "leaning against wall",
                "crouching",
                "stretching",
                "looking back over shoulder",
                "arms crossed",
                "hands on hips",
            ],
            "intimate": [
                "laying down",
                "hugging",
                "nuzzling",
                "cuddling",
                "holding hands",
                "leaning on shoulder",
                "back to back",
            ],
            "action": [
                "fighting stance",
                "mid-punch",
                "casting spell",
                "drawing weapon",
                "aiming",
                "dodging",
                "blocking",
                "action pose",
                "battle ready",
            ],
        },
        "feral": {
            "neutral": [
                "standing",
                "sitting",
                "laying down",
                "walking",
                "running",
                "jumping",
                "pouncing",
                "stalking",
                "alert ears",
                "looking back",
                "stretching",
            ],
            "action": [
                "hunting stance",
                "mid-leap",
                "pouncing",
                "running at full speed",
                "sprinting",
                "bounding",
            ],
        },
    },
    "canine": {
        "anthro": {
            "specific": [
                "tail wagging",
                "ears perked",
                "playful stance",
                "nose twitching",
                "head tilted",
            ],
        },
        "feral": {
            "specific": [
                "play bow",
                "tail wagging",
                "ears perked up",
                "sniffing the ground",
                "howling",
                "panting",
            ],
        },
    },
    "feline": {
        "anthro": {
            "specific": [
                "tail swishing",
                "grooming paw",
                "ears flicking",
                "stretching claws",
                "crouched ready to pounce",
                "balancing on ledge",
            ],
        },
        "feral": {
            "specific": [
                "grooming",
                "tail twitching",
                "kneading",
                "arched back",
                "stalking low to ground",
                "pouncing position",
                "stretched out sunbathing",
            ],
        },
    },
    "avian": {
        "anthro": {
            "specific": [
                "wings spread",
                "wings folded",
                "preening feathers",
                "head cocked",
                "wings partially extended",
                "feathers ruffled",
            ],
        },
        "feral": {
            "specific": [
                "wings spread wide",
                "perched",
                "in flight",
                "landing",
                "taking off",
                "gliding",
                "preening",
                "wings folded",
            ],
        },
    },
    "reptile": {
        "anthro": {
            "specific": [
                "tail curled",
                "sunning on rock",
                "scales gleaming",
                "tongue flicking",
                "showing fangs",
                "slithering motion",
            ],
        },
        "feral": {
            "specific": [
                "basking",
                "slithering",
                "coiled",
                "swimming",
                "tail raised",
                "tongue flicking",
                "scales glistening",
            ],
        },
    },
    "equine": {
        "anthro": {
            "specific": [
                "tail swishing",
                "pawing ground",
                "ears forward",
                "mane flowing",
                "tossing head",
            ],
        },
        "feral": {
            "specific": [
                "galloping",
                "trotting",
                "rearing",
                "grazing",
                "pawing the ground",
                "ears flicking",
                "tail swishing",
            ],
        },
    },
    "rodent": {
        "anthro": {
            "specific": [
                "nibbling",
                "whiskers twitching",
                "nose wiggling",
                "ears perked",
                "cheeks full",
                "tail curled",
            ],
        },
        "feral": {
            "specific": [
                "gnawing",
                "whiskers twitching",
                "gathering food",
                "storing food in cheeks",
                "cleaning whiskers",
                "tail twitching",
            ],
        },
    },
}

# Clothing items suitable for different body types and styles
CLOTHING = {
    "anthro": {
        "casual": [
            "t-shirt",
            "hoodie",
            "jeans",
            "shorts",
            "tank top",
            "crop top",
            "sweater",
            "sweatpants",
            "jacket",
            "vest",
            "leggings",
            "sundress",
            "flannel shirt",
            "beanie hat",
            "baseball cap",
            "sneakers",
            "boots",
        ],
        "formal": [
            "suit",
            "dress shirt",
            "tie",
            "blazer",
            "slacks",
            "tuxedo",
            "evening gown",
            "formal dress",
            "high heels",
            "oxford shoes",
            "dress shoes",
            "bow tie",
        ],
        "fantasy": [
            "robes",
            "cloak",
            "leather armor",
            "chainmail",
            "plate armor",
            "wizard hat",
            "adventurer's outfit",
            "tunic",
            "medieval dress",
            "fantasy armor",
            "ranger's cloak",
            "mage robes",
        ],
        "sporty": [
            "track suit",
            "jersey",
            "sports bra",
            "gym shorts",
            "athletic shoes",
            "sweatbands",
            "running shoes",
            "basketball shorts",
            "sports uniform",
            "athletic wear",
        ],
        "swimwear": [
            "swimming trunks",
            "bikini",
            "one-piece swimsuit",
            "board shorts",
            "swim cap",
            "wetsuit",
            "swim goggles",
        ],
        "professional": [
            "lab coat",
            "scrubs",
            "chef's uniform",
            "police uniform",
            "firefighter gear",
            "military uniform",
            "business casual",
            "office attire",
        ],
    },
    "feral": {
        "accessories": [
            "bandana",
            "neck scarf",
            "flower crown",
            "simple harness",
            "cape",
            "backpack",
            "small hat",
            "collar",
            "bow",
            "simple bracelet",
            "anklet",
            "saddle",
        ],
    }
}

# Accessories suitable for different character types
ACCESSORIES = {
    "common": [
        "glasses",
        "sunglasses",
        "watch",
        "bracelet",
        "necklace",
        "earrings",
        "backpack",
        "bag",
        "scarf",
        "hat",
        "headphones",
        "bell",
    ],
    "fantasy": [
        "staff",
        "wand",
        "amulet",
        "magic pendant",
        "potion vials",
        "spell book",
        "crown",
        "tiara",
        "magic ring",
        "enchanted bracelet",
        "rune-engraved jewelry",
    ],
    "tech": [
        "smart watch",
        "AR glasses",
        "tablet",
        "smartphone",
        "wireless earbuds",
        "holographic projector",
        "techno-goggles",
        "cybernetic enhancements",
        "digital assistant device",
    ],
    "natural": [
        "flower crown",
        "leaf wreath",
        "vine bracelet",
        "feather ornament",
        "seashell necklace",
        "wooden beads",
        "bone charm",
        "crystal pendant",
    ],
    "functional": [
        "utility belt",
        "tool pouch",
        "holster",
        "quiver",
        "bandolier",
        "satchel",
        "medical kit",
        "survival gear",
    ],
}

# Backgrounds by environment type
BACKGROUNDS = {
    "natural": {
        "forest": [
            "dense forest",
            "mystical woods",
            "forest clearing",
            "redwood forest",
            "pine forest",
            "bamboo grove",
            "jungle",
            "rainforest",
            "forest path",
            "forest undergrowth",
            "misty forest",
        ],
        "mountain": [
            "mountain peak",
            "snow-capped mountains",
            "mountain range",
            "rocky mountainside",
            "mountain pass",
            "mountain valley",
            "highlands",
            "foggy mountains",
            "mountain vista",
        ],
        "water": [
            "ocean shore",
            "tropical beach",
            "rocky beach",
            "lake shore",
            "river bank",
            "waterfall",
            "coral reef",
            "ocean depths",
            "underwater cave",
            "hot springs",
            "ice lake",
        ],
        "desert": [
            "sand dunes",
            "desert oasis",
            "rocky desert",
            "salt flats",
            "canyon",
            "mesa",
            "red desert",
            "stone arches",
            "dried riverbed",
        ],
        "grassland": [
            "rolling hills",
            "meadow",
            "prairie",
            "savanna",
            "wildflower field",
            "grassy plain",
            "steppe",
            "wheat field",
            "lavender field",
        ],
    },
    "urban": {
        "city": [
            "cityscape",
            "city street",
            "skyscraper roof",
            "urban alley",
            "downtown",
            "city park",
            "neon city",
            "residential district",
            "city square",
            "city market",
        ],
        "architecture": [
            "ancient ruins",
            "castle",
            "palace",
            "temple",
            "cathedral",
            "modern building",
            "futuristic structure",
            "bridge",
            "tower",
            "lighthouse",
            "observatory",
        ],
        "interior": [
            "cozy living room",
            "elegant bedroom",
            "library",
            "laboratory",
            "workshop",
            "kitchen",
            "fancy restaurant",
            "cafe",
            "theater",
            "art gallery",
            "museum",
        ],
    },
    "fantasy": {
        "magical": [
            "crystal cave",
            "floating islands",
            "magic forest",
            "enchanted castle",
            "wizard's tower",
            "portal realm",
            "astral plane",
            "fairy glade",
            "dragon's lair",
            "arcane sanctuary",
        ],
        "otherworldly": [
            "alien planet",
            "cosmic void",
            "nebula",
            "celestial realm",
            "dreamscape",
            "spirit world",
            "mirror dimension",
            "elemental plane",
            "alternate reality",
        ],
    },
}


def get_pose_by_taxonomy(
    taxonomy: str, 
    form_type: str, 
    category: str = "all"
) -> List[str]:
    """Get poses appropriate for a specific species taxonomy and form type.
    
    Args:
        taxonomy: The species taxonomy (e.g., "canine", "feline")
        form_type: The form type ("anthro" or "feral")
        category: The pose category (or "all" for all categories)
        
    Returns:
        A list of appropriate poses
    """
    # Start with default poses for the form type
    pose_options = []
    
    # Add default poses for the specified form type based on category
    if category == "all" or category == "neutral":
        pose_options.extend(POSES["default"][form_type].get("neutral", []))
    
    if form_type == "anthro" and (category == "all" or category == "intimate"):
        pose_options.extend(POSES["default"][form_type].get("intimate", []))
    
    if category == "all" or category == "action":
        pose_options.extend(POSES["default"][form_type].get("action", []))
    
    # Add species-specific poses if available
    if taxonomy in POSES:
        if form_type in POSES[taxonomy]:
            if "specific" in POSES[taxonomy][form_type]:
                pose_options.extend(POSES[taxonomy][form_type]["specific"])
    
    return pose_options


def get_random_pose(
    taxonomy: str = "default", 
    form_type: str = "anthro", 
    category: str = "all"
) -> str:
    """Get a random pose appropriate for a specific species taxonomy and form.
    
    Args:
        taxonomy: The species taxonomy (e.g., "canine", "feline")
        form_type: The form type ("anthro" or "feral")
        category: The pose category (or "all" for all categories)
        
    Returns:
        A randomly selected appropriate pose
    """
    import random
    
    poses = get_pose_by_taxonomy(taxonomy, form_type, category)
    return random.choice(poses) if poses else "standing"


def get_clothing_by_type(
    form_type: str = "anthro", 
    style: str = "all"
) -> List[str]:
    """Get clothing items appropriate for a specific form type and style.
    
    Args:
        form_type: The form type ("anthro" or "feral")
        style: The clothing style (or "all" for all styles)
        
    Returns:
        A list of appropriate clothing items
    """
    clothing_options = []
    
    # For feral characters, we only use accessories
    if form_type == "feral":
        return CLOTHING["feral"]["accessories"]
    
    # For anthro characters, get the requested style(s)
    if style == "all":
        # Combine all styles
        for clothing_style in CLOTHING["anthro"]:
            clothing_options.extend(CLOTHING["anthro"][clothing_style])
    elif style in CLOTHING["anthro"]:
        clothing_options.extend(CLOTHING["anthro"][style])
    
    return clothing_options


def get_random_clothing(
    form_type: str = "anthro", 
    style: str = "all",
    count: int = 1
) -> List[str]:
    """Get random clothing items appropriate for a specific form type and style.
    
    Args:
        form_type: The form type ("anthro" or "feral")
        style: The clothing style (or "all" for all styles)
        count: How many clothing items to return
        
    Returns:
        A list of randomly selected appropriate clothing items
    """
    import random
    
    clothing_items = get_clothing_by_type(form_type, style)
    
    # Limit count to available items
    count = min(count, len(clothing_items))
    
    return random.sample(clothing_items, count) if clothing_items else []


def get_accessories_by_type(
    accessory_type: str = "all"
) -> List[str]:
    """Get accessories of a specific type.
    
    Args:
        accessory_type: The accessory type (or "all" for all types)
        
    Returns:
        A list of appropriate accessories
    """
    accessory_options = []
    
    if accessory_type == "all":
        # Combine all accessory types
        for acc_type in ACCESSORIES:
            accessory_options.extend(ACCESSORIES[acc_type])
    elif accessory_type in ACCESSORIES:
        accessory_options.extend(ACCESSORIES[accessory_type])
    
    return accessory_options


def get_random_accessories(
    accessory_type: str = "all",
    count: int = 1
) -> List[str]:
    """Get random accessories of a specific type.
    
    Args:
        accessory_type: The accessory type (or "all" for all types)
        count: How many accessories to return
        
    Returns:
        A list of randomly selected accessories
    """
    import random
    
    accessories = get_accessories_by_type(accessory_type)
    
    # Limit count to available items
    count = min(count, len(accessories))
    
    return random.sample(accessories, count) if accessories else []


def get_backgrounds_by_type(
    environment_type: str = "all",
    subtype: str = "all"
) -> List[str]:
    """Get backgrounds of a specific environment type and subtype.
    
    Args:
        environment_type: The environment type (or "all" for all types)
        subtype: The environment subtype (or "all" for all subtypes)
        
    Returns:
        A list of appropriate backgrounds
    """
    background_options = []
    
    if environment_type == "all":
        # Combine all environment types and subtypes
        for env_type in BACKGROUNDS:
            for env_subtype in BACKGROUNDS[env_type]:
                background_options.extend(BACKGROUNDS[env_type][env_subtype])
    elif environment_type in BACKGROUNDS:
        if subtype == "all":
            # Combine all subtypes for the specified environment type
            for env_subtype in BACKGROUNDS[environment_type]:
                background_options.extend(BACKGROUNDS[environment_type][env_subtype])
        elif subtype in BACKGROUNDS[environment_type]:
            background_options.extend(BACKGROUNDS[environment_type][subtype])
    
    return background_options


def get_random_background(
    environment_type: str = "all",
    subtype: str = "all"
) -> str:
    """Get a random background of a specific environment type and subtype.
    
    Args:
        environment_type: The environment type (or "all" for all types)
        subtype: The environment subtype (or "all" for all subtypes)
        
    Returns:
        A randomly selected background
    """
    import random
    
    backgrounds = get_backgrounds_by_type(environment_type, subtype)
    return random.choice(backgrounds) if backgrounds else "forest clearing" 