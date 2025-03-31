"""
Unified LoRA Classification and Management System for CringeGen

This module provides a centralized system for all LoRA-related data, including
classification systems, identification patterns, and filtering mechanisms.
It consolidates LoRA information previously distributed across multiple files.

Key components:
- LoRA type classification keywords (style, character, concept, kink)
- Detection patterns for different LoRA types
- Artist detection patterns
- Exclusion lists for certain LoRAs
- Configuration values for LoRA strength and selection
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Union

# =========================================================================
# LoRA Type Classification
# =========================================================================

# Style LoRA keywords
STYLE_KEYWORDS: Set[str] = {
    "style",
    "artist",
    "aesthetic",
    "artistic",
    "art",
    "render",
    "rendering",
    "technique",
    "textured",
    "realistic",
    "stylized",
    "cartoon",
    "anime",
    "manga",
    "painterly",
    "illustration",
    "illustrative",
    "painted",
    "digital",
    "traditional",
    "painting",
    "drawing",
    "sketch",
    "watercolor",
    "oil",
    "acrylic",
    "charcoal",
    "pencil",
    "chalk",
    "pastel",
    "crayon",
    "comic",
    "graphite",
    "pen",
    "ink",
    "marker",
    "surreal",
    "impressionist",
    "expressionist",
    "cubist",
    "abstract",
    "minimalist",
    "photorealistic",
    "hyperrealistic",
    "cell-shaded",
    "pixel",
    "voxel",
    "low-poly",
    "isometric",
    "flat",
    "3d",
    "cg",
    "render",
    "procedural",
    "generative",
    "fantasy",
    "sci-fi",
    "horror",
    "gothic",
    "cyberpunk",
    "steampunk",
    "dieselpunk",
    "noir",
    "retro",
    "vintage",
    "modern",
    "futuristic",
    "historical",
}

# Character LoRA keywords
CHARACTER_KEYWORDS: Set[str] = {
    "character",
    "person",
    "personage",
    "figure",
    "individual",
    "subject",
    "protagonist",
    "actress",
    "actor",
    "male",
    "female",
    "anthropomorphic",
    "anthro",
    "furry",
    "fursona",
    "persona",
    "avatar",
    "hero",
    "heroine",
    "villain",
    "antagonist",
    "wizard",
    "witch",
    "warrior",
    "knight",
    "archer",
    "mage",
    "sorcerer",
    "hunter",
    "fighter",
    "paladin",
    "ranger",
    "druid",
    "bard",
    "cleric",
    "healer",
    "thief",
    "assassin",
    "rogue",
    "spy",
    "agent",
    "detective",
    "soldier",
    "guard",
    "officer",
    "king",
    "queen",
    "prince",
    "princess",
    "lord",
    "lady",
    "duke",
    "duchess",
    "emperor",
    "empress",
    "noble",
    "peasant",
    "servant",
    "slave",
    "prisoner",
    "dragon",
    "fox",
    "wolf",
    "cat",
    "dog",
    "bear",
    "deer",
    "horse",
    "lion",
    "tiger",
    "otter",
    "rabbit",
    "bunny",
    "squirrel",
    "bird",
    "avian",
    "reptile",
    "lizard",
    "snake",
    "species",
    "creature",
    "beast",
    "monster",
    "demon",
    "angel",
    "deity",
    "god",
    "goddess",
}

# Kink LoRA keywords
KINK_KEYWORDS: Set[str] = {
    "kink",
    "fetish",
    "bdsm",
    "bondage",
    "fart",
    "fart_fetish",
    "foot_fetish",
    "paw_fetish",
    "spanking",
    "latex",
    "leather",
    "rubber",
    "inflation",
    "vore",
    "macro",
    "micro",
    "giantess",
    "giant",
    "tiny",
    "watersports",
    "diaper",
    "diapers",
    "scat",
    "piss",
    "tickling",
    "breathplay",
    "petplay",
    "puppy",
    "kitten",
    "rope",
    "hogtie",
    "hogtied",
    "suspension",
    "restraint",
    "restrained",
    "collar",
    "leash",
    "master",
    "mistress",
    "dominant",
    "domme",
    "submissive",
    "switch",
    "slave",
    "pet",
    "humiliation",
    "degradation",
    "chastity",
    "orgasm denial",
    "edging",
    "impact play",
    "flogger",
    "crop",
    "whip",
    "paddle",
    "cane",
    "pain",
    "pleasure",
    "masochist",
    "sadist",
    "feet",
    "foot",
    "paws",
    "paw",
    "musk",
    "scent",
    "smell",
    "stench",
    "odor",
    "olfactory",
    "pregnancy",
    "pregnant",
    "birth",
    "lactation",
    "milking",
    "tentacle",
    "tentacles",
    "oviposition",
    "eggs",
    "egg-laying",
    "stuffing",
    "stuffed",
    "expansion",
    "expanded",
    "distension",
    "weight gain",
    "fat",
    "obese",
    "chubby",
    "bbw",
    "hyper",
    "transformation",
    "tf",
    "mind control",
    "brainwash",
    "hypno",
    "hypnosis",
    "petplay",
    "prey",
    "predator",
}

# Concept LoRA keywords
CONCEPT_KEYWORDS: Set[str] = {
    "concept",
    "idea",
    "theme",
    "motif",
    "element",
    "design",
    "architecture",
    "vehicle",
    "location",
    "place",
    "environment",
    "scene",
    "background",
    "prop",
    "item",
    "object",
    "weapon",
    "artifact",
    "clothing",
    "outfit",
    "costume",
    "attire",
    "accessory",
    "jewelry",
    "technology",
    "machine",
    "device",
    "food",
    "beverage",
    "animal",
    "plant",
    "flora",
    "fauna",
    "landscape",
    "weather",
    "season",
    "time",
    "event",
    "celebration",
    "ritual",
    "ceremony",
    "festival",
    "holiday",
    "occupation",
    "profession",
    "job",
    "role",
    "activity",
    "action",
    "pose",
    "gesture",
    "expression",
    "emotion",
}

# =========================================================================
# Pattern Matching
# =========================================================================

# Patterns for identifying style LoRAs
STYLE_PATTERNS: List[str] = [
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
    r".*_aesthetic$",
    r".*-aesthetic$",
    r".*artist$",
    r".*_artist$",
    r".*-artist$",
]

# Patterns for identifying character LoRAs
CHARACTER_PATTERNS: List[str] = [
    r"^character_.*",
    r"^char_.*",
    r"^chara_.*",
    r".*_character$",
    r".*_chara$",
    r".*_char$",
    r"^persona_.*",
    r".*_persona$",
    r"^avatar_.*",
    r".*_avatar$",
    r"^hero_.*",
    r".*_hero$",
    r"^villain_.*",
    r".*_villain$",
    r"^anthro_.*",
    r".*_anthro$",
    r"^furry_.*",
    r".*_furry$",
    r"^fursona_.*",
    r".*_fursona$",
]

# Patterns for identifying concept LoRAs
CONCEPT_PATTERNS: List[str] = [
    r"^concept_.*",
    r".*_concept$",
    r"^theme_.*",
    r".*_theme$",
    r"^idea_.*",
    r".*_idea$",
    r"^design_.*",
    r".*_design$",
    r"^item_.*",
    r".*_item$",
    r"^object_.*",
    r".*_object$",
    r"^prop_.*",
    r".*_prop$",
    r"^weapon_.*",
    r".*_weapon$",
    r"^clothing_.*",
    r".*_clothing$",
    r"^outfit_.*",
    r".*_outfit$",
    r"^costume_.*",
    r".*_costume$",
    r"^scene_.*",
    r".*_scene$",
    r"^environment_.*",
    r".*_environment$",
    r"^background_.*",
    r".*_background$",
]

# Artist detection patterns for extracting artist names from LoRA filenames
ARTIST_PATTERNS: List[str] = [
    # noob-artist_name pattern
    r"noob-([a-zA-Z0-9_]+)",
    # artist_name-v pattern
    r"([a-zA-Z0-9_]+)-v\d+",
    # artist_name_style pattern
    r"([a-zA-Z0-9_]+)_style",
    # artist_name_art pattern
    r"([a-zA-Z0-9_]+)_art",
]

# =========================================================================
# LoRA Categorization
# =========================================================================

# Specific LoRAs to categorize as kink-focused regardless of name
SPECIFIC_KINK_LORAS: Set[str] = {
    "fart_fetish-v1s3000",
    "fart_fetish-v2s3000",
    "paw_fetish-v1",
    "foot_fetish-v1",
    "bondage-v1",
    "bdsm-v1",
    "latex-v1",
    "rubber-v1",
    "inflation-v1",
    "vore-v1",
    "macro-v1",
    "micro-v1",
    "giantess-v1",
    "giant-v1",
    "tiny-v1",
    "watersports-v1",
    "diaper-v1",
    "diapers-v1",
    "scat-v1",
    "piss-v1",
    "tickling-v1",
    "rope-v1",
    "petplay-v1",
}

# Specific LoRAs to categorize as character-focused regardless of name
SPECIFIC_CHARACTER_LORAS: Set[str] = {
    "foxparks-v1s1600",
    "retsuko-v1",
    "legoshi-v1",
    "haru-v1",
    "louis-v1",
    "juno-v1",
    "nick_wilde-v1",
    "judy_hopps-v1",
    "krystal-v1",
    "renamon-v1",
    "lucario-v1",
    "zoroark-v1",
    "spyro-v1",
    "loona-v1",
    "roxanne_wolf-v1",
    "amaterasu-v1",
    "king-v1",
    "mordecai-v1",
    "stolas-v1",
    "rocket_raccoon-v1",
    "maid_marian-v1",
    "robin_hood-v1",
    "penn-v2s2600",
}

# Specific LoRAs to categorize as concept-focused regardless of name
SPECIFIC_CONCEPT_LORAS: Set[str] = {
    "surrounded_by_penis-v1s2400",  # Penis/NSFW concept LoRA
}

# =========================================================================
# LoRA Filtering and Configuration
# =========================================================================

# LoRAs that should be excluded from random style selection
EXCLUDED_LORAS: Set[str] = {
    "noob/fart_fetish-v2s3000.safetensors",  # Concept/kink LoRA
    "noob/fart_fetish-v1s3000.safetensors",  # Concept/kink LoRA
    "noob/foxparks-v1s1600.safetensors",     # Character LoRA
    "noob/surrounded_by_penis-v1s2400.safetensors",  # Concept LoRA
    "noob/penn-v2s2600.safetensors",         # Character LoRA
}

# Configuration for artist selection history
ARTIST_SELECTION_CONFIG: Dict[str, float] = {
    "MAX_HISTORY": 10,       # How many previous selections to remember
    "SAME_RUN_PENALTY": 0.8, # Reduce weight of artists used in the same run
}

# Configuration for weighted random LoRA strength selection
LORA_STRENGTH_RANGES: List[Tuple[float, float, float]] = [
    (0.30, 0.40, 0.70),  # 70% chance for 0.30-0.40 range
    (0.40, 0.50, 0.20),  # 20% chance for 0.40-0.50 range
    (0.50, 0.60, 0.07),  # 7% chance for 0.50-0.60 range
    (0.60, 0.70, 0.03),  # 3% chance for 0.60-0.70 range
]

# Kink-specific LoRA recommendations
KINK_LORA_RECOMMENDATIONS: Dict[str, Dict[str, Union[str, List[str]]]] = {
    "fart": {
        "lora": "fart_fetish-v2s3000",
        "checkpoint": "noobaiXLVpredv10.safetensors",
        "terms": ["fart", "gas", "flatulence", "fart fetish"]
    }
    # Additional kink recommendations can be added here
}

# =========================================================================
# Legacy LoRA Datasets and Collections
# =========================================================================

# LoRA datasets by collection/source
LORA_DATASETS: Dict[str, List[str]] = {
    "noob": [
        "noob/animestyle-v1.safetensors",
        "noob/animelineartv2-v1.safetensors",
        "noob/anime-v1s3000.safetensors",
        "noob/chunie-v1s2000.safetensors",
        "noob/foxparks-v1s1600.safetensors",
        "noob/kenket-v1s3000.safetensors",
        "noob/miles-df-v1s20000.safetensors",
        "noob/zaush-v1s3000.safetensors",
        "noob/treedom-v1.safetensors",
        "noob/kadath-v1.safetensors",
        "noob/fluff-v1.safetensors",
    ],
    "civitai": [
        "civitai/add_detail.safetensors",
        "civitai/furrystyle.safetensors",
        "civitai/furry_3d_style.safetensors",
        "civitai/furry_realistic.safetensors",
        "civitai/furry_focused.safetensors",
    ],
    "custom": [
        "custom/fluffystyle.safetensors",
        "custom/pawfeatures.safetensors",
        "custom/cute_furry.safetensors",
    ],
}

# LoRA styles categorized by visual aesthetic
LORA_STYLES: Dict[str, List[str]] = {
    "anime": [
        "noob/anime-v1s3000.safetensors",
        "noob/animestyle-v1.safetensors",
        "noob/animelineartv2-v1.safetensors",
    ],
    "painterly": [
        "noob/chunie-v1s2000.safetensors",
        "noob/kenket-v1s3000.safetensors",
        "noob/treedom-v1.safetensors",
        "noob/kadath-v1.safetensors",
    ],
    "sketchy": [
        "noob/miles-df-v1s20000.safetensors",
        "noob/fluff-v1.safetensors",
    ],
    "realistic": [
        "civitai/furry_realistic.safetensors",
        "noob/zaush-v1s3000.safetensors",
    ],
    "3d": [
        "civitai/furry_3d_style.safetensors",
    ],
    "enhancement": [
        "civitai/add_detail.safetensors",
        "civitai/furry_focused.safetensors",
    ],
    "cute": [
        "custom/cute_furry.safetensors",
        "custom/fluffystyle.safetensors",
    ],
}

# LoRA artists mapped to their characteristic style LoRAs
LORA_ARTISTS: Dict[str, List[str]] = {
    "chunie": ["noob/chunie-v1s2000.safetensors"],
    "kenket": ["noob/kenket-v1s3000.safetensors"],
    "miles-df": ["noob/miles-df-v1s20000.safetensors"],
    "zaush": ["noob/zaush-v1s3000.safetensors"],
    "treedom": ["noob/treedom-v1.safetensors"],
    "kadath": ["noob/kadath-v1.safetensors"],
    "fluff": ["noob/fluff-v1.safetensors"],
    "foxparks": ["noob/foxparks-v1s1600.safetensors"],
}

# LoRA character models
LORA_CHARACTERS: Dict[str, Dict[str, Any]] = {
    "foxparks": {
        "path": "noob/foxparks-v1s1600.safetensors",
        "species": "fox",
        "gender": "male",
        "style": "painterly",
        "recommended_strength": 0.7,
        "trigger_words": ["foxparks", "anthro fox", "fox character"],
    },
    "penn": {
        "path": "noob/penn-v2s2600.safetensors",
        "species": "dragon",
        "gender": "male",
        "style": "painterly",
        "recommended_strength": 0.7,
        "trigger_words": ["penn, dragon character", "anthro dragon"],
    },
    "nick_wilde": {
        "path": "custom/nick_wilde-v1.safetensors",
        "species": "fox",
        "gender": "male",
        "style": "disney",
        "recommended_strength": 0.8,
        "trigger_words": ["nick wilde", "zootopia fox", "sly fox"],
    },
    "judy_hopps": {
        "path": "custom/judy_hopps-v1.safetensors",
        "species": "rabbit",
        "gender": "female",
        "style": "disney",
        "recommended_strength": 0.8,
        "trigger_words": ["judy hopps", "zootopia rabbit", "police rabbit"],
    },
}

# All available LoRAs combined from different sources
ALL_LORAS: Dict[str, Dict[str, Any]] = {}

# Populate ALL_LORAS with style LoRAs
for style, loras in LORA_STYLES.items():
    for lora in loras:
        lora_name = lora.split("/")[-1].split(".")[0]
        ALL_LORAS[lora_name] = {
            "path": lora,
            "type": "style",
            "style": style,
            "recommended_strength": 0.6,
        }

# Add character LoRAs
for name, data in LORA_CHARACTERS.items():
    lora_name = data["path"].split("/")[-1].split(".")[0]
    ALL_LORAS[lora_name] = {
        "path": data["path"],
        "type": "character",
        "species": data.get("species", "unknown"),
        "gender": data.get("gender", "unknown"),
        "style": data.get("style", "unknown"),
        "recommended_strength": data.get("recommended_strength", 0.7),
        "trigger_words": data.get("trigger_words", []),
    }

# Add artist LoRAs
for artist, loras in LORA_ARTISTS.items():
    for lora in loras:
        lora_name = lora.split("/")[-1].split(".")[0]
        if lora_name not in ALL_LORAS:
            ALL_LORAS[lora_name] = {
                "path": lora,
                "type": "artist",
                "artist": artist,
                "recommended_strength": 0.6,
            }

def get_available_loras(comfy_url: str = "http://127.0.0.1:8188") -> List[str]:
    """
    Get a list of available LoRAs from the ComfyUI server.
    If the connection fails, a default list is returned.
    
    Args:
        comfy_url: URL of the ComfyUI server
        
    Returns:
        List of available LoRA file paths
    """
    try:
        import json
        import requests
        
        # Query the ComfyUI API for available LoRAs
        response = requests.get(f"{comfy_url}/object_info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "LoraLoader" in data:
                lora_loader = data["LoraLoader"]
                if "input" in lora_loader and "lora_name" in lora_loader["input"]:
                    return lora_loader["input"]["lora_name"]["filebrowser_config"]["root_paths"]
    except Exception as e:
        print(f"Error getting available LoRAs: {e}")
    
    # If the connection fails or the data structure is unexpected,
    # return a default list using our known LoRAs
    default_loras = []
    for dataset in LORA_DATASETS.values():
        default_loras.extend(dataset)
    return default_loras 