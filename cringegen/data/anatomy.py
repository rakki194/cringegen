"""
Anatomy Data for cringegen

This module provides anatomical reference data for various species in the cringegen toolkit.
It includes male and female anatomy terms organized by taxonomy group.

Key components:
- Male anatomical terms by taxonomy
- Female anatomical terms by taxonomy
- Anthro characteristics and body types
- Anatomical terminology for different species
"""

from typing import Dict, List

# Male anatomical terms by taxonomy group
MALE_ANATOMY: Dict[str, List[str]] = {
    "canine": ["animal penis", "canine penis", "knot", "presenting sheath"],
    "feline": ["animal penis", "feline penis", "barbed penis", "presenting sheath"],
    "equine": ["animal penis", "equine penis", "flared penis", "presenting sheath"],
    "bovine": ["animal penis", "bovine penis", "presenting sheath"],
    "rodent": ["animal penis", "rodent penis", "presenting sheath"],
    "lagomorph": ["animal penis", "lagomorph penis", "presenting sheath"],
    "reptile": ["animal penis", "reptile penis", "hemipenes", "genital slit"],
    "avian": ["animal penis", "avian penis", "genital slit", "cloaca"],
    "deer": ["animal penis", "cervid penis", "presenting sheath"],
    "procyonid": ["animal penis", "procyonid penis", "presenting sheath"],
    "mustelid": ["animal penis", "mustelid penis", "presenting sheath"],
    "ursid_humanoid": ["animal penis", "ursine penis", "presenting sheath"],
    "mephitid": ["animal penis", "mephitid penis", "presenting sheath"],
    "hyaenid": ["animal penis", "hyaenid penis", "presenting sheath"],
    "bovid": ["animal penis", "caprid penis", "presenting sheath"],
    "ovid": ["animal penis", "ovid penis", "presenting sheath"],
    "macropod": ["animal penis", "macropod penis", "presenting sheath"],
    "sergal": ["animal penis", "tapered penis", "presenting sheath"],
    "protogen": ["unusual penis", "glowing penis", "presenting sheath"],
    "avali": ["animal penis", "avali penis", "cloaca", "presenting sheath"],
    "dragon": ["tapering penis", "presenting sheath"],
    "wickerbeast": ["animal penis", "presenting sheath"],
    "synthetic": ["unusual penis", "glowing penis"],
    "kobold": ["animal penis", "reptile penis", "hemipenes", "genital slit"],
    "yangochiropteran": ["animal penis", "bat penis", "presenting sheath"],
    "shark": ["animal penis", "shark penis", "claws", "genital slit"],
    "cetacean": ["animal penis", "cetacean penis", "tapering penis", "genital slit"],
}

# Female anatomical terms by taxonomy group
FEMALE_ANATOMY: Dict[str, List[str]] = {
    "canine": ["animal pussy", "canine pussy"],
    "feline": ["animal pussy", "feline pussy"],
    "equine": ["animal pussy", "equine pussy"],
    "bovine": ["animal pussy", "bovine pussy"],
    "rodent": ["animal pussy", "rodent pussy"],
    "lagomorph": ["animal pussy", "lagomorph pussy"],
    "reptile": ["animal pussy", "reptile pussy", "genital slit", "cloaca"],
    "avian": ["animal pussy", "avian pussy", "cloaca"],
    "deer": ["animal pussy", "cervid pussy"],
    "procyonid": ["animal pussy", "procyonid pussy"],
    "mustelid": ["animal pussy", "mustelid pussy"],
    "ursid_humanoid": ["animal pussy", "ursine pussy"],
    "mephitid": ["animal pussy", "mephitid pussy"],
    "hyaenid": ["animal pussy", "hyaenid pussy"],
    "bovid": ["animal pussy", "caprid pussy"],
    "ovid": ["animal pussy", "ovid pussy"],
    "macropod": ["animal pussy", "macropod pussy"],
    "sergal": ["animal pussy", "sergal pussy", "prehensile clitoral hood"],
    "protogen": ["unusual pussy", "glowing pussy"],
    "synthetic": ["unusual pussy", "synth pussy", "glowing pussy"],
    "avali": ["cloaca"],
    "dragon": ["cloaca"],
    "wickerbeast": ["animal pussy", "wickerbeast pussy"],
    "kobold": ["animal pussy", "reptile pussy", "cloaca"],
    "yangochiropteran": ["animal pussy", "bat pussy"],
    "shark": ["animal pussy", "shark pussy", "genital slit", "cloaca"],
    "cetacean": ["animal pussy", "cetacean pussy", "genital slit"],
}

# Anthropomorphic levels (from feral to human-like)
ANTHRO_LEVELS: List[str] = [
    "feral",  # Animal proportions and posture, animal-like behavior
    "semi-feral",  # Animal proportions but more expressive/intelligent
    "semi-anthro",  # Mix of animal and human proportions, may be digitigrade
    "anthro",  # Human proportions with animal features, typically digitigrade
    "plantigrade",  # Human-like feet and stance
    "kemonomimi",  # Mostly human with minimal animal traits (ears, tail)
]

# Body types for character generation
BODY_TYPES: Dict[str, Dict[str, List[str]]] = {
    "male": {
        "build": ["slim", "athletic", "muscular", "stocky", "chubby", "bulky", "lean", "toned"],
        "height": ["short", "average height", "tall", "very tall"],
        "body_shape": ["lithe", "broad-shouldered", "barrel-chested", "compact"],
    },
    "female": {
        "build": [
            "petite",
            "slim",
            "athletic",
            "curvy",
            "voluptuous",
            "muscular",
            "toned",
            "full-figured",
        ],
        "height": ["short", "average height", "tall", "very tall"],
        "body_shape": ["hourglass", "pear-shaped", "athletic", "slender"],
    },
    "androgynous": {
        "build": ["slim", "athletic", "lean", "androgynous", "toned", "willowy"],
        "height": ["short", "average height", "tall"],
        "body_shape": ["slim", "balanced", "neutral", "lean"],
    },
}

# Special features for anthro characters
ANTHRO_FEATURES: Dict[str, List[str]] = {
    "canine": [
        "humanoid_pointy_ears",
        "snout",
        "paws",
        "tail",
        "claws",
        "whiskers",
        "digitigrade",
    ],
    "feline": [
        "humanoid_pointy_ears",
        "whiskers",
        "slit pupils",
        "tail",
        "paws",
        "claws",
    ],
    "equine": ["long_snout", "mane", "tail", "hooves", "humanoid_pointy_ears"],
    "bovine": ["horn", "snout", "tail", "hooves", "broad shoulders"],
    "lagomorph": ["long ears", "short tail", "whiskers", "teeth", "paws"],
    "rodent": ["humanoid_pointy_ears", "whiskers", "teeth", "tail", "paws"],
    "reptile": ["scales", "tail", "claws", "slit pupils", "fangs", "forked tongue"],
    "avian": ["beak", "feathered wings", "tail feathers", "head_crest", "talons", "feather_tuft"],
    "common_hippopotamus": [
        "tuft",
        "inner_ear_fluff",
        "neck_tuft",
        "chest_tuft",
        "tail",
        "markings",
        "heterochromia",
    ],
}

# Animal-specific anatomical terms for prompt generation
ANIMAL_ANATOMICAL_TERMS: Dict[str, List[str]] = {
    "common_hippopotamus": ["paws", "claws", "snout", "snout", "tail", "fur", "ears"],
    "canine": ["snout", "paws", "claws", "canines", "humanoid_pointy_ears", "tail"],
    "feline": ["whiskers", "claws", "humanoid_pointy_ears", "tail"],
    "equine": ["snout", "hooves", "mane", "tail", "withers"],
    "avian": ["beak", "wings", "feathers", "talons", "head_crest"],
    "reptile": ["scales", "claws", "fangs", "forked tongue", "head_crest"],
    "lagomorph": ["twitching nose", "long ears", "fluffy tail", "powerful hindlegs"],
}

# Human anatomical terms for prompt generation
HUMAN_ANATOMICAL_TERMS: List[str] = [
    "face",
    "eyes",
    "nose",
    "mouth",
    "lips",
    "teeth",
    "ears",
    "hair",
    "neck",
    "shoulders",
    "arms",
    "elbows",
    "wrists",
    "hands",
    "fingers",
    "chest",
    "torso",
    "back",
    "waist",
    "hips",
    "buttocks",
    "legs",
    "knees",
    "ankles",
    "feet",
    "toes",
]

# Species-specific anatomy traits
SPECIES_ANATOMY_TRAITS: Dict[str, Dict[str, List[str]]] = {
    "wolf": {
        "head": ["humanoid_pointy_ears", "long_snout", "strong jaw"],
        "body": ["muscular shoulders", "sleek body", "bushy tail"],
        "limbs": ["digitigrade", "paws", "claws"],
    },
    "fox": {
        "head": ["humanoid_pointy_ears", "narrow muzzle", "alert expression"],
        "body": ["slender frame", "fluffy tail", "bushy brush"],
        "limbs": ["slender legs", "dainty paws", "sharp claws"],
    },
    "dragon": {
        "head": ["horn", "frills", "elongated snout", "sharp teeth"],
        "body": ["scales", "spikes", "wings", "long tail"],
        "limbs": ["clawed feet", "talons", "powerful legs"],
    },
    "cat": {
        "head": ["humanoid_pointy_ears", "short muzzle", "whiskers", "slit pupils"],
        "body": ["flexible spine", "slender frame", "expressive tail"],
        "limbs": ["claws", "sensitive paws", "agile legs"],
    },
    "rabbit": {
        "head": ["long ears", "twitching nose", "short muzzle", "teeth"],
        "body": ["fluffy tail", "rounded body", "soft fur"],
        "limbs": ["powerful hindlegs", "quick feet", "short forelimbs"],
    },
    "bird": {
        "head": ["beak", "round eyes", "head_crest"],
        "body": ["wings", "tail feathers", "light frame"],
        "limbs": ["talons", "scaly legs", "grasping feet"],
    },
}
