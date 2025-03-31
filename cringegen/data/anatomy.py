"""
Anatomy Data for CringeGen

This module provides anatomical reference data for various species in the CringeGen toolkit.
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
    "default": ["male genitalia", "penis", "testicles"],
    "canine": ["animal penis", "canine penis", "knot", "sheath"],
    "feline": ["animal penis", "feline penis", "barbed penis", "sheath"],
    "equine": ["animal penis", "equine penis", "flared penis", "sheath"],
    "bovine": ["animal penis", "bovine penis", "sheath"],
    "rodent": ["animal penis", "rodent penis", "sheath"],
    "lagomorph": ["animal penis", "lagomorph penis", "sheath"],
    "reptile": ["animal penis", "reptile penis", "hemipenes", "genital slit"],
    "avian": ["animal penis", "avian penis", "genital slit", "cloaca"],
    "cervid": ["animal penis", "cervid penis", "sheath"],
    "procyonid": ["animal penis", "procyonid penis", "sheath"],
    "mustelid": ["animal penis", "mustelid penis", "sheath"],
    "ursid": ["animal penis", "ursine penis", "sheath"],
    "mephitid": ["animal penis", "mephitid penis", "sheath"],
    "hyaenid": ["animal penis", "hyaenid penis", "sheath"],
    "caprid": ["animal penis", "caprid penis", "sheath"],
    "ovid": ["animal penis", "ovid penis", "sheath"],
    "macropod": ["animal penis", "macropod penis", "sheath"],
    "sergal": ["animal penis", "tapered penis", "sheath"],
    "protogen": ["unusual penis", "glowing penis", "sheath"],
    "primagen": ["unusual penis", "glowing penis", "sheath"],
    "avali": ["animal penis", "avali penis", "cloaca", "sheath"],
    "dutch angel dragon": ["unusual penis", "dutch angel dragon penis", "sheath"],
    "wickerbeast": ["animal penis", "wickerbeast penis", "sheath"],
    "synth": ["unusual penis", "synth penis", "glowing penis"],
    "kobold": ["animal penis", "reptile penis", "hemipenes", "genital slit"],
    "chiroptera": ["animal penis", "bat penis", "sheath"],
    "selachii": ["animal penis", "shark penis", "claspers", "genital slit"],
    "cetacean": ["animal penis", "cetacean penis", "tapering penis", "genital slit"],
}

# Female anatomical terms by taxonomy group
FEMALE_ANATOMY: Dict[str, List[str]] = {
    "default": ["female genitalia", "vagina", "pussy"],
    "canine": ["animal pussy", "canine pussy"],
    "feline": ["animal pussy", "feline pussy"],
    "equine": ["animal pussy", "equine pussy"],
    "bovine": ["animal pussy", "bovine pussy"],
    "rodent": ["animal pussy", "rodent pussy"],
    "lagomorph": ["animal pussy", "lagomorph pussy"],
    "reptile": ["animal pussy", "reptile pussy", "genital slit", "cloaca"],
    "avian": ["animal pussy", "avian pussy", "cloaca"],
    "cervid": ["animal pussy", "cervid pussy"],
    "procyonid": ["animal pussy", "procyonid pussy"],
    "mustelid": ["animal pussy", "mustelid pussy"],
    "ursid": ["animal pussy", "ursine pussy"],
    "mephitid": ["animal pussy", "mephitid pussy"],
    "hyaenid": ["animal pussy", "hyaenid pussy"],
    "caprid": ["animal pussy", "caprid pussy"],
    "ovid": ["animal pussy", "ovid pussy"],
    "macropod": ["animal pussy", "macropod pussy"],
    "sergal": ["animal pussy", "sergal pussy", "prehensile clitoral hood"],
    "protogen": ["unusual pussy", "glowing pussy"],
    "primagen": ["unusual pussy", "glowing pussy"],
    "avali": ["animal pussy", "avali pussy", "cloaca"],
    "dutch angel dragon": ["unusual pussy", "dutch angel dragon pussy"],
    "wickerbeast": ["animal pussy", "wickerbeast pussy"],
    "synth": ["unusual pussy", "synth pussy", "glowing pussy"],
    "kobold": ["animal pussy", "reptile pussy", "cloaca"],
    "chiroptera": ["animal pussy", "bat pussy"],
    "selachii": ["animal pussy", "shark pussy", "genital slit", "cloaca"],
    "cetacean": ["animal pussy", "cetacean pussy", "genital slit"],
}

# Anthropomorphic levels (from feral to human-like)
ANTHRO_LEVELS: List[str] = [
    "feral",           # Animal proportions and posture, animal-like behavior
    "semi-feral",      # Animal proportions but more expressive/intelligent
    "semi-anthro",     # Mix of animal and human proportions, may be digitigrade
    "anthro",          # Human proportions with animal features, typically digitigrade
    "plantigrade",     # Human-like feet and stance
    "kemonomimi",      # Mostly human with minimal animal traits (ears, tail)
]

# Body types for character generation
BODY_TYPES: Dict[str, Dict[str, List[str]]] = {
    "male": {
        "build": ["slim", "athletic", "muscular", "stocky", "chubby", "bulky", "lean", "toned"],
        "height": ["short", "average height", "tall", "very tall"],
        "body_shape": ["lithe", "broad-shouldered", "barrel-chested", "compact"],
    },
    "female": {
        "build": ["petite", "slim", "athletic", "curvy", "voluptuous", "muscular", "toned", "full-figured"],
        "height": ["short", "average height", "tall", "very tall"],
        "body_shape": ["hourglass", "pear-shaped", "athletic", "slender"],
    },
    "androgynous": {
        "build": ["slim", "athletic", "lean", "androgynous", "toned", "willowy"],
        "height": ["short", "average height", "tall"],
        "body_shape": ["slim", "balanced", "neutral", "lean"],
    }
}

# Special features for anthro characters
ANTHRO_FEATURES: Dict[str, List[str]] = {
    "canine": ["pointed ears", "snout", "paw pads", "tail", "claws", "whiskers", "digitigrade legs"],
    "feline": ["triangular ears", "whiskers", "slit pupils", "tail", "paw pads", "retractable claws"],
    "equine": ["long muzzle", "mane", "tail", "hooves", "pointed ears"],
    "bovine": ["horns", "muzzle", "tail", "hooves", "broad shoulders"],
    "lagomorph": ["long ears", "short tail", "whiskers", "pronounced incisors", "paw pads"],
    "rodent": ["rounded ears", "whiskers", "buck teeth", "tail", "paw pads"],
    "reptile": ["scales", "tail", "claws", "slit pupils", "fangs", "forked tongue"],
    "avian": ["beak", "feathered wings", "tail feathers", "crest", "talons", "feather patterns"],
    "common": ["tufted ears", "ear fluff", "neck fluff", "chest fluff", "tail tip", "markings", "heterochromia"],
}

# Animal-specific anatomical terms for prompt generation
ANIMAL_ANATOMICAL_TERMS: Dict[str, List[str]] = {
    "common": ["paws", "claws", "muzzle", "snout", "tail", "fur", "ears"],
    "canine": ["muzzle", "paws", "claws", "canines", "pointed ears", "tail"],
    "feline": ["whiskers", "retractable claws", "triangular ears", "tail"],
    "equine": ["muzzle", "hooves", "mane", "tail", "withers"],
    "avian": ["beak", "wings", "feathers", "talons", "crest"],
    "reptile": ["scales", "claws", "fangs", "forked tongue", "crest"],
    "lagomorph": ["twitching nose", "long ears", "fluffy tail", "powerful hindlegs"],
}

# Human anatomical terms for prompt generation
HUMAN_ANATOMICAL_TERMS: List[str] = [
    "face", "eyes", "nose", "mouth", "lips", "teeth", "ears", "hair", "neck", 
    "shoulders", "arms", "elbows", "wrists", "hands", "fingers", "chest", "torso", 
    "back", "waist", "hips", "buttocks", "legs", "knees", "ankles", "feet", "toes"
]

# Species-specific anatomy traits
SPECIES_ANATOMY_TRAITS: Dict[str, Dict[str, List[str]]] = {
    "wolf": {
        "head": ["pointed ears", "long muzzle", "strong jaw"],
        "body": ["muscular shoulders", "sleek body", "bushy tail"],
        "limbs": ["digitigrade legs", "paw pads", "claws"],
    },
    "fox": {
        "head": ["triangular ears", "narrow muzzle", "alert expression"],
        "body": ["slender frame", "fluffy tail", "bushy brush"],
        "limbs": ["slender legs", "dainty paws", "sharp claws"],
    },
    "dragon": {
        "head": ["horns", "frills", "elongated snout", "sharp teeth"],
        "body": ["scales", "spikes", "wings", "long tail"],
        "limbs": ["clawed feet", "talons", "powerful legs"],
    },
    "cat": {
        "head": ["triangular ears", "short muzzle", "whiskers", "slit pupils"],
        "body": ["flexible spine", "slender frame", "expressive tail"],
        "limbs": ["retractable claws", "sensitive paw pads", "agile legs"],
    },
    "rabbit": {
        "head": ["long ears", "twitching nose", "short muzzle", "buck teeth"],
        "body": ["fluffy tail", "rounded body", "soft fur"],
        "limbs": ["powerful hindlegs", "quick feet", "short forelimbs"],
    },
    "bird": {
        "head": ["beak", "round eyes", "head crest"],
        "body": ["wings", "tail feathers", "light frame"],
        "limbs": ["talons", "scaly legs", "grasping feet"],
    },
    "default": {
        "head": ["expressive face", "distinctive ears", "species-appropriate mouth"],
        "body": ["fur/scale covered", "appropriate proportions", "tail"],
        "limbs": ["four limbs", "paws/hooves/etc", "appropriate mobility features"],
    }
} 