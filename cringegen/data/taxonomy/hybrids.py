"""
Hybrid Species Data for cringegen

This module defines hybrid combinations, compatibility, and traits for mixed species
used in the cringegen prompt generation system.
"""

from typing import Dict, List, Set, Tuple

# Valid hybrid combinations with known names and traits
HYBRID_COMBINATIONS: Dict[str, Dict[str, str]] = {
    "wolf_fox": {
        "name": "folf",
        "taxonomy": "canine",
        "description": "Wolf and fox hybrid with traits from both parent species",
    },
    "lion_tiger": {
        "name": "liger",
        "taxonomy": "feline",
        "description": "Hybrid between male lion and female tiger, typically larger than either parent",
    },
    "tiger_lion": {
        "name": "tigon",
        "taxonomy": "feline",
        "description": "Hybrid between male tiger and female lion, typically smaller than a liger",
    },
    "horse_donkey": {
        "name": "mule",
        "taxonomy": "equine",
        "description": "Offspring of a male donkey and female horse, known for strength and endurance",
    },
    "donkey_horse": {
        "name": "hinny",
        "taxonomy": "equine",
        "description": "Offspring of a male horse and female donkey, typically smaller than a mule",
    },
    "wolf_husky": {
        "name": "wolfdog",
        "taxonomy": "canine",
        "description": "Mix of wolf and domestic dog traits, often with wild characteristics",
    },
    "domestic_cat_wildcat": {
        "name": "hybrid cat",
        "taxonomy": "feline",
        "description": "Domestic cat with wild feline ancestry, such as bengal or savannah cats",
    },
    "dragon_wolf": {
        "name": "dragon wolf",
        "taxonomy": "hybrid",
        "description": "Fantasy hybrid with dragon scales, wings or features on a wolf-like form",
    },
    "cat_dog": {
        "name": "cat dog hybrid",
        "taxonomy": "hybrid",
        "description": "Imaginative hybrid combining feline and canine characteristics",
    },
}

# Species groups that can reasonably be hybridized for creative content
COMPATIBLE_HYBRIDS: Dict[str, List[str]] = {
    "canine": ["canine", "wolf", "fox", "dog", "coyote", "husky", "jackal"],
    "feline": ["feline", "tiger", "lion", "leopard", "cheetah", "cat", "lynx", "jaguar"],
    "equine": ["equine", "horse", "pony", "zebra", "donkey", "unicorn"],
    "dragon": ["dragon", "eastern_dragon", "reptile", "lizard", "snake", "kobold"],
    "rodent": ["rodent", "mouse", "rat", "squirrel", "chipmunk", "hamster"],
    "lagomorph": ["lagomorph", "rabbit", "hare", "bunny"],
    "avian": ["avian", "bird", "eagle", "hawk", "owl", "falcon", "crow", "raven"],
    "cervid": ["deer", "elk", "moose", "reindeer"],
    "fantasy": ["dragon", "sergal", "protogen", "wickerbeast", "dutch_angel_dragon", "avali"],
}

# Trait mixing for hybrid species
HYBRID_TRAIT_MIXING: Dict[str, Dict[str, List[str]]] = {
    "appearance": {
        "canine_feline": [
            "pointed ears",
            "mixed fur patterns",
            "facial features between feline and canine",
            "mixed tail shape",
            "retractable claws",
            "varied eye shape",
        ],
        "mammal_reptile": [
            "scales on parts of the body",
            "fur with scaley patches",
            "mixed limb structure",
            "reptilian eyes",
            "mammalian body shape",
            "scaled tail",
        ],
        "mammal_avian": [
            "feathers mixed with fur",
            "wing structures",
            "beak-like snout",
            "feathered tail",
            "avian feet",
            "mammalian body shape",
        ],
    },
    "behavior": {
        "canine_feline": [
            "mixed social tendencies",
            "balanced independence and pack behavior",
            "combined hunting techniques",
            "varied vocalization",
        ],
        "mammal_reptile": [
            "mixed thermoregulation needs",
            "balanced activity cycles",
            "combined territorial behaviors",
            "mixed dietary preferences",
        ],
        "mammal_avian": [
            "nesting behavior",
            "mixed migration instincts",
            "varied mating displays",
            "combined foraging techniques",
        ],
    },
}

# Body part inheritance patterns for hybrids
HYBRID_BODY_INHERITANCE: Dict[str, Dict[str, List[str]]] = {
    "canine_feline": {
        "canine_dominant": ["snout shape", "ear structure", "leg proportions", "sociability"],
        "feline_dominant": ["eye shape", "claw retraction", "agility", "independence"],
        "mixed": ["fur pattern", "tail shape", "hunting style", "vocalization"],
    },
    "mammal_reptile": {
        "mammal_dominant": ["body shape", "limb structure", "facial features", "fur coverage"],
        "reptile_dominant": [
            "skin texture",
            "eye structure",
            "temperature regulation",
            "regenerative abilities",
        ],
        "mixed": ["coloration", "tail structure", "defensive behaviors", "sensory perception"],
    },
    "mammal_avian": {
        "mammal_dominant": [
            "body proportion",
            "limb structure",
            "facial features",
            "mammary development",
        ],
        "avian_dominant": [
            "feather coverage",
            "hollow bones",
            "air sac respiration",
            "color vision",
        ],
        "mixed": ["beak/snout structure", "limb functionality", "vocalization", "nesting behavior"],
    },
}
