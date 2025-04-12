"""
Species Descriptors for cringegen

This module provides descriptive terms and phrases used to characterize different species
in a taxonomically appropriate way.
"""

from typing import Dict, List, Set

# Generic anthro descriptors that apply to most anthropomorphic characters
ANTHRO_DESCRIPTOR_SET: Set[str] = {
    "anthro",
    "anthropomorphic",
    "furry",
    "kemonomimi",
    "humanoid",
    "anthro_animal",
    "furry_character",
    "anthro_character",
    "zoomorphic",
    "therianthrope",
}

# Species-specific descriptors for more tailored prompt generation
SPECIES_DESCRIPTORS: Dict[str, List[str]] = {
    "wolf": [
        "lupine",
        "canine",
        "wolfish",
        "feral wolf",
        "timber wolf",
        "arctic wolf",
        "werewolf",
        "wolf-like",
    ],
    "fox": [
        "vulpine",
        "canine",
        "foxy",
        "fox-like",
        "firefox",
        "kitsune",
        "fennec",
        "foxxo",
    ],
    "dog": [
        "canine",
        "doggo",
        "pupper",
        "dog-like",
        "domestic canine",
        "canis familiaris",
        "puppy",
        "hound",
    ],
    "cat": [
        "feline",
        "kitty",
        "felis catus",
        "domestic cat",
        "house cat",
        "kitten",
        "tomcat",
        "cat-like",
    ],
    "tiger": [
        "feline",
        "tigress",
        "striped tiger",
        "panthera tigris",
        "big cat",
        "tiger-like",
        "bengal tiger",
    ],
    "lion": [
        "feline",
        "lioness",
        "big cat",
        "king of beasts",
        "panthera leo",
        "lion-like",
        "maned lion",
    ],
    "horse": [
        "equine",
        "steed",
        "mare",
        "stallion",
        "equus ferus",
        "horse-like",
        "colt",
        "filly",
    ],
    "dragon": [
        "draconic",
        "drake",
        "wyvern",
        "wyrm",
        "fire-breathing",
        "winged dragon",
        "dragon-like",
        "dragoness",
    ],
    "rabbit": [
        "lagomorph",
        "bunny",
        "lepus",
        "leporid",
        "rabbit-like",
        "hare",
        "cottontail",
        "buck",
        "doe",
    ],
    "raccoon": [
        "procyonid",
        "trash panda",
        "procyon lotor",
        "bandits",
        "raccoon-like",
        "washing bear",
    ],
    "otter": [
        "mustelid",
        "lutrinae",
        "river otter",
        "sea otter",
        "otter-like",
        "aquatic mustelid",
    ],
    "bear": [
        "ursine",
        "bruin",
        "ursidae",
        "bear-like",
        "grizzly",
        "brown bear",
        "black bear",
        "polar bear",
    ],
    "bird": [
        "avian",
        "feathered",
        "bird-like",
        "winged",
        "flying",
        "beaked",
        "ornithological",
    ],
    "shark": [
        "selachian",
        "elasmobranch",
        "shark-like",
        "cartilaginous fish",
        "finned",
        "aquatic predator",
    ],
    "sergal": [
        "sergal-like",
        "vilous",
        "wedge-headed",
        "cheese-headed",
        "northern sergal",
        "southern sergal",
    ],
    "protogen": [
        "cybernetic",
        "digital",
        "mechanical",
        "visor-faced",
        "robotic",
        "technorganic",
        "proto",
    ],
}

# Personality trait descriptors often associated with certain species
SPECIES_PERSONALITY_TRAITS: Dict[str, List[str]] = {
    "wolf": ["loyal", "pack-oriented", "fierce", "protective", "wild", "cunning"],
    "fox": ["clever", "sly", "curious", "playful", "tricky", "mischievous"],
    "dog": ["friendly", "loyal", "energetic", "devoted", "playful", "affectionate"],
    "cat": ["independent", "aloof", "curious", "graceful", "mysterious", "proud"],
    "rabbit": ["timid", "alert", "fast", "nervous", "gentle", "skittish"],
    "dragon": ["proud", "haughty", "intelligent", "fierce", "protective", "wise"],
    "raccoon": ["curious", "clever", "dexterous", "opportunistic", "resourceful"],
    "otter": ["playful", "energetic", "social", "aquatic", "agile", "mischievous"],
    "sergal": ["predatory", "tribal", "fierce", "social", "adaptable"],
    "protogen": ["logical", "analytical", "curious", "technological", "adaptive"],
}

# Cultural/behavioral associations by species type
SPECIES_CULTURAL_TRAITS: Dict[str, List[str]] = {
    "canine": ["pack mentality", "hierarchical", "territorial", "social grooming"],
    "feline": ["solitary hunters", "independence", "territorial marking", "grooming"],
    "equine": ["herd mentality", "grazing", "displays of strength", "running"],
    "lagomorph": ["burrowing", "alertness", "rapid reproduction", "hopping"],
    "dragon": ["treasure hoarding", "territorial", "flying", "magical affinity"],
    "avian": ["flock behavior", "migration patterns", "nest building", "singing"],
    "sergal": ["tribal societies", "hunting traditions", "ritualistic marking"],
    "protogen": ["technological integration", "data collection", "adaptation"],
}
