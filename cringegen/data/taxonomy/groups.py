"""
Taxonomy Groups for cringegen

This module defines the taxonomic groupings and hierarchies used in the cringegen system.
It organizes species into broader taxonomic categories and relationships.
"""

from typing import Dict, List, Set

# Taxonomic groupings and categories
TAXONOMY_GROUPS: Dict[str, str] = {
    "canine": "mammal",
    "feline": "mammal",
    "equine": "mammal",
    "bovine": "mammal",
    "rodent": "mammal",
    "lagomorph": "mammal",
    "deer": "mammal",
    "procyonid": "mammal",
    "mustelid": "mammal",
    "ursid_humanoid": "mammal",
    "mephitid": "mammal",
    "hyena": "mammal",
    "goat": "mammal",
    "sheep": "mammal",
    "macropod": "mammal",
    "marsupial": "mammal",
    "yangochiropteran": "mammal",
    "reptile": "reptile",
    "avian": "bird",
    "shark": "fish",
    "cetacean": "mammal",
    "fish": "fish",
    "cephalopod": "mollusk",
    "amphibian": "amphibian",
    # Fantasy species
    "sergal": "fantasy",
    "protogen": "synthetic",
    "synth_(vader-san)": "synthetic",
    "primagen": "synthetic",
    "wickerbeast": "fantasy",
    "dutch_angel_dragon": "fantasy",
    "avali": "fantasy",
    "kobold": "fantasy",
    "eastern_dragon": "fantasy",
    "synx": "fantasy",
    "manticore": "fantasy",
    "kitsune": "fantasy",
    "drekkubus": "fantasy",
    "latex_creature": "fantasy",
    "goo_creature": "fantasy",
    "sparkledog": "fantasy",
    # Pokemon
    "pokemon": "fantasy",
}

# Taxonomic hierarchy relationships
TAXONOMY_HIERARCHY: Dict[str, Dict[str, List[str]]] = {
    "mammal": {
        "canine": ["wolf", "fox", "dog", "coyote", "jackal", "husky", "dingo", "african_wild_dog"],
        "feline": [
            "cat",
            "tiger",
            "lion",
            "leopard",
            "cheetah",
            "lynx",
            "caracal",
            "serval",
            "jaguar",
            "panther",
        ],
        "equine": ["horse", "pony", "zebra", "donkey", "unicorn", "pegasus"],
        "bovine": ["cow", "bull", "ox", "bison", "buffalo", "yak"],
        "rodent": [
            "mouse",
            "rat",
            "squirrel",
            "chipmunk",
            "chinchilla",
            "hamster",
            "guinea_pig",
            "capybara",
        ],
        "lagomorph": ["rabbit", "hare", "bunny"],
        "mustelid": ["ferret", "otter", "weasel", "mink", "badger", "wolverine", "mongoose"],
        "ursid_humanoid": ["bear", "polar_bear", "grizzly_bear", "panda"],
        "procyonid": ["raccoon", "red panda"],
        "mephitid": ["skunk"],
        "hyena": ["hyena", "spotted_hyena", "aardwolf"],
        "caprid": ["goat", "mountain_goat"],
        "ovid": ["sheep", "ram"],
        "yangochiropteran": ["bat", "flying_fox", "vampire_bat"],
        "cetacean": ["dolphin", "whale", "orca", "porpoise"],
        "macropod": ["kangaroo", "wallaby"],
        "marsupial": ["possum", "opossum"],
    },
    "reptile": {
        "lizard": ["gecko", "iguana", "chameleon", "bearded_dragon", "monitor_lizard"],
        "snake": ["python", "anaconda", "viper"],
        "crocodilian": ["crocodile", "alligator"],
        "chelonian": ["turtle", "tortoise"],
        "dragon": ["dragon", "eastern_dragon"],
    },
    "bird": {
        "raptor": ["eagle", "owl", "falcon", "hawk"],
        "corvid": ["crow", "raven", "jay", "magpie"],
        "psittacine": ["parrot", "cockatoo", "budgie", "macaw"],
        "passerine": ["finch", "canary"],
        "anatidae": ["swan", "goose", "duck"],
        "spheniscidae": ["penguin"],
        "galliformes": ["chicken", "rooster"],
    },
    "fish": {
        "chondrichthyes": ["shark"],
        "osteichthyes": ["salmon", "tuna", "goldfish", "koi", "betta", "angelfish", "clownfish"],
    },
    "mollusk": {
        "cephalopod": ["octopus", "squid", "cuttlefish"],
    },
    "amphibian": {
        "urodela": ["axolotl"],
    },
    "fantasy": {
        "original": ["sergal", "wickerbeast", "dutch_angel_dragon", "avali", "synx"],
        "mythological": ["kitsune", "manticore", "eastern_dragon"],
        "hybrid": ["sparkledog", "drekkubus", "latex_creature", "goo_creature"],
    },
    "synthetic": {
        "mechanical": ["protogen", "synth_(vader-san)", "primagen"],
    },
}

# Higher-level taxonomic categories
TAXONOMIC_RANKS: Dict[str, List[str]] = {
    "kingdom": ["animalia"],
    "phylum": ["chordata", "mollusca", "arthropoda"],
    "class": [
        "mammalia",
        "reptilia",
        "aves",
        "chondrichthyes",
        "osteichthyes",
        "cephalopoda",
        "amphibia",
    ],
    "order": [
        "carnivora",
        "rodentia",
        "perissodactyla",
        "artiodactyla",
        "lagomorpha",
        "chiroptera",
        "squamata",
        "crocodilia",
        "testudines",
        "passeriformes",
        "falconiformes",
    ],
    "family": [
        "canidae",
        "felidae",
        "equidae",
        "bovidae",
        "muridae",
        "leporidae",
        "mustelidae",
        "ursidae",
        "procyonidae",
        "mephitidae",
        "hyaenidae",
    ],
}
