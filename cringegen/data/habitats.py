"""
Habitats Data for cringegen

This module provides habitat information for different species in the cringegen toolkit.
It maps species to their natural or commonly associated habitats.

Key components:
- Species to habitat mappings
"""

from typing import Dict, List

# Species to habitats mapping
SPECIES_HABITATS: Dict[str, List[str]] = {
    # Canines
    "wolf": ["forest", "mountains", "tundra", "wilderness"],
    "dog": ["urban", "home", "park", "meadow"],
    "fox": ["forest", "grassland", "urban fringe", "woodland"],
    "coyote": ["desert", "grassland", "mountains", "urban fringe"],
    "husky": ["tundra", "snow", "mountain cabin"],
    "jackal": ["desert", "savannah", "grassland"],
    # Felines
    "cat": ["urban", "home", "garden", "alley"],
    "tiger": ["jungle", "tropical forest", "grassland", "swamp"],
    "lion": ["savannah", "grassland", "rocky outcrop"],
    "leopard": ["forest", "grassland", "mountains", "tree canopy"],
    "cheetah": ["savannah", "grassland", "open plains"],
    "panther": ["forest", "jungle", "swamp"],
    "lynx": ["forest", "mountains", "tundra"],
    "sabertooth": ["prehistoric landscape", "tundra", "cave"],
    # Equines
    "horse": ["meadow", "farm", "stable", "grassland"],
    "pony": ["meadow", "farm", "stable", "grassland"],
    "zebra": ["savannah", "grassland", "open plains"],
    "donkey": ["farm", "desert fringe", "rocky terrain"],
    # Bovines
    "bull": ["farm", "field", "ranch", "arena"],
    "cow": ["farm", "field", "meadow", "barn"],
    "buffalo": ["grassland", "plains", "river valley"],
    "bison": ["grassland", "plains", "prairie"],
    # Rodents
    "mouse": ["urban", "home", "field", "forest floor"],
    "rat": ["urban", "sewer", "alley", "dock"],
    "squirrel": ["forest", "park", "urban", "tree canopy"],
    "chipmunk": ["forest", "woodland", "garden", "park"],
    "red panda": ["forest", "mountains", "bamboo thicket"],
    # Lagomorphs
    "rabbit": ["meadow", "forest edge", "burrow", "garden"],
    "hare": ["grassland", "tundra", "forest edge", "open field"],
    "bunny": ["meadow", "garden", "forest edge", "home"],
    # Reptiles
    "dragon": ["mountains", "cave", "volcano", "ancient ruins", "castle"],
    "lizard": ["desert", "jungle", "rocky terrain", "forest"],
    "snake": ["jungle", "desert", "grassland", "forest"],
    "crocodile": ["river", "swamp", "lake", "mangrove"],
    "alligator": ["swamp", "marsh", "river", "lake"],
    "kobold": ["cave", "mine", "dungeon", "volcanic cavern"],
    # Avians
    "bird": ["forest", "meadow", "urban", "cliff", "coast"],
    "eagle": ["mountains", "cliff", "coast", "forest"],
    "hawk": ["grassland", "forest edge", "mountains", "open sky"],
    "owl": ["forest", "barn", "ruins", "woodland"],
    "falcon": ["cliff", "mountains", "open sky", "urban towers"],
    # Chiroptera
    "bat": ["cave", "forest", "ruins", "urban", "night sky"],
    # Aquatic
    "shark": ["ocean", "reef", "open water", "deep sea"],
    "dolphin": ["ocean", "coastal waters", "open sea", "bay"],
    "orca": ["ocean", "cold waters", "pod", "open sea"],
    # Miscellaneous
    "deer": ["forest", "meadow", "woodland", "mountain"],
    "raccoon": ["forest", "urban", "stream", "park"],
    "otter": ["river", "lake", "coast", "stream"],
    "bear": ["forest", "mountains", "tundra", "cave"],
    "skunk": ["forest", "meadow", "urban fringe", "burrow"],
    "hyena": ["savannah", "desert", "grassland", "rocky terrain"],
    "goat": ["mountains", "farm", "rocky terrain", "meadow"],
    "sheep": ["meadow", "farm", "hills", "mountain pasture"],
    "kangaroo": ["grassland", "outback", "scrubland", "plains"],
    # Fantasy species
    "sergal": ["tundra", "steppe", "northern plains", "military base"],
    "protogen": ["cyberpunk city", "space station", "tech lab", "digital realm"],
    "primagen": ["cyberpunk city", "space colony", "military base", "research facility"],
    "avali": ["alien tundra", "snow planet", "space station", "crystal caves"],
    "dutch angel dragon": ["cloud realm", "fairy glen", "ethereal forest", "dream dimension"],
    "wickerbeast": ["jungle", "tribal village", "ancient ruins", "bamboo forest"],
    "synth": ["cyberpunk city", "future urban", "tech lab", "neon district"],
}
