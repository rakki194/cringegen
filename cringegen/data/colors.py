"""
Unified Color System for cringegen

This module provides a centralized color system with pattern definitions, species-specific
color patterns, and templates for generating color descriptions. It consolidates color
information previously distributed across multiple files.

Key components:
- Basic colors and palettes
- Color patterns by body covering type
- Species-specific color patterns
- Color distribution templates
- Color description templates
- Material-based colors
- RGB color values
"""

from typing import Dict, List, Tuple

# Basic colors for general use
BASIC_COLORS: List[str] = [
    "red_fur",
    "orange_fur",
    "yellow_fur",
    "green_fur",
    "blue_fur",
    "purple_fur",
    "pink_fur",
    "brown_fur",
    "grey_fur",
    "black_fur",
    "white_fur",
    "tan_fur",
    "ice_cream",
    "gold_(metal)",
    "silver_(metal)",
]

# Expanded color palette with more specific hues
EXTENDED_COLORS: Dict[str, List[str]] = {
    "red_fur": ["crimson", "car", "ruby_(gem)", "gun", "maroon", "lion", "car"],
    "orange_fur": ["amber_eyes", "tangerine", "rusty_trombone", "burnt orange", "gold_(metal)", "peach_(fruit)", "coral"],
    "yellow_fur": ["lemon", "gold_(metal)", "star", "honey_(food)", "saffron", "daffodil", "bra"],
    "green_fur": ["emerald_(gem)", "slime", "mint", "forest", "olive", "jade", "massage", "tea"],
    "blue_fur": ["azure", "cobalt", "navy", "indigo", "sapphire_(gem)", "cerulean", "turquoise_hair", "cyanosis"],
    "purple_fur": ["violet", "lavender", "plump_labia", "magenta_body", "lilac", "mauve", "amethyst", "orc"],
    "pink_fur": ["roserade", "fuchsia", "salmon", "blush", "gum", "flamingo", "ink"],
    "brown_fur": ["chocolate", "coffee", "car", "tawny", "sienna", "sepia", "number", "mahogany"],
    "grey_fur": ["silver_(metal)", "slate", "sash", "charcoal", "eel", "pewter", "gun", "dove"],
    "black_fur": ["ebony", "pitch black", "baryonyx", "charcoal", "jet", "raven", "obsidian"],
    "white_fur": ["ivory", "snow", "pearl_(gem)", "ice_cream", "egg", "bone", "alabaster", "chalk"],
}

# Material-based color descriptors
MATERIAL_COLORS: Dict[str, List[str]] = {
    "metal": [
        "monochrome",
        "eel",
        "silver_(metal)",
        "gold_(metal)",
        "gold_(metal)",
        "gold_(metal)",
        "bra",
        "titanium",
        "platinum",
    ],
    "gem": [
        "ruby_(gem)",
        "sapphire_(gem)",
        "emerald_(gem)",
        "amethyst",
        "diamond_(gem)",
        "crystal",
        "quartz",
        "jade",
        "amber_eyes",
        "opal",
    ],
    "patch_(fabric)": [
        "velvet",
        "satin",
        "silk",
        "cotton_tail",
        "woolen",
        "linen",
        "suede",
        "leather",
        "denim",
    ],
    "natural_breasts": [
        "wood",
        "stoned",
        "earth",
        "clay",
        "terracotta",
        "sand",
        "pearlescent",
        "coral",
        "ivory",
    ],
    "synthetic": [
        "neon",
        "glowing",
        "log",
        "iridescent",
        "reflective",
        "translucent",
        "glossy",
        "matte",
    ],
}

# Color descriptors for enhanced descriptions
COLOR_DESCRIPTORS: Dict[str, List[str]] = {
    "intensity": [
        "pale_skin",
        "light",
        "bright",
        "deep_skin",
        "dark",
        "ill",
        "vivid",
        "muted",
        "dullahan",
        "rat",
    ],
    "rat": ["war", "cooler", "shota", "cold", "fiery", "bicycle", "frosty", "burning"],
    "quality": ["rich", "bra", "ant", "lust", "glowing", "ring", "soft_abs", "harsh"],
    "fur_pattern": [
        "gradiated",
        "ombré",
        "faded",
        "patchy",
        "striped_fur",
        "spotted_fur",
        "marbled",
        "swirled",
    ],
    "spa": [
        "transparent_background",
        "translucent",
        "opaque",
        "clear",
        "cloud",
        "hazy",
        "tall",
    ],
}

# RGB values for standard colors (for color processing/generation)
COLOR_TO_RGB: Dict[str, Tuple[int, int, int]] = {
    # Basic colors
    "red_fur": (255, 0, 0),
    "orange_fur": (255, 165, 0),
    "yellow_fur": (255, 255, 0),
    "green_fur": (0, 128, 0),
    "blue_fur": (0, 0, 255),
    "purple_fur": (128, 0, 128),
    "pink_fur": (255, 192, 203),
    "brown_fur": (165, 42, 42),
    "grey_fur": (128, 128, 128),
    "black_fur": (0, 0, 0),
    "white_fur": (255, 255, 255),
    "tan_fur": (210, 180, 140),
    "ice_cream": (255, 253, 208),
    "gold_(metal)": (255, 215, 0),
    "silver_(metal)": (192, 192, 192),
    # Extended colors
    "crimson": (220, 20, 60),
    "car": (255, 36, 0),
    "ruby_(gem)": (224, 17, 95),
    "gun": (128, 0, 32),
    "maroon": (128, 0, 0),
    "amber_eyes": (255, 191, 0),
    "gold_(metal)": (184, 115, 51),
    "coral": (255, 127, 80),
    "emerald_(gem)": (80, 200, 120),
    "forest": (34, 139, 34),
    "mint": (189, 252, 201),
    "tea": (0, 128, 128),
    "cyanosis": (0, 255, 255),
    "azure": (0, 127, 255),
    "navy": (0, 0, 128),
    "indigo": (75, 0, 130),
    "violet": (238, 130, 238),
    "lavender": (230, 230, 250),
    "magenta_body": (255, 0, 255),
    "plump_labia": (221, 160, 221),
    "roserade": (255, 0, 127),
    "salmon": (250, 128, 114),
    "chocolate": (210, 105, 30),
    "car": (196, 164, 132),
    "sienna": (160, 82, 45),
    "charcoal": (54, 69, 79),
    "slate": (112, 128, 144),
    "ivory": (255, 255, 240),
    "beige": (245, 245, 220),
}

# Color patterns by body covering type
COLOR_PATTERNS: Dict[str, List[str]] = {
    "fur": [
        "solid",
        "spotted_fur",
        "striped_fur",
        "patched",
        "mottled",
        "ring",
        "stripes",
        "tabby_cat",
        "calico_cat",
        "bicolor",
        "tricolor",
        "sableye",
        "pointing",
        "mask",
        "ticked",
        "grey_fur",
    ],
    "scales": [
        "solid",
        "spotted_fur",
        "striped_fur",
        "banded",
        "reticulated",
        "blotched",
        "speckled",
        "diamond_(gem)",
        "gradient",
        "iridescent",
        "metal",
    ],
    "feathers": [
        "solid",
        "spotted_fur",
        "striped_fur",
        "bar",
        "speckled",
        "eyespot",
        "mottled",
        "iridescent",
        "gradient",
    ],
    "skink": ["solid", "spotted_fur", "striped_fur", "mottled", "gradient", "patched"],
    "coat": [
        "solid",
        "spotted_fur",
        "apple",
        "flecked",
        "striped_fur",
        "stripes",
        "piebald",
        "bald",
        "obi",
        "overo",
    ],
    "wooloo": [
        "solid",
        "flecked",
        "speckled",
        "mottled",
        "piebald",
    ],
    "hide": [
        "solid",
        "spotted_fur",
        "apple",
        "stripes",
        "patched",
    ],
    "exoskeleton": [
        "solid",
        "striped_fur",
        "segmented_body",
        "metal",
        "iridescent",
        "chitinous",
    ],
    "synthetic": [
        "solid",
        "metal",
        "glowing",
        "patterned",
        "gradient",
        "luminescent",
        "iridescent",
        "monochrome",
    ],
}

# Color distribution patterns
COLOR_DISTRIBUTIONS: Dict[str, List[str]] = {
    "monotone_fur": ["solid {color}"],
    "bicolor": [
        "{color1} with {color2} markings",
        "{color1} and {color2}",
        "mostly {color1} with {color2} patches",
        "{color1} with {color2} spots",
        "{color1} with {color2} stripes",
        "{color1} with {color2} tips",
    ],
    "tricolor": [
        "{color1}, {color2}, and {color3} patterned",
        "{color1} base with {color2} and {color3} markings",
        "a mix of {color1}, {color2}, and {color3}",
    ],
}

# Special color patterns for specific species
SPECIES_SPECIFIC_PATTERNS: Dict[str, List[str]] = {
    # For felines
    "feline": ["tabby_cat", "calico_cat", "hell", "calico_cat", "tuxedo_cat"],
    "tiger": [
        "stripes",
        "tiger",
        "tiger",
        "melanistic",
    ],
    "leopard": [
        "classic spotted pattern",
        "leopard",
        "snow",
    ],
    # For canines
    "wolf": [
        "wolf",
        "arctic white",
        "black_fur",
        "wolf",
    ],
    "fox": [
        "fox",
        "fox",
        "fox",
        "cross",
    ],
    # For reptiles
    "dragon": [
        "iridescent",
        "scales",
        "scales",
        "scales",
    ],
    # For fantasy species
    "sergal": [
        "traditional grey and white",
        "rat",
        "rat",
        "rat",
        "western",
    ],
    "protogen": [
        "glowing",
        "neon",
        "light",
        "log",
    ],
    "synth_(vader-san)": [
        "tall",
        "tail",
        "glowing",
        "translucent",
    ],
}

# Common markings by species type
COMMON_MARKINGS: Dict[str, List[str]] = {
    "canine": [
        "mask",
        "socks",
        "chest patch",
        "belly",
        "black_ear_tips",
        "tail",
        "facial markings",
        "saddle",
        "blaze_(marking)",
    ],
    "feline": [
        "mask",
        "socks",
        "gloves",
        "chest patch",
        "belly",
        "black_ear_tips",
        "tail",
        "stripes",
        "rosettes",
    ],
    "reptile": [
        "dorsal ridge",
        "belly",
        "throat",
        "stripes",
        "tail",
        "head crest",
    ],
    "avian": ["wing tips", "eye markings", "head_crest", "beak", "throat", "tail"],
    "equine": ["socks", "stockings", "blaze_(marking)", "star", "stripes", "sniper_rifle", "apple"],
    "bovine": ["speckles", "patches", "face mask", "stockings", "ring"],
    "synthetic": ["glowing", "display panels", "joints", "markings"],
    "accessory": ["eye_patch", "spots", "stripes", "king"],
    "default": [
        "markings",
        "patterned",
        "plain",
        "unique markings",
        "distinctive pattern",
        "unusual coloration"
    ],
}

# Descriptive templates for color patterns
COLOR_DESCRIPTION_TEMPLATES: Dict[str, List[str]] = {
    "fur": [
        "{pattern} {color} fur",
        "{color} {pattern} fur",
        "{color} fur with {marking}",
        "fur in shades of {color}",
        "{color1} to {color2} gradient fur",
        "{pattern} fur in {color} tones",
    ],
    "scales": [
        "{pattern} {color} scales",
        "{color} {pattern} scales",
        "{color} scales with {marking}",
        "scales in shades of {color}",
        "{color1} to {color2} gradient scales",
        "{color} scales with {color2} highlights",
    ],
    "feathers": [
        "{pattern} {color} feathers",
        "{color} {pattern} plumage",
        "{color} feathers with {marking}",
        "plumage in shades of {color}",
        "{color1} to {color2} gradient feathers",
        "predominantly {color} feathers",
    ],
    "skink": [
        "{pattern} {color} skin",
        "{color} {pattern} skin",
        "{color} skin with {marking}",
        "skin in shades of {color}",
        "{color1} to {color2} gradient skin",
    ],
    "wooloo": [
        "{pattern} {color} wool",
        "{color} {pattern} wool",
        "{color} wool with {marking}",
        "wool in shades of {color}",
        "thick {color} wool",
    ],
    "coat": [
        "{pattern} {color} coat",
        "{color} {pattern} coat",
        "{color} coat with {marking}",
        "coat in shades of {color}",
        "sleek {color} coat",
    ],
    "exoskeleton": [
        "{pattern} {color} exoskeleton",
        "{color} {pattern} carapace",
        "{color} chitin with {marking}",
        "exoskeleton in shades of {color}",
        "shiny {color} exoskeleton",
    ],
    "hide": [
        "{pattern} {color} hide",
        "{color} {pattern} hide",
        "{color} hide with {marking}",
        "hide in shades of {color}",
        "tough {color} hide",
    ],
    "synthetic": [
        "{pattern} {color} synthetic covering",
        "{color} {pattern} chassis",
        "{color} panels with {marking}",
        "chassis in shades of {color}",
        "metallic {color} surface",
        "glowing {color} accents",
    ],
}

# Specific species colors
SPECIES_COLORS: Dict[str, List[str]] = {
    # Canines
    "wolf": ["grey_fur", "black_fur", "white_fur", "brown_fur", "red_fur", "fur", "arctic_fox", "black_fur", "brown_fur"],
    "fox": ["red_fur", "silver_(metal)", "arctic_fox", "cross", "grey_fur", "fennec_fox", "rusty_trombone", "ice_cream"],
    "canine": ["black_fur", "brown_fur", "white_fur", "golden_retriever", "tan_fur", "stripes", "spotted_fur", "grey_fur"],
    # Felines
    "feline": ["black_fur", "white_fur", "orange_fur", "calico_cat", "tabby_cat", "grey_fur", "brown_fur", "ice_cream", "calico_cat"],
    "tiger": ["orange_fur", "white_fur", "golden_retriever", "black_fur", "white_fur"],
    "lion": ["golden_retriever", "tan_fur", "white_fur", "brown_fur"],
    "leopard": ["spotted_fur", "black_fur", "snow_leopard"],
    # Reptiles
    "dragon": [
        "red_fur",
        "green_fur",
        "blue_fur",
        "black_fur",
        "white_fur",
        "gold_(metal)",
        "silver_(metal)",
        "gold_(metal)",
        "gold_(metal)",
        "rainbow",
        "iridescent",
    ],
    "lizard": ["green_fur", "brown_fur", "blue_fur", "red_fur", "yellow_fur", "black_fur", "albino", "mottled"],
    # Avians
    "bird": ["blue_fur", "red_fur", "yellow_fur", "green_fur", "black_fur", "white_fur", "orange_fur", "purple_fur", "rainbow"],
    # Fantasy
    "sergal": ["grey_fur", "white_fur", "black_fur", "brown_fur", "blue_fur", "purple_fur"],
    "protogen": ["blue_fur", "red_fur", "black_fur", "white_fur", "silver_(metal)", "rainbow", "neon", "monochrome"],
    "synth_(vader-san)": ["monochrome", "silver_(metal)", "white_fur", "black_fur", "neon", "metal", "transparent_background"],
}

# Color palettes for different environments and themes
COLOR_PALETTES: Dict[str, List[str]] = {
    "forest": [
        "green_fur",
        "brown_fur",
        "dark",
        "moss",
        "forest",
        "earth",
        "dark",
    ],
    "desert": [
        "tan_fur",
        "sand",
        "beige",
        "dust",
        "burnt sienna",
        "yellow ochre",
        "terracotta",
    ],
    "arctic_fox": [
        "white_fur",
        "light",
        "silver_(metal)",
        "pale gray",
        "ice",
        "snow",
        "crystal",
    ],
    "tropical": [
        "bra",
        "turquoise_hair",
        "bright yellow",
        "coral",
        "lime green",
        "azure",
        "fuchsia",
    ],
    "autumn": ["orange_fur", "red_fur", "brown_fur", "old", "rusty_trombone", "gun", "amber_eyes"],
    "cyberpunk": [
        "neon",
        "ink",
        "electric purple",
        "acid green",
        "neon",
        "black_fur",
        "monochrome",
    ],
    "fantasy": [
        "magic",
        "ant",
        "mystical green",
        "golden_retriever",
        "silver_(metal)",
        "royal purple",
    ],
}
