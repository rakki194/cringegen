"""
Unified Color System for CringeGen

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

from typing import Dict, List, Set, Tuple, Any, Optional

# Basic colors for general use
BASIC_COLORS: List[str] = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "gray",
    "black",
    "white",
    "tan",
    "cream",
    "gold",
    "silver",
]

# Expanded color palette with more specific hues
EXTENDED_COLORS: Dict[str, List[str]] = {
    "red": ["crimson", "scarlet", "ruby", "burgundy", "maroon", "vermilion", "carmine"],
    "orange": ["amber", "tangerine", "rust", "burnt orange", "copper", "peach", "coral"],
    "yellow": ["lemon", "gold", "mustard", "honey", "saffron", "daffodil", "brass"],
    "green": ["emerald", "lime", "mint", "forest green", "olive", "jade", "sage", "teal"],
    "blue": ["azure", "cobalt", "navy", "indigo", "sapphire", "cerulean", "turquoise", "cyan"],
    "purple": ["violet", "lavender", "plum", "magenta", "lilac", "mauve", "amethyst", "orchid"],
    "pink": ["rose", "fuchsia", "salmon", "blush", "bubblegum", "flamingo", "hot pink"],
    "brown": ["chocolate", "coffee", "caramel", "tawny", "sienna", "sepia", "umber", "mahogany"],
    "gray": ["silver", "slate", "ash", "charcoal", "steel", "pewter", "gunmetal", "dove"],
    "black": ["ebony", "pitch black", "onyx", "charcoal", "jet black", "raven", "obsidian"],
    "white": ["ivory", "snow", "pearl", "cream", "eggshell", "bone", "alabaster", "chalk"],
}

# Material-based color descriptors
MATERIAL_COLORS: Dict[str, List[str]] = {
    "metallic": [
        "chrome",
        "steel",
        "silver",
        "gold",
        "bronze",
        "copper",
        "brass",
        "titanium",
        "platinum",
    ],
    "gemstone": [
        "ruby",
        "sapphire",
        "emerald",
        "amethyst",
        "diamond",
        "crystal",
        "quartz",
        "jade",
        "amber",
        "opal",
    ],
    "fabric": [
        "velvet",
        "satin",
        "silky",
        "cotton",
        "woolen",
        "linen",
        "suede",
        "leather",
        "denim",
    ],
    "natural": [
        "wooden",
        "stone",
        "earthy",
        "clay",
        "terracotta",
        "sandy",
        "pearlescent",
        "coral",
        "ivory",
    ],
    "synthetic": [
        "neon",
        "glowing",
        "holographic",
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
        "pale",
        "light",
        "bright",
        "deep",
        "dark",
        "brilliant",
        "vivid",
        "muted",
        "dull",
        "saturated",
    ],
    "temperature": ["warm", "cool", "hot", "cold", "fiery", "icy", "frosty", "burning"],
    "quality": ["rich", "vibrant", "radiant", "lustrous", "glowing", "shimmering", "soft", "harsh"],
    "pattern": [
        "gradiated",
        "ombré",
        "faded",
        "patchy",
        "striped",
        "spotted",
        "marbled",
        "swirled",
    ],
    "transparency": [
        "transparent",
        "translucent",
        "opaque",
        "clear",
        "cloudy",
        "hazy",
        "crystalline",
    ],
}

# RGB values for standard colors (for color processing/generation)
COLOR_TO_RGB: Dict[str, Tuple[int, int, int]] = {
    # Basic colors
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "tan": (210, 180, 140),
    "cream": (255, 253, 208),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    # Extended colors
    "crimson": (220, 20, 60),
    "scarlet": (255, 36, 0),
    "ruby": (224, 17, 95),
    "burgundy": (128, 0, 32),
    "maroon": (128, 0, 0),
    "amber": (255, 191, 0),
    "copper": (184, 115, 51),
    "coral": (255, 127, 80),
    "emerald": (80, 200, 120),
    "forest green": (34, 139, 34),
    "mint": (189, 252, 201),
    "teal": (0, 128, 128),
    "cyan": (0, 255, 255),
    "azure": (0, 127, 255),
    "navy": (0, 0, 128),
    "indigo": (75, 0, 130),
    "violet": (238, 130, 238),
    "lavender": (230, 230, 250),
    "magenta": (255, 0, 255),
    "plum": (221, 160, 221),
    "rose": (255, 0, 127),
    "salmon": (250, 128, 114),
    "chocolate": (210, 105, 30),
    "caramel": (196, 164, 132),
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
        "spotted",
        "striped",
        "patched",
        "mottled",
        "ringed",
        "brindle",
        "tabby",
        "calico",
        "bicolor",
        "tricolor",
        "sable",
        "point",
        "masked",
        "ticked",
        "merle",
    ],
    "scales": [
        "solid",
        "spotted",
        "striped",
        "banded",
        "reticulated",
        "blotched",
        "speckled",
        "diamond",
        "gradient",
        "iridescent",
        "metallic",
    ],
    "feathers": [
        "solid",
        "spotted",
        "striped",
        "barred",
        "speckled",
        "eyespot",
        "mottled",
        "iridescent",
        "gradient",
    ],
    "skin": ["solid", "spotted", "striped", "mottled", "gradient", "patched"],
    "coat": [
        "solid",
        "spotted",
        "dappled",
        "flecked",
        "striped",
        "brindle",
        "piebald",
        "skewbald",
        "tobiano",
        "overo",
    ],
    "wool": [
        "solid",
        "flecked",
        "speckled",
        "mottled",
        "piebald",
    ],
    "hide": [
        "solid",
        "spotted",
        "dappled",
        "brindle",
        "patched",
    ],
    "exoskeleton": [
        "solid",
        "striped",
        "segmented",
        "metallic",
        "iridescent",
        "chitinous",
    ],
    "synthetic": [
        "solid",
        "metallic",
        "glowing",
        "patterned",
        "gradient",
        "luminescent",
        "iridescent",
        "chrome",
    ],
}

# Color distribution patterns
COLOR_DISTRIBUTIONS: Dict[str, List[str]] = {
    "monotone": ["solid {color}"],
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
    "cat": ["tabby", "calico", "tortoiseshell", "colorpoint", "tuxedo"],
    "tiger": [
        "traditional orange and black stripes",
        "white tiger with black stripes",
        "golden tiger",
        "melanistic (black)",
    ],
    "leopard": [
        "classic spotted pattern",
        "black leopard (panther)",
        "snow leopard (pale with dark rosettes)",
    ],
    # For canines
    "wolf": [
        "timber wolf (gray with black points)",
        "arctic white",
        "black phase",
        "red wolf coloration",
    ],
    "fox": [
        "red fox (rusty with white belly)",
        "arctic fox (white or blue-gray)",
        "silver fox (black with silver-tipped guard hairs)",
        "cross fox (red with dark markings)",
    ],
    # For reptiles
    "dragon": [
        "iridescent scales",
        "metallic scales",
        "ember-glowing scales",
        "crystalline scales",
    ],
    # For fantasy species
    "sergal": [
        "traditional grey and white",
        "northern coloration",
        "southern coloration",
        "eastern coloration",
        "western coloration",
    ],
    "protogen": [
        "metallic panels with glowing accents",
        "chrome with neon detailing",
        "matte with bright LED highlights",
        "holographic display panels",
    ],
    "synth": [
        "metallic with LED accents",
        "chrome with fiber optic details",
        "matte finish with glowing joints",
        "translucent panels with internal lighting",
    ],
}

# Common markings by species type
COMMON_MARKINGS: Dict[str, List[str]] = {
    "canine": [
        "mask",
        "socks",
        "chest patch",
        "belly patch",
        "ear tips",
        "tail tip",
        "facial markings",
        "saddle",
        "blaze",
    ],
    "feline": [
        "mask",
        "socks",
        "gloves",
        "chest patch",
        "belly patch",
        "ear tips",
        "tail tip",
        "facial stripes",
        "rosettes",
    ],
    "reptile": [
        "dorsal ridge",
        "belly plates",
        "throat fan",
        "eye stripes",
        "tail bands",
        "head crest",
    ],
    "avian": ["wing tips", "eye markings", "crest", "beak color", "throat patch", "tail bands"],
    "equine": ["socks", "stockings", "blaze", "star", "stripe", "snip", "dapples"],
    "bovine": ["speckles", "patches", "face mask", "stockings", "muzzle ring"],
    "synthetic": ["glowing seams", "display panels", "luminescent joints", "interface markings"],
    "default": ["patch", "spot", "stripe", "marking"],
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
    "skin": [
        "{pattern} {color} skin",
        "{color} {pattern} skin",
        "{color} skin with {marking}",
        "skin in shades of {color}",
        "{color1} to {color2} gradient skin",
    ],
    "wool": [
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
    "wolf": ["gray", "black", "white", "brown", "red", "timber", "arctic", "black phase", "agouti"],
    "fox": ["red", "silver", "arctic", "cross", "gray", "fennec", "rust", "cream"],
    "dog": ["black", "brown", "white", "golden", "tan", "brindle", "spotted", "merle"],
    # Felines
    "cat": ["black", "white", "orange", "calico", "tabby", "gray", "brown", "cream", "colorpoint"],
    "tiger": ["orange", "white", "golden", "black", "bengal white"],
    "lion": ["golden", "tan", "white", "brown"],
    "leopard": ["golden spotted", "black", "snow leopard silver"],
    # Reptiles
    "dragon": [
        "red",
        "green",
        "blue",
        "black",
        "white",
        "gold",
        "silver",
        "bronze",
        "copper",
        "rainbow",
        "iridescent",
    ],
    "lizard": ["green", "brown", "blue", "red", "yellow", "black", "albino", "mottled"],
    # Avians
    "bird": ["blue", "red", "yellow", "green", "black", "white", "orange", "purple", "rainbow"],
    # Fantasy
    "sergal": ["gray", "white", "black", "brown", "blue", "purple"],
    "protogen": ["blue", "red", "black", "white", "silver", "rainbow", "neon", "chrome"],
    "synth": ["chrome", "silver", "white", "black", "neon", "metallic", "transparent"],
}

# Color palettes for different environments and themes
COLOR_PALETTES: Dict[str, List[str]] = {
    "forest": [
        "green",
        "brown",
        "dark green",
        "moss green",
        "forest green",
        "earthy brown",
        "dark brown",
    ],
    "desert": [
        "tan",
        "sand",
        "beige",
        "dusty orange",
        "burnt sienna",
        "yellow ochre",
        "terracotta",
    ],
    "arctic": [
        "white",
        "light blue",
        "silver",
        "pale gray",
        "ice blue",
        "snow white",
        "crystal blue",
    ],
    "tropical": [
        "vibrant green",
        "turquoise",
        "bright yellow",
        "coral",
        "lime green",
        "azure",
        "fuchsia",
    ],
    "autumn": ["orange", "red", "brown", "golden yellow", "rust", "burgundy", "amber"],
    "cyberpunk": [
        "neon blue",
        "hot pink",
        "electric purple",
        "acid green",
        "neon yellow",
        "black",
        "chrome",
    ],
    "fantasy": [
        "magical blue",
        "enchanted purple",
        "mystical green",
        "golden",
        "silver",
        "royal purple",
    ],
}
