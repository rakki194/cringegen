"""
Body Covering Data for cringegen

This module contains information about body coverings for different species taxonomies,
including fur, scales, feathers, and other surface characteristics.
"""

from typing import Dict, List

# Body covering types by taxonomy group
BODY_COVERING_BY_TAXONOMY: Dict[str, List[str]] = {
    "canine": ["fur", "fluffy", "soft fur", "thick fur", "short fur", "long fur"],
    "feline": ["fur", "soft fur", "sleek fur", "short fur", "fluffy"],
    "equine": ["short fur", "hide", "coat", "mane", "tail hair"],
    "bovine": ["hide", "short fur", "coat", "thick hide"],
    "rodent": ["fur", "soft fur", "whiskers", "fluffy"],
    "lagomorph": ["fur", "soft fur", "fluffy", "fuzzy"],
    "reptile": ["scales", "tough scales", "smooth scales", "dry scales", "glossy scales"],
    "avian": ["feathers", "plumage", "soft feathers", "colorful feathers", "wing feathers"],
    "deer": ["fur", "short fur", "dappled coat", "spotted coat"],
    "procyonid": ["fur", "soft fur", "thick fur", "fluffy"],
    "mustelid": ["fur", "sleek fur", "thick fur", "soft fur"],
    "ursid_humanoid": ["fur", "thick fur", "heavy fur", "dense fur"],
    "mephitid": ["fur", "striped fur", "soft fur", "fluffy"],
    "hyaenid": ["fur", "spotted fur", "coarse fur", "bristly fur"],
    "caprid": ["wool", "curly wool", "thick coat", "soft wool"],
    "ovid": ["wool", "fluffy wool", "thick wool", "soft wool"],
    "macropod": ["fur", "short fur", "dense fur", "pouch"],
    "marsupial": ["fur", "short fur", "dense fur", "pouch"],
    "cetacean": ["smooth skin", "rubbery skin", "blubber", "slick skin"],
    "shark": ["rough skin", "dermal denticles", "cartilage", "slick skin"],
    "fish": ["scales", "slippery scales", "iridescent scales", "colorful scales"],
    "cephalopod": ["soft body", "tentacles", "chromatophores", "smooth skin"],
    "amphibian": ["moist skin", "smooth skin", "slick skin", "permeable skin"],
    "yangochiropteran": ["fur", "leathery wings", "soft fur", "wing membranes"],
    # Fantasy species
    "sergal": ["fur", "fluffy", "scaled patches", "mane"],
    "protogen": ["fur", "metal parts", "screens", "LED accents", "glass", "circuits", "visor"],
    "synth_(vader-san)": [
        "plastic",
        "silicone",
        "metal parts",
        "screens",
        "LED accents",
        "glowing parts",
    ],
    "primagen": ["fur", "metal parts", "screens", "LED accents", "visor", "protruding parts"],
    "wickerbeast": ["fur", "patterns", "markings", "fluffy"],
    "dutch_angel_dragon": ["fur", "fluffy", "wing feathers", "fluff", "plush"],
    "avali": ["feathers", "downy feathers", "quills", "talons", "scales"],
    "kobold": ["scales", "horns", "spines", "tough scales"],
    "eastern_dragon": ["scales", "silken scales", "whiskers", "mane", "glowing scales"],
    "manticore": ["fur", "scales", "leathery wings", "scorpion tail", "mane"],
    "kitsune": ["fur", "silky fur", "multiple tails", "glowing markings"],
    "latex_creature": ["latex", "shiny", "stretchy", "dripping", "glossy", "fluid"],
    "goo_creature": ["goo", "slime", "gel", "dripping", "glossy", "fluid", "sticky"],
    "drekkubus": ["scales", "horns", "soft scales", "leathery wings", "heart-tipped tail"],
    "sparkledog": ["colorful fur", "neon fur", "glowing markings", "rainbow", "sparkles"],
    # Pokemon species
    "pokemon": ["various", "elemental markings", "distinctive features"],
}

# Body covering patterns by species type
BODY_COVERING_PATTERNS: Dict[str, List[str]] = {
    "canine": ["solid", "bicolor", "tricolor", "merle", "brindle", "spotted", "marked"],
    "feline": ["solid", "tabby", "tortoiseshell", "calico", "spotted", "striped", "pointed"],
    "equine": ["solid", "dappled", "pinto", "appaloosa", "bay", "roan"],
    "bovine": ["solid", "spotted", "patched", "banded"],
    "reptile": ["solid", "banded", "spotted", "reticulated", "striped", "diamond pattern"],
    "avian": ["solid", "mottled", "speckled", "striped", "patterned", "barred"],
}

# Markings and features by taxonomy
BODY_MARKINGS: Dict[str, List[str]] = {
    "canine": ["facial mask", "eyebrows", "ear tips", "muzzle", "socks", "chest", "tail tip"],
    "feline": ["stripes", "rosettes", "spots", "ear tips", "mask", "tail tip", "paw pads"],
    "equine": ["blaze", "star", "snip", "socks", "stockings", "bald face"],
    "bovine": ["facial markings", "leg markings", "underbelly", "tail switch"],
    "avian": ["crest", "eye rings", "throat patch", "wing bars", "tail bands"],
    "reptile": ["dorsal stripe", "banding", "head pattern", "tail rings"],
}

# Body texture variations
BODY_TEXTURES: Dict[str, List[str]] = {
    "fur": ["soft", "plush", "silky", "smooth", "thick", "fluffy", "coarse", "bristly", "downy"],
    "scales": ["smooth", "rough", "glossy", "matte", "overlapping", "armored", "iridescent"],
    "feathers": ["smooth", "fluffy", "sleek", "downy", "ruffled", "iridescent", "glossy"],
    "skin": ["smooth", "rough", "leathery", "velvety", "tough", "wrinkled", "glossy"],
}
