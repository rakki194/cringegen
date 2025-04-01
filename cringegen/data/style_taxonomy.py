"""
Hierarchical Style Taxonomy System for cringegen

This module provides a hierarchical taxonomy of art styles, mediums, and aesthetic approaches.
It organizes styles in a more structured way with clear relationships and categories to
enable more sophisticated style selection and combination.

Key components:
- Style taxonomy by medium, period, technique, etc.
- Style relationships and cross-references
- Style attributes and characteristics
- Helper functions for style selection and combination
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Union

# =============================================================================
# Style Taxonomy Core Structure
# =============================================================================

# Medium-based classification (the material/technology used)
STYLE_MEDIUM: Dict[str, Dict[str, Any]] = {
    "traditional": {
        "name": "Traditional Art",
        "description": "Art created using physical media and traditional techniques",
        "techniques": {
            "painting": {
                "name": "Painting",
                "description": "Application of paint to a surface",
                "materials": [
                    "oil",
                    "acrylic",
                    "watercolor",
                    "gouache",
                    "tempera",
                    "encaustic",
                    "fresco",
                    "ink wash",
                ],
                "tools": ["brush", "palette knife", "sponge", "airbrush"],
                "surfaces": ["canvas", "wood panel", "paper", "wall", "board"],
            },
            "drawing": {
                "name": "Drawing",
                "description": "Creating images using lines and shading",
                "materials": [
                    "graphite",
                    "charcoal",
                    "ink",
                    "colored pencil",
                    "pastel",
                    "crayon",
                    "marker",
                    "chalk",
                    "silverpoint",
                ],
                "tools": ["pencil", "pen", "brush", "stylus", "eraser"],
                "surfaces": ["paper", "board", "vellum", "parchment"],
            },
            "printmaking": {
                "name": "Printmaking",
                "description": "Creating images by transferring ink from a matrix to paper",
                "subtypes": [
                    "woodcut",
                    "linocut",
                    "engraving",
                    "etching",
                    "lithography",
                    "screen printing",
                    "monotype",
                    "collagraph",
                ],
                "tools": ["press", "brayer", "burin", "needle", "screen"],
                "surfaces": ["paper", "fabric", "wood", "metal plate"],
            },
            "sculpture": {
                "name": "Sculpture",
                "description": "Three-dimensional art made by shaping materials",
                "materials": [
                    "clay",
                    "stone",
                    "wood",
                    "metal",
                    "plaster",
                    "glass",
                    "wire",
                    "found objects",
                    "mixed media",
                ],
                "techniques": [
                    "carving",
                    "modeling",
                    "casting",
                    "assembling",
                    "welding",
                    "mold-making",
                    "construction",
                ],
            },
            "textile": {
                "name": "Textile Art",
                "description": "Art using fabric and fiber materials",
                "techniques": [
                    "weaving",
                    "embroidery",
                    "quilting",
                    "knitting",
                    "crocheting",
                    "felting",
                    "tapestry",
                    "batik",
                    "tie-dye",
                ],
                "materials": ["fabric", "yarn", "thread", "fiber", "wool", "silk", "cotton"],
            },
            "collage": {
                "name": "Collage",
                "description": "Art created by assembling different forms and materials",
                "materials": ["paper", "photographs", "fabric", "found objects", "mixed media"],
                "techniques": ["cutting", "pasting", "assembling", "layering"],
            },
        },
    },
    "digital": {
        "name": "Digital Art",
        "description": "Art created using digital technology",
        "techniques": {
            "digital_painting": {
                "name": "Digital Painting",
                "description": "Creating painterly images using digital tools",
                "software": [
                    "Photoshop",
                    "Procreate",
                    "Krita",
                    "Corel Painter",
                    "Clip Studio Paint",
                    "GIMP",
                    "Paint Tool SAI",
                ],
                "approaches": ["brushwork", "layering", "blending", "texturing"],
            },
            "vector_art": {
                "name": "Vector Art",
                "description": "Art created using vector graphics software",
                "software": ["Adobe Illustrator", "Inkscape", "Affinity Designer", "CorelDRAW"],
                "characteristics": ["scalable", "geometric", "clean lines", "flat colors"],
            },
            "3d_art": {
                "name": "3D Digital Art",
                "description": "Creating three-dimensional art in a digital environment",
                "software": [
                    "Blender",
                    "Maya",
                    "ZBrush",
                    "Cinema 4D",
                    "3ds Max",
                    "Substance Painter",
                    "Houdini",
                ],
                "techniques": [
                    "modeling",
                    "sculpting",
                    "texturing",
                    "rigging",
                    "animation",
                    "rendering",
                    "simulations",
                ],
            },
            "pixel_art": {
                "name": "Pixel Art",
                "description": "Digital art created at the pixel level",
                "software": ["Aseprite", "Pyxel Edit", "Photoshop", "GIMP"],
                "characteristics": ["pixelated", "limited palette", "hand-placed pixels"],
            },
            "photomanipulation": {
                "name": "Photo Manipulation",
                "description": "Transforming or combining photographs to create new imagery",
                "software": ["Photoshop", "GIMP", "Affinity Photo"],
                "techniques": ["compositing", "retouching", "color grading", "effects"],
            },
            "generative": {
                "name": "Generative Art",
                "description": "Art created with autonomous systems",
                "subtypes": [
                    "algorithm-based",
                    "AI-generated",
                    "diffusion art",
                    "GAN art",
                    "procedural",
                    "fractal art",
                ],
                "software": ["Processing", "Stable Diffusion", "Midjourney", "DALL-E"],
            },
        },
    },
    "photography": {
        "name": "Photography",
        "description": "Art created through capturing light with cameras",
        "genres": {
            "portrait": {
                "name": "Portrait Photography",
                "description": "Capturing the personality of a person or group",
                "subtypes": [
                    "headshot",
                    "environmental",
                    "fashion",
                    "self-portrait",
                    "group portrait",
                ],
            },
            "landscape": {
                "name": "Landscape Photography",
                "description": "Capturing natural scenery and environments",
                "subtypes": ["scenic", "nature", "urban landscape", "seascape", "nightscape"],
            },
            "documentary": {
                "name": "Documentary Photography",
                "description": "Capturing real-life events and situations",
                "subtypes": ["photojournalism", "street photography", "social documentary"],
            },
            "abstract": {
                "name": "Abstract Photography",
                "description": "Non-representational visual imagery through photography",
                "techniques": [
                    "macro",
                    "intentional camera movement",
                    "light painting",
                    "defocusing",
                ],
            },
            "commercial": {
                "name": "Commercial Photography",
                "description": "Photography for commercial purposes",
                "subtypes": ["product", "food", "fashion", "real estate", "advertising"],
            },
        },
        "techniques": [
            "digital",
            "film",
            "long exposure",
            "HDR",
            "black and white",
            "infrared",
            "panoramic",
            "time-lapse",
            "light painting",
        ],
    },
    "mixed_media": {
        "name": "Mixed Media",
        "description": "Art using a combination of different media",
        "combinations": [
            "traditional + digital",
            "painting + collage",
            "photography + painting",
            "sculpture + digital",
            "drawing + painting",
            "printmaking + collage",
        ],
    },
}

# Historical and contemporary art movements
STYLE_MOVEMENTS: Dict[str, Dict[str, Any]] = {
    "classical": {
        "name": "Classical Art",
        "period": "Ancient to Renaissance",
        "characteristics": ["realistic", "idealized", "harmonious", "balanced", "proportional"],
        "notable_examples": ["Greek sculpture", "Roman frescoes", "Renaissance paintings"],
    },
    "renaissance": {
        "name": "Renaissance",
        "period": "14th-17th century",
        "characteristics": [
            "perspective",
            "anatomy",
            "realism",
            "classical themes",
            "religious themes",
        ],
        "notable_artists": ["Leonardo da Vinci", "Michelangelo", "Raphael", "Botticelli"],
    },
    "baroque": {
        "name": "Baroque",
        "period": "17th-18th century",
        "characteristics": ["dramatic", "dynamic", "ornate", "theatrical", "chiaroscuro"],
        "notable_artists": ["Caravaggio", "Rembrandt", "Rubens", "Bernini"],
    },
    "romanticism": {
        "name": "Romanticism",
        "period": "Late 18th-19th century",
        "characteristics": ["emotional", "individualistic", "nature", "imagination", "exotic"],
        "notable_artists": ["Turner", "Friedrich", "Delacroix", "Goya"],
    },
    "impressionism": {
        "name": "Impressionism",
        "period": "19th century",
        "characteristics": [
            "light effects",
            "visible brushstrokes",
            "outdoor scenes",
            "everyday life",
        ],
        "notable_artists": ["Monet", "Renoir", "Degas", "Cassatt"],
    },
    "expressionism": {
        "name": "Expressionism",
        "period": "Early 20th century",
        "characteristics": ["emotional", "subjective", "distorted", "intense colors"],
        "notable_artists": ["Van Gogh", "Munch", "Kandinsky", "Kirchner"],
    },
    "cubism": {
        "name": "Cubism",
        "period": "Early 20th century",
        "characteristics": ["geometric forms", "multiple perspectives", "fragmented", "abstract"],
        "notable_artists": ["Picasso", "Braque", "Léger", "Gris"],
    },
    "surrealism": {
        "name": "Surrealism",
        "period": "1920s-1950s",
        "characteristics": ["dreamlike", "unconscious", "irrational", "symbolic", "juxtaposition"],
        "notable_artists": ["Dalí", "Magritte", "Ernst", "Kahlo"],
    },
    "abstract": {
        "name": "Abstract Art",
        "period": "20th century-present",
        "characteristics": ["non-representational", "geometric", "color-focused", "form-focused"],
        "subtypes": [
            "abstract expressionism",
            "geometric abstraction",
            "color field",
            "lyrical abstraction",
        ],
    },
    "pop_art": {
        "name": "Pop Art",
        "period": "1950s-1970s",
        "characteristics": [
            "popular culture",
            "mass media",
            "bold colors",
            "irony",
            "everyday objects",
        ],
        "notable_artists": ["Warhol", "Lichtenstein", "Oldenburg", "Hamilton"],
    },
    "contemporary": {
        "name": "Contemporary Art",
        "period": "Late 20th century-present",
        "characteristics": ["diverse", "conceptual", "experimental", "multi-disciplinary"],
        "subtypes": [
            "installation",
            "performance",
            "new media",
            "digital art",
            "conceptual art",
            "street art",
            "video art",
        ],
    },
}

# Genre and subject matter
STYLE_GENRES: Dict[str, Dict[str, Any]] = {
    "portrait": {
        "name": "Portrait",
        "description": "Depiction of a person or group",
        "subtypes": ["self-portrait", "group portrait", "formal portrait", "candid portrait"],
    },
    "landscape": {
        "name": "Landscape",
        "description": "Depiction of natural scenery",
        "subtypes": ["seascape", "cityscape", "wilderness", "pastoral", "fantasy landscape"],
    },
    "still_life": {
        "name": "Still Life",
        "description": "Depiction of inanimate objects",
        "subtypes": ["vanitas", "flower painting", "food still life", "tabletop arrangement"],
    },
    "narrative": {
        "name": "Narrative Art",
        "description": "Art that tells a story",
        "subtypes": ["historical", "mythological", "religious", "allegorical", "literary"],
    },
    "abstract": {
        "name": "Abstract Art",
        "description": "Non-representational art",
        "subtypes": ["geometric", "lyrical", "expressionist", "minimalist"],
    },
    "figurative": {
        "name": "Figurative Art",
        "description": "Art representing recognizable forms, especially human figures",
        "subtypes": ["nude", "clothed figure", "figure study", "group scene"],
    },
    "fantasy": {
        "name": "Fantasy Art",
        "description": "Art depicting imaginative or supernatural themes",
        "subtypes": ["high fantasy", "dark fantasy", "surreal fantasy", "mythological"],
    },
    "sci_fi": {
        "name": "Science Fiction Art",
        "description": "Art depicting futuristic or technological themes",
        "subtypes": ["space", "cyberpunk", "dystopian", "mecha", "alien worlds"],
    },
    "horror": {
        "name": "Horror Art",
        "description": "Art intended to disturb, frighten or shock",
        "subtypes": ["gothic", "body horror", "cosmic horror", "psychological horror"],
    },
    "wildlife": {
        "name": "Wildlife Art",
        "description": "Art depicting animals in their natural habitat",
        "subtypes": ["bird art", "marine life", "big cat art", "insect studies"],
    },
}

# Animation and cartoon styles
STYLE_ANIMATION: Dict[str, Dict[str, Any]] = {
    "traditional_animation": {
        "name": "Traditional Animation",
        "description": "Hand-drawn, frame-by-frame animation",
        "subtypes": ["cel animation", "rotoscoping", "cutout animation"],
        "examples": ["Disney classics", "Studio Ghibli films", "Looney Tunes"],
    },
    "stop_motion": {
        "name": "Stop Motion Animation",
        "description": "Physical objects animated frame-by-frame",
        "subtypes": ["claymation", "puppet animation", "object animation", "pixilation"],
        "examples": ["Wallace and Gromit", "Coraline", "Kubo and the Two Strings"],
    },
    "3d_animation": {
        "name": "3D Computer Animation",
        "description": "Animation created in a digital 3D environment",
        "subtypes": ["photorealistic", "stylized", "motion capture based"],
        "examples": ["Pixar films", "Dreamworks films", "video game cinematics"],
    },
    "anime": {
        "name": "Anime",
        "description": "Japanese animation style",
        "subtypes": ["shonen", "shojo", "seinen", "josei", "mecha", "magical girl"],
        "studios": ["Studio Ghibli", "Kyoto Animation", "MAPPA", "Bones", "Ufotable"],
    },
    "cartoon": {
        "name": "Cartoon",
        "description": "Simplified, often exaggerated drawing style",
        "subtypes": [
            "classic cartoon",
            "Saturday morning cartoon",
            "newspaper comic",
            "political cartoon",
            "web comic",
        ],
    },
    "experimental_animation": {
        "name": "Experimental Animation",
        "description": "Non-conventional animation techniques and styles",
        "subtypes": ["abstract animation", "direct-on-film", "sand animation", "paint-on-glass"],
    },
}

# Aesthetic movements and visual styles
STYLE_AESTHETICS: Dict[str, Dict[str, Any]] = {
    "minimalist": {
        "name": "Minimalism",
        "description": "Simplified, stripped down aesthetic",
        "characteristics": ["simple", "clean", "uncluttered", "monochromatic", "geometric"],
        "applications": ["art", "design", "architecture", "fashion", "interior design"],
    },
    "maximalist": {
        "name": "Maximalism",
        "description": "Bold, complex, ornate aesthetic",
        "characteristics": ["busy", "colorful", "pattern-rich", "textured", "eclectic"],
        "applications": ["art", "design", "fashion", "interior design"],
    },
    "retro": {
        "name": "Retro",
        "description": "Style inspired by the recent past",
        "periods": ["50s", "60s", "70s", "80s", "90s", "Y2K"],
        "characteristics": ["nostalgic", "vintage-inspired", "era-specific elements"],
    },
    "futuristic": {
        "name": "Futuristic",
        "description": "Forward-looking aesthetic",
        "subtypes": ["sci-fi", "cyberpunk", "solarpunk", "space age", "techno-utopian"],
        "characteristics": ["sleek", "technological", "innovative", "speculative"],
    },
    "punk": {
        "name": "Punk Aesthetics",
        "description": "Counter-cultural aesthetics",
        "subtypes": [
            "cyberpunk",
            "steampunk",
            "dieselpunk",
            "biopunk",
            "solarpunk",
            "clockpunk",
            "gothpunk",
        ],
        "characteristics": ["rebellious", "DIY", "anti-establishment", "gritty"],
    },
    "kawaii": {
        "name": "Kawaii",
        "description": "Japanese cute aesthetic",
        "characteristics": ["cute", "colorful", "simple", "childlike", "rounded forms"],
        "applications": ["character design", "fashion", "products", "illustration"],
    },
    "vaporwave": {
        "name": "Vaporwave",
        "description": "Internet-born nostalgic aesthetic",
        "characteristics": [
            "80s/90s nostalgia",
            "glitch",
            "pastel colors",
            "classical sculpture",
            "early digital graphics",
            "Japanese text",
        ],
    },
    "cottagecore": {
        "name": "Cottagecore",
        "description": "Romanticized rural lifestyle aesthetic",
        "characteristics": ["pastoral", "rustic", "natural", "traditional", "nostalgic"],
        "related_styles": ["farmcore", "grandmacore", "fairycore", "goblincore"],
    },
}

# =============================================================================
# Style Relationships and Cross-References
# =============================================================================

# Style relationships - shows connections between different styles
STYLE_RELATIONSHIPS: Dict[str, Dict[str, List[str]]] = {
    "complementary": {
        # Styles that work well together
        "impressionism": ["post-impressionism", "plein air", "naturalism"],
        "cubism": ["futurism", "abstract", "constructivism"],
        "surrealism": ["expressionism", "dada", "fantasy art", "psychedelic"],
        "minimalism": ["geometric abstraction", "color field", "modern design"],
    },
    "derivative": {
        # Styles that evolved from others
        "post-impressionism": ["impressionism"],
        "abstract expressionism": ["expressionism", "abstract"],
        "neo-expressionism": ["expressionism"],
        "pop surrealism": ["pop art", "surrealism"],
        "digital painting": ["traditional painting"],
    },
    "contrasting": {
        # Styles with opposing characteristics
        "minimalism": ["maximalism", "baroque", "rococo"],
        "realism": ["abstract", "expressionism", "surrealism"],
        "classical": ["modern", "avant-garde", "primitive"],
        "pop art": ["abstract expressionism", "high art"],
    },
    "thematic_connections": {
        # Styles connected by themes or subjects
        "cyberpunk": ["dystopian", "sci-fi", "tech noir", "industrial"],
        "fantasy": ["fairy tale", "mythological", "heroic", "magical realism"],
        "horror": ["gothic", "dark art", "macabre", "grotesque"],
        "nature-focused": ["landscape", "botanical", "wildlife", "environmental"],
    },
}

# Style attributes - characteristics that can apply across different styles
STYLE_ATTRIBUTES: Dict[str, Dict[str, Any]] = {
    "visual_qualities": {
        "colorful": {
            "description": "Using a wide range of colors",
            "examples": ["fauvism", "pop art", "psychedelic art", "rainbow palette"],
        },
        "monochromatic": {
            "description": "Using variations of a single color",
            "examples": ["grayscale", "sepia", "blue period", "monotone"],
        },
        "high_contrast": {
            "description": "Strong difference between light and dark elements",
            "examples": ["chiaroscuro", "noir", "high key photography", "silhouette"],
        },
        "detailed": {
            "description": "Containing many small, precise elements",
            "examples": ["hyperrealism", "miniature painting", "engraving", "intricate"],
        },
        "simplified": {
            "description": "Reduced to essential elements",
            "examples": ["minimalism", "flat design", "pictogram", "abstracted"],
        },
    },
    "emotional_tones": {
        "serene": {
            "description": "Calm, peaceful quality",
            "examples": ["Japanese landscape", "minimalism", "tonalism"],
        },
        "dramatic": {
            "description": "Intense, theatrical quality",
            "examples": ["baroque", "romanticism", "expressionism"],
        },
        "whimsical": {
            "description": "Playful, fanciful quality",
            "examples": ["children's illustration", "naive art", "storybook style"],
        },
        "melancholic": {
            "description": "Sad, reflective quality",
            "examples": ["vanitas", "symbolism", "gothic", "dark romanticism"],
        },
        "energetic": {
            "description": "Dynamic, lively quality",
            "examples": ["futurism", "action painting", "street art", "pop art"],
        },
    },
    "technical_approaches": {
        "precise": {
            "description": "Exact, meticulous execution",
            "examples": ["academic art", "hyperrealism", "technical illustration"],
        },
        "loose": {
            "description": "Free, gestural execution",
            "examples": ["expressionism", "gesture drawing", "abstract expressionism"],
        },
        "textured": {
            "description": "Emphasis on tactile surface qualities",
            "examples": ["impasto", "collage", "mixed media", "encaustic"],
        },
        "geometric": {
            "description": "Based on geometric forms and shapes",
            "examples": ["cubism", "constructivism", "hard-edge painting", "op art"],
        },
        "atmospheric": {
            "description": "Emphasis on mood and ambiance",
            "examples": ["sfumato", "tonalism", "color field", "ambient lighting"],
        },
    },
}

# =============================================================================
# Helper Functions
# =============================================================================


def get_style_by_medium(medium: str) -> Dict[str, Any]:
    """Return all styles associated with a specific medium."""
    if medium in STYLE_MEDIUM:
        return STYLE_MEDIUM[medium]
    return {}


def get_style_by_movement(movement: str) -> Dict[str, Any]:
    """Return information about a specific art movement."""
    if movement in STYLE_MOVEMENTS:
        return STYLE_MOVEMENTS[movement]
    return {}


def get_related_styles(style_name: str) -> List[str]:
    """Return a list of styles related to the given style."""
    related = []

    # Check in complementary relationships
    for style, complements in STYLE_RELATIONSHIPS["complementary"].items():
        if style == style_name:
            related.extend(complements)
        elif style_name in complements:
            related.append(style)

    # Check in derivative relationships
    for style, derivatives in STYLE_RELATIONSHIPS["derivative"].items():
        if style == style_name:
            related.extend(derivatives)
        elif style_name in derivatives:
            related.append(style)

    # Check in thematic connections
    for theme, connected_styles in STYLE_RELATIONSHIPS["thematic_connections"].items():
        if style_name in connected_styles:
            related.extend([s for s in connected_styles if s != style_name])

    return list(set(related))  # Remove duplicates


def get_style_attributes(style_name: str) -> List[str]:
    """Return a list of attributes associated with a given style."""
    attributes = []

    # Check through all categories of attributes
    for category, attr_dict in STYLE_ATTRIBUTES.items():
        for attr_name, attr_info in attr_dict.items():
            if style_name in attr_info["examples"]:
                attributes.append(attr_name)

    return attributes


def get_styles_by_attribute(attribute: str) -> List[str]:
    """Return all styles that have a specific attribute."""
    styles = []

    # Search through all attribute categories
    for category, attr_dict in STYLE_ATTRIBUTES.items():
        if attribute in attr_dict:
            styles.extend(attr_dict[attribute]["examples"])

    return styles
