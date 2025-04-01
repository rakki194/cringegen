"""
Utilities for bidirectional conversion between natural language and tags.

This module provides functions to:
1. Convert a set of tags to natural language descriptions
2. Parse natural language to extract relevant tags
"""

import re
from typing import Dict, List, Optional, Set, Tuple

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)

# Tag categories and their natural language representations
TAG_CATEGORIES = {
    "subject": ["character", "person", "creature", "animal", "subject"],
    "species": ["species", "breed", "race", "type"],
    "quality": ["quality", "resolution", "detail"],
    "style": ["style", "art style", "artistic style", "medium"],
    "setting": ["location", "place", "setting", "environment", "background"],
    "pose": ["pose", "posture", "stance", "position"],
    "action": ["action", "activity", "doing", "movement"],
    "expression": ["expression", "emotion", "feeling", "mood"],
    "clothing": ["clothing", "outfit", "attire", "dress", "wear", "wearing"],
    "color": ["color", "coloration", "hue", "tone", "shade"],
    "accessory": ["accessory", "accessorize", "jewelry", "decoration"],
    "anatomical": ["genitalia", "genitals", "anatomical", "anatomy", "body part"],
    "nsfw_rating": ["rating", "content rating", "explicitness", "maturity level"],
    "viewer_interaction": ["viewer interaction", "gaze", "looking", "eye contact"],
}

# Reverse mapping from category words to the category
CATEGORY_MAPPING = {}
for category, words in TAG_CATEGORIES.items():
    for word in words:
        CATEGORY_MAPPING[word] = category

# Common tag prefixes and their natural language equivalents
TAG_PREFIXES = {
    "high quality": "high quality",
    "best quality": "excellent quality",
    "masterpiece": "masterpiece",
    "detailed": "highly detailed",
    "beautiful": "beautiful",
    "intricate": "with intricate details",
    "professional": "professional quality",
    "hires": "high resolution",
    "ultra detailed": "ultra detailed",
    "ultra high res": "ultra high resolution",
}

# Common conjunctions and connecting phrases
CONJUNCTIONS = [
    "and",
    "with",
    "featuring",
    "including",
    "having",
    "wearing",
    "showing",
    "displaying",
    "in",
    "at",
    "on",
]


def tags_to_text(tags: List[str], style: str = "descriptive") -> str:
    """
    Convert a list of tags to natural language text.

    Args:
        tags: List of tags to convert
        style: Text style ('descriptive', 'concise', 'detailed')

    Returns:
        Natural language description of the tags
    """
    if not tags:
        return ""

    # Categorize tags
    categorized_tags = categorize_tags(tags)

    # Generate text based on style
    if style == "concise":
        return generate_concise_text(categorized_tags)
    elif style == "detailed":
        return generate_detailed_text(categorized_tags)
    else:  # descriptive (default)
        return generate_descriptive_text(categorized_tags)


def text_to_tags(text: str) -> List[str]:
    """
    Convert natural language text to a list of tags.

    Args:
        text: Natural language text to convert

    Returns:
        List of extracted tags
    """
    if not text:
        return []

    # Tokenize and normalize text
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)

    # Extract potential tags
    potential_tags = extract_potential_tags(text, pos_tags)

    # Clean and normalize tags
    return normalize_tags(potential_tags)


def categorize_tags(tags: List[str]) -> Dict[str, List[str]]:
    """
    Categorize tags into different categories.

    Args:
        tags: List of tags to categorize

    Returns:
        Dictionary mapping categories to lists of tags
    """
    categories = {
        "subject": [],
        "species": [],
        "quality": [],
        "style": [],
        "setting": [],
        "pose": [],
        "action": [],
        "expression": [],
        "clothing": [],
        "color": [],
        "accessory": [],
        "anatomical": [],  # New category for genitalia and body parts
        "nsfw_rating": [],  # New category for explicit/nsfw ratings
        "viewer_interaction": [],  # New category for viewer interaction
        "other": [],
    }

    # Import species taxonomy data for proper categorization
    from cringegen.data.taxonomy import SPECIES_TAXONOMY
    # Import art styles data
    from cringegen.data.styles import ART_STYLES
    
    # Flatten the art styles dictionary for easier lookup
    all_art_styles = []
    for style_category, styles in ART_STYLES.items():
        all_art_styles.extend(styles)
    
    # Common clothing items that might not be in the basic list
    additional_clothing = [
        "robe", "robes", "jeans", "shorts", "hoodie", "sweatshirt", "sweater", 
        "t-shirt", "tshirt", "blouse", "tank top", "bikini", "swimsuit",
        "costume", "armor", "attire", "garment", "lingerie", "panties", "bra",
        "underwear", "boxers", "briefs", "socks", "stockings", "leggings",
        "tights", "pantyhose", "kilt", "skort", "camisole"
    ]
    
    # Common accessories
    accessories = [
        "necklace", "pendant", "ring", "bracelet", "earring", "crown", "tiara",
        "glasses", "sunglasses", "monocle", "watch", "wristband", "anklet",
        "choker", "collar", "piercing", "tattoo", "mask", "scarf", "bandana",
        "headband", "ribbon", "bow", "hairpin", "barrette", "brooch", "badge",
        "pin", "button", "patch", "emblem", "sword", "dagger", "knife", "staff",
        "wand", "gun", "pistol", "rifle", "bow", "arrow", "shield", "backpack",
        "bag", "purse", "satchel", "briefcase", "luggage", "pouch", "wallet",
        "phone", "smartphone", "laptop", "tablet", "camera", "headphones", "earbuds",
        "glowing staff", "magical staff", "robotic arm", "cybernetic implant"
    ]
    
    # Common expressions
    expressions = [
        "happy", "sad", "angry", "excited", "surprised", "shocked", "afraid",
        "scared", "terrified", "worried", "anxious", "nervous", "confused",
        "puzzled", "thoughtful", "pensive", "bored", "tired", "sleepy",
        "exhausted", "relaxed", "calm", "peaceful", "serene", "content",
        "satisfied", "proud", "smug", "cocky", "arrogant", "shy", "embarrassed",
        "ashamed", "guilty", "innocent", "hopeful", "desperate", "determined",
        "confident", "uncertain", "doubtful", "suspicious", "trusting", "loving",
        "hateful", "jealous", "envious", "disgusted", "amused", "entertained",
        "interested", "bored", "indifferent", "neutral", "blank", "empty",
        "grinning", "smiling", "frowning", "grimacing", "squinting", "winking",
        "blinking", "laughing", "crying", "sobbing", "screaming", "yelling",
        "whispering", "shouting", "sighing", "gasping", "panting", "breathing",
        "sneezing", "coughing", "yawning", "snoring", "sleeping", "dreaming",
        "thinking", "wondering", "contemplating", "meditating", "concentrating",
        "studying", "observing", "watching", "staring", "glaring", "glancing",
        "peering", "peeking", "squinting", "blinking", "closing eyes"
    ]
    
    # Anthro-specific terms
    anthro_terms = [
        "anthro", "anthropomorphic", "furry", "kemono", "fursona", "anthro_", "anthromorphic"
    ]
    
    # List of art styles to prevent them from being classified as actions
    art_style_exact_matches = [
        "oil painting", "watercolor", "acrylic painting", "digital painting", 
        "ink drawing", "pencil drawing", "charcoal drawing", "pastel drawing"
    ]
    
    # List of settings to ensure proper categorization
    setting_exact_matches = [
        "ancient library", "ruined temple", "modern office", "cozy bedroom",
        "dark alley", "spaceship interior", "medieval tavern", "fantasy village",
        "sci-fi laboratory", "abandoned warehouse", "hidden cave", "underground bunker",
        "floating island", "crystal cave", "bustling marketplace", "throne room",
        "space station", "cyberpunk city", "futuristic metropolis", "desert oasis",
        "mountain peak", "forest clearing", "jungle temple", "arctic research station"
    ]
    
    # List of physical features that shouldn't be classified as settings
    physical_features = [
        "battle scars", "scars", "marks", "tattoos", "birthmarks", "freckles",
        "spots", "stripes", "patterns", "markings", "mane", "long ears", "short ears",
        "long tail", "short tail", "bushy tail", "large eyes", "small eyes", "long hair",
        "short hair", "curly hair", "straight hair", "muscular", "slim", "slender",
        "fit", "athletic", "buff", "toned", "heavy set", "chubby", "big", "small",
        "tall", "short", "average height", "petite", "thicc", "thick"
    ]
    
    # NEW: Anatomical terms for proper categorization
    anatomical_terms = [
        "penis", "cock", "dick", "member", "phallus", "shaft", "testicles", "balls", "testes",
        "genitalia", "genitals", "genital", "sheath", "knot", "knotted", "flared", 
        "canine penis", "equine penis", "horse cock", "barbed penis", "knotted penis",
        "vagina", "pussy", "vulva", "cunt", "slit", "cloaca", "vent",
        "animal pussy", "canine pussy", "feline pussy", "equine pussy",
        "anus", "butthole", "tailhole", "rear", "rump", "butt", "behind",
        "breasts", "tits", "boobs", "chest", "mammaries", "nipples", "mammary",
        "canine genitalia", "equine genitalia", "feline genitalia", "reptile genitalia", 
        "avian genitalia", "dragon genitalia", "shark genitalia", "dolphin genitalia",
        "hemipenes", "genital slit", "claspers", "ovipositor",
        "bulge", "erection", "aroused", "arousal", "erect", "stiff", "hard", "wet",
        "cum", "semen", "ejaculate", "precum", "pre", "seed", "sperm",
        "fluids", "juices", "lubricant", "slick", "moist", "dripping",
        "balls deep", "penetration", "insertion", "breeding", "mating", "copulation",
        "masturbation", "jerking off", "masturbating", "pleasuring", "fingering",
        "orgasm", "climax", "cumming", "squirting", "ejaculating"
    ]
    
    # NEW: Presenting pose terms
    presenting_poses = [
        "presenting", "presenting hindquarters", "presenting sheath", "presenting genitals",
        "presenting pussy", "presenting anus", "presenting ass", "presenting rear",
        "spread legs", "spread pussy", "spread anus", "spread cheeks", "spread buttocks",
        "legs spread", "legs up", "ass up", "butt up", "face down ass up", "fdau",
        "on back", "on all fours", "doggy style", "doggy position", "mating position",
        "mating press", "breeding position", "breeding stance", "mounting position",
        "rear view", "from behind", "from the back", "butt view", "ass view",
        "bent over", "bending over", "on knees", "kneeling", "prone", "prone bone"
    ]
    
    # NEW: Viewer interaction terms
    viewer_interactions = [
        "looking at viewer", "looking back at viewer", "eye contact", "direct gaze",
        "facing viewer", "facing camera", "towards viewer", "towards camera",
        "watching viewer", "observing viewer", "staring at viewer", "glancing at viewer",
        "inviting gaze", "seductive look", "teasing viewer", "teasing look",
        "flirtatious gaze", "flirting with viewer", "beckoning viewer", "beckoning",
        "third person view", "pov", "point of view", "first person view"
    ]
    
    # NEW: NSFW rating terms
    nsfw_ratings = [
        "nsfw", "explicit", "adult", "mature", "18+", "adult content", "mature content",
        "rating:explicit", "rating:questionable", "rating:mature", "rating:adult",
        "questionable", "questionable content", "pornographic", "porn", "erotic", "lewd",
        "xxx", "r-18", "r18", "r-rated", "x-rated", "nc-17", "not safe for work"
    ]
    
    for tag in tags:
        tag_lower = tag.lower().strip()
        tag_words = tag_lower.split()
        
        # NEW: Check for NSFW ratings first
        if any(rating in tag_lower for rating in nsfw_ratings):
            categories["nsfw_rating"].append(tag)
            continue
            
        # NEW: Check for anatomical terms
        if any(term in tag_lower for term in anatomical_terms):
            categories["anatomical"].append(tag)
            continue
            
        # NEW: Check for presenting poses
        if any(pose in tag_lower for pose in presenting_poses):
            categories["pose"].append(tag)
            continue
            
        # NEW: Check for viewer interactions
        if any(interaction in tag_lower for term in viewer_interactions for interaction in [term.lower()]):
            categories["viewer_interaction"].append(tag)
            continue
        
        # Check for exact matches first
        if tag_lower in art_style_exact_matches:
            categories["style"].append(tag)
            continue
            
        if tag_lower in setting_exact_matches:
            categories["setting"].append(tag)
            continue
        
        if tag_lower in additional_clothing or "top" in tag_lower or "shirt" in tag_lower:
            categories["clothing"].append(tag)
            continue
            
        if tag_lower in physical_features or tag_lower in ["battle scars", "battle scarred"]:
            # Features like "battle scars" or "long ears" go to other, not setting
            if "scars" in tag_lower or "scarred" in tag_lower:
                categories["other"].append(tag)
            else:
                categories["other"].append(tag)
            continue
            
        # Check for anthro species combinations (e.g., "anthro fox", "female anthro wolf")
        is_anthro = any(anthro in tag_lower for anthro in anthro_terms)
        contains_species = any(species in tag_lower for species in SPECIES_TAXONOMY)
        
        # Handle anthro species specifically
        if is_anthro and contains_species:
            # Extract the species part
            for species in SPECIES_TAXONOMY:
                if species in tag_lower:
                    # Add to both subject and species
                    categories["subject"].append(tag)
                    if species not in categories["species"]:
                        categories["species"].append(species)
                    break
            continue
            
        # Check if tag is a known species
        is_species = tag_lower in SPECIES_TAXONOMY
        
        # Check if tag contains color + species (e.g., "red fox", "blue dragon")
        colors = ["red", "blue", "green", "yellow", "purple", "black", "white", 
                 "orange", "brown", "pink", "gray", "grey", "cyan", "magenta", 
                 "teal", "silver", "gold", "bronze", "copper", "ruby", "emerald", 
                 "sapphire", "amber", "turquoise", "indigo", "violet", "crimson", 
                 "scarlet", "navy", "maroon", "olive", "ivory", "cream", "beige", 
                 "tan", "chocolate", "coffee", "slate", "charcoal", "ash", "ebony", 
                 "onyx", "obsidian", "jet", "raven", "midnight", "aqua", "azure", 
                 "cerulean", "cobalt", "lime", "mint", "peach", "coral", "salmon", 
                 "lavender", "lilac", "plum", "mauve", "fuchsia", "rose", "rust", 
                 "cinnamon", "caramel", "honey", "lemon", "vanilla", "snow"]
        
        is_color_species = False
        if len(tag_words) > 1:
            last_word = tag_words[-1]
            if last_word in SPECIES_TAXONOMY:
                color_part = " ".join(tag_words[:-1])
                if any(color in color_part for color in colors):
                    # It's a color + species combination
                    is_color_species = True
                    categories["color"].append(color_part)
                    # Add to both subject and species
                    categories["subject"].append(last_word)
                    categories["species"].append(last_word)
                    continue
        
        # Check if tag is a known art style or contains art style terms
        style_terms = ["style", "art", "painting", "illustration", "drawing", "sketch", "render", 
                      "artwork", "aesthetic", "concept", "visual", "graphic", "design"]
                      
        is_art_style = tag_lower in all_art_styles
        contains_art_style = any(term in tag_lower for term in style_terms)
        digital_art_terms = ["digital", "cgi", "3d", "rendered", "computer generated", "photo manipulation"]
        is_digital_art = any(term in tag_lower for term in digital_art_terms)
        
        # Check for popular art styles not in the predefined list
        additional_styles = ["anime", "manga", "cartoon", "comic", "pixel art", "8-bit", "16-bit",
                            "cel shaded", "chibi", "realistic", "photorealistic", "hyperrealistic",
                            "surrealistic", "impressionist", "expressionist", "cubist", "minimalist",
                            "abstract", "fantasy", "sci-fi", "cyberpunk", "steampunk", "dieselpunk",
                            "biopunk", "solarpunk", "atompunk", "vaporwave", "retrowave", "synthwave"]
        
        is_additional_style = any(style in tag_lower for style in additional_styles)

        # Check quality tags first
        if any(prefix in tag_lower for prefix in TAG_PREFIXES):
            categories["quality"].append(tag)
            continue

        # Check for color descriptors with special handling for "X fur" patterns
        if "fur" in tag_lower or "skin" in tag_lower or "scales" in tag_lower or "feathers" in tag_lower:
            if any(color in tag_lower for color in colors):
                categories["color"].append(tag)
                continue
                
        # General color check
        if any(color in tag_lower for color in colors) and "style" not in tag_lower:
            categories["color"].append(tag)
            continue

        # Check for clothing items - expanded list
        if any(item in tag_lower for item in ["shirt", "pants", "dress", "skirt", "hat", "jacket", 
                                             "coat", "shoes", "boots", "gloves", "scarf", "outfit", 
                                             "uniform", "clothes", "suit", "tie", "belt", "socks"]) or \
           any(item in tag_lower for item in additional_clothing):
            categories["clothing"].append(tag)
            continue
        
        # Check for accessories
        if any(item in tag_lower for item in accessories):
            categories["accessory"].append(tag)
            continue
            
        # Check for expressions
        if any(expr in tag_lower for expr in expressions) or "expression" in tag_lower:
            categories["expression"].append(tag)
            continue

        # Check for setting/location - expand this list
        setting_terms = [
            "room", "forest", "beach", "city", "mountain", "field", "house", "building", 
            "street", "park", "lake", "river", "ocean", "sky", "space", "indoor", "outdoor", 
            "landscape", "scenery", "background", "castle", "ruins", "temple", "church", 
            "cathedral", "mosque", "shrine", "palace", "mansion", "cottage", "cabin", 
            "apartment", "office", "school", "university", "hospital", "restaurant", "café", 
            "bar", "club", "shop", "store", "mall", "market", "bazaar", "plaza", "square", 
            "garden", "park", "meadow", "prairie", "savanna", "desert", "tundra", "arctic", 
            "jungle", "rainforest", "swamp", "marsh", "bog", "cave", "cavern", "grotto", 
            "waterfall", "stream", "brook", "creek", "pond", "pool", "spring", "oasis", 
            "coast", "shore", "cliff", "canyon", "gorge", "valley", "hill", "volcano", 
            "island", "peninsula", "cape", "bay", "gulf", "strait", "channel", "dock", 
            "harbor", "port", "pier", "bridge", "dam", "lighthouse", "tower", "skyscraper", 
            "monument", "statue", "fountain", "arch", "gate", "wall", "fence", "path", 
            "trail", "road", "highway", "railway", "station", "airport", "spaceport", 
            "vehicle", "car", "truck", "train", "subway", "trolley", "tram", "bus", 
            "bicycle", "motorcycle", "boat", "ship", "yacht", "submarine", "aircraft", 
            "airplane", "helicopter", "rocket", "spaceship", "ufo", "satellite"
        ]
        
        if any(item in tag_lower for item in setting_terms) or "background" in tag_lower:
            categories["setting"].append(tag)
            continue
            
        # Check for art style - expanded check
        if is_art_style or is_digital_art or contains_art_style or is_additional_style:
            categories["style"].append(tag)
            # Don't continue here - some tags might be both style and subject
        
        # Check for species - add to both subject and species if it's a primary subject
        if is_species:
            categories["species"].append(tag)
            # If it's just the species name (like "fox"), it's also a subject
            if len(tag_lower.split()) == 1:
                categories["subject"].append(tag)
            continue
            
        # Check for action words
        action_words = ["running", "walking", "jumping", "flying", "swimming", "climbing", 
                       "sitting", "standing", "lying", "sleeping", "fighting", "battling", 
                       "dancing", "singing", "playing", "reading", "writing", "drawing", 
                       "painting", "eating", "drinking", "cooking", "baking", "cleaning", 
                       "washing", "drying", "ironing", "sewing", "knitting", "crocheting", 
                       "hunting", "fishing", "gardening", "farming", "harvesting", "planting", 
                       "gathering", "collecting", "searching", "finding", "seeking", "looking", 
                       "watching", "observing", "studying", "learning", "teaching", "working", 
                       "resting", "relaxing", "meditating", "exercising", "training", "competing",
                       "looking at viewer", "looking back at viewer"]  # Added viewer interaction terms
                       
        if any(word in tag_lower for word in action_words) and tag_lower not in art_style_exact_matches:
            categories["action"].append(tag)
            continue
            
        # Check for pose descriptors
        pose_words = ["pose", "posing", "stance", "position", "posture", "kneeling", "crouching", 
                     "squatting", "bending", "leaning", "stretching", "flexing", "slouching", 
                     "reclining", "prone", "supine", "upright", "bent", "twisted", "arched"]
                     
        if any(word in tag_lower for word in pose_words):
            categories["pose"].append(tag)
            continue
            
        # If we haven't continued yet, and the tag wasn't added to any category,
        # add it to the "other" category
        if tag not in [item for sublist in categories.values() for item in sublist]:
            categories["other"].append(tag)

    return categories


def generate_concise_text(categorized_tags: Dict[str, List[str]]) -> str:
    """
    Generate concise natural language text from categorized tags.

    Args:
        categorized_tags: Dictionary mapping categories to lists of tags

    Returns:
        Concise natural language description
    """
    parts = []

    # Add quality descriptors if any
    if categorized_tags["quality"]:
        parts.append(categorized_tags["quality"][0])  # Just use the first quality descriptor for conciseness

    # Add subject/species
    if categorized_tags["subject"]:
        parts.append(categorized_tags["subject"][0])  # Just use the first subject
    elif categorized_tags["species"]:
        parts.append(categorized_tags["species"][0])  # Just use the first species if no subject

    # Add action or pose if any
    if categorized_tags["action"]:
        parts.append(categorized_tags["action"][0])  # Just use the first action
    elif categorized_tags["pose"]:
        parts.append(categorized_tags["pose"][0])  # Just use the first pose if no action

    # Add setting if any
    if categorized_tags["setting"]:
        parts.append(f"in {categorized_tags['setting'][0]}")

    # Add style if any
    if categorized_tags["style"]:
        parts.append(f"in {categorized_tags['style'][0]} style")

    # Add NSFW content if any
    if categorized_tags["nsfw_rating"]:
        parts.append("NSFW")
        
    # Add anatomical mention if any
    if categorized_tags["anatomical"]:
        parts.append("explicit")
        
    # Add viewer interaction if any  
    if categorized_tags["viewer_interaction"]:
        parts.append(categorized_tags["viewer_interaction"][0])

    # Join all parts
    return ", ".join(parts)


def generate_descriptive_text(categorized_tags: Dict[str, List[str]]) -> str:
    """
    Generate descriptive natural language text from categorized tags.

    Args:
        categorized_tags: Dictionary mapping categories to lists of tags

    Returns:
        Descriptive natural language description
    """
    # Start with quality descriptors
    quality_desc = ""
    if categorized_tags["quality"]:
        quality_desc = " ".join(categorized_tags["quality"]) + " "

    # Build subject description
    subject_desc = ""
    if categorized_tags["subject"]:
        subject_desc = categorized_tags["subject"][0]  # Take only the first subject to avoid repetition
    elif categorized_tags["species"]:
        subject_desc = categorized_tags["species"][0]  # Take only the first species if no subject
    else:
        subject_desc = "subject"

    # Add features
    features = []

    # Add colors
    if categorized_tags["color"]:
        if len(categorized_tags["color"]) == 1:
            features.append("with " + categorized_tags["color"][0])
        else:
            features.append("with " + " and ".join(categorized_tags["color"]))

    # Add clothing
    if categorized_tags["clothing"]:
        if len(categorized_tags["clothing"]) == 1:
            features.append("wearing " + categorized_tags["clothing"][0])
        else:
            features.append("wearing " + " and ".join(categorized_tags["clothing"]))

    # Add accessories
    if categorized_tags["accessory"]:
        if len(categorized_tags["accessory"]) == 1:
            features.append("with " + categorized_tags["accessory"][0])
        else:
            features.append("with " + " and ".join(categorized_tags["accessory"]))

    # Add pose/action
    if categorized_tags["pose"] or categorized_tags["action"]:
        pose_action = []
        if categorized_tags["pose"]:
            pose_action.extend(categorized_tags["pose"])
        if categorized_tags["action"]:
            pose_action.extend(categorized_tags["action"])
        
        if len(pose_action) == 1:
            features.append(pose_action[0])
        else:
            features.append(" and ".join(pose_action))

    # Add expression
    if categorized_tags["expression"]:
        if len(categorized_tags["expression"]) == 1:
            features.append("with a " + categorized_tags["expression"][0] + " expression")
        else:
            features.append("with a " + " and ".join(categorized_tags["expression"]) + " expression")

    # Add setting/location
    if categorized_tags["setting"]:
        if len(categorized_tags["setting"]) == 1:
            features.append("in a " + categorized_tags["setting"][0])
        else:
            features.append("in " + " and ".join(categorized_tags["setting"]))

    # Add style
    if categorized_tags["style"]:
        if len(categorized_tags["style"]) == 1:
            features.append("in " + categorized_tags["style"][0] + " style")
        else:
            features.append("in " + " and ".join(categorized_tags["style"]) + " style")
        
    # Add anatomical features
    if categorized_tags["anatomical"]:
        if len(categorized_tags["anatomical"]) == 1:
            features.append("showing " + categorized_tags["anatomical"][0])
        else:
            features.append("showing " + " and ".join(categorized_tags["anatomical"]))
        
    # Add viewer interaction
    if categorized_tags["viewer_interaction"]:
        features.append(categorized_tags["viewer_interaction"][0])  # Usually just one interaction
        
    # Add NSFW rating
    if categorized_tags["nsfw_rating"]:
        features.append(categorized_tags["nsfw_rating"][0])  # Usually just one rating

    # Add other tags
    if categorized_tags["other"]:
        for tag in categorized_tags["other"]:
            features.append(tag)

    # Combine all parts
    if features:
        return f"A {quality_desc}{subject_desc} {', '.join(features)}"
    else:
        return f"A {quality_desc}{subject_desc}"


def generate_detailed_text(categorized_tags: Dict[str, List[str]]) -> str:
    """
    Generate detailed natural language text from categorized tags.

    Args:
        categorized_tags: Dictionary mapping categories to lists of tags

    Returns:
        Detailed natural language description
    """
    # Start with an opening sentence
    sentences = []

    # Quality and subject
    quality_terms = []
    if categorized_tags["quality"]:
        for quality in categorized_tags["quality"]:
            if quality.lower() in ["masterpiece", "best quality", "high quality"]:
                quality_terms.append(f"{quality}")
    
    quality_text = " ".join(quality_terms) if quality_terms else ""
    
    # Handle subject and species intelligently
    subject = ""
    if categorized_tags["subject"]:
        subject = categorized_tags["subject"][0]  # Take only the first subject
    elif categorized_tags["species"]:
        subject = categorized_tags["species"][0]  # Take only the first species if no subject
    else:
        subject = "character"

    # Add style information to opening
    style_text = ""
    if categorized_tags["style"]:
        style = categorized_tags["style"][0]  # Take only the first style
        style_text = f" in {style} style"

    # Color information
    color_text = ""
    if categorized_tags["color"]:
        colors = " and ".join(categorized_tags["color"])
        color_text = f" with {colors}"
    
    # Build the opening sentence
    if quality_text:
        opening = f"A {quality_text} artwork depicting a {subject}{color_text}{style_text}."
    else:
        opening = f"A detailed artwork of a {subject}{color_text}{style_text}."
        
    sentences.append(opening)

    # Add appearance details
    appearance_details = []

    # Add clothing
    if categorized_tags["clothing"]:
        clothing = " and ".join(categorized_tags["clothing"])
        appearance_details.append(f"The character is wearing {clothing}.")

    # Add accessories
    if categorized_tags["accessory"]:
        accessories = " and ".join(categorized_tags["accessory"])
        appearance_details.append(f"They have {accessories}.")

    # Add pose/action
    if categorized_tags["pose"] or categorized_tags["action"]:
        pose_action = []
        if categorized_tags["pose"]:
            pose_action.extend(categorized_tags["pose"])
        if categorized_tags["action"]:
            pose_action.extend(categorized_tags["action"])
        appearance_details.append(f"They are {' and '.join(pose_action)}.")

    # Add expression
    if categorized_tags["expression"]:
        expression = " and ".join(categorized_tags["expression"])
        appearance_details.append(f"Their expression is {expression}.")

    # Add setting/location
    if categorized_tags["setting"]:
        setting = " and ".join(categorized_tags["setting"])
        appearance_details.append(f"The scene takes place in {setting}.")
        
    # Add anatomical features with appropriate phrasing
    if categorized_tags["anatomical"]:
        if "nsfw_rating" in categorized_tags and categorized_tags["nsfw_rating"]:
            anatomical = " and ".join(categorized_tags["anatomical"])
            appearance_details.append(f"The image explicitly shows {anatomical}.")
        else:
            anatomical = " and ".join(categorized_tags["anatomical"])
            appearance_details.append(f"Anatomical features are visible.")
        
    # Add viewer interaction with natural phrasing
    if categorized_tags["viewer_interaction"]:
        interaction = categorized_tags["viewer_interaction"][0]
        if "looking" in interaction.lower():
            appearance_details.append(f"The character is {interaction}.")
        elif "eye contact" in interaction.lower():
            appearance_details.append(f"The character makes eye contact with the viewer.")
        else:
            appearance_details.append(f"There is {interaction} with the viewer.")
        
    # Add NSFW rating if present
    if categorized_tags["nsfw_rating"]:
        rating = categorized_tags["nsfw_rating"][0]
        if "explicit" in rating.lower():
            appearance_details.append(f"This is explicit adult content.")
        elif "nsfw" in rating.lower():
            appearance_details.append(f"This artwork contains NSFW elements.")
        elif "questionable" in rating.lower():
            appearance_details.append(f"This artwork has questionable content.")
        else:
            appearance_details.append(f"This has mature content.")

    # Add other details
    if categorized_tags["other"]:
        other_parts = []
        for detail in categorized_tags["other"]:
            if detail.lower() in ["muscular", "slim", "athletic", "fit"]:
                other_parts.append(f"The character has a {detail} build.")
            elif detail.lower() in ["evening", "morning", "night", "noon", "dawn", "dusk"]:
                other_parts.append(f"The time is {detail}.")
            elif detail.lower() in ["soft lighting", "harsh lighting", "dramatic lighting", "warm lighting"]:
                other_parts.append(f"The scene has {detail}.")
            else:
                other_parts.append(detail)
        
        if other_parts:
            appearance_details.extend(other_parts)

    # Add appearance details as sentences
    sentences.extend(appearance_details)

    # Join all sentences
    return " ".join(sentences)


def extract_potential_tags(text: str, pos_tags: List[Tuple[str, str]]) -> List[str]:
    """
    Extract potential tags from tokenized and POS-tagged text.

    Args:
        text: Original text
        pos_tags: List of (token, POS tag) tuples

    Returns:
        List of potential tags
    """
    potential_tags = []

    # Extract known quality descriptors
    for prefix, _ in TAG_PREFIXES.items():
        if prefix in text.lower():
            potential_tags.append(prefix)

    # Extract noun phrases (simple approach - can be enhanced)
    i = 0
    while i < len(pos_tags):
        # Check for adjective + noun sequence
        if (
            i + 1 < len(pos_tags)
            and pos_tags[i][1].startswith("JJ")
            and pos_tags[i + 1][1].startswith("NN")
        ):
            phrase = f"{pos_tags[i][0]} {pos_tags[i+1][0]}"
            potential_tags.append(phrase)
            i += 2
        # Check for single nouns
        elif pos_tags[i][1].startswith("NN"):
            potential_tags.append(pos_tags[i][0])
            i += 1
        # Check for verbs (actions)
        elif pos_tags[i][1].startswith("VB"):
            potential_tags.append(pos_tags[i][0])
            i += 1
        else:
            i += 1

    # Extract color terms
    color_terms = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "black",
        "white",
        "orange",
        "brown",
        "pink",
        "gray",
        "grey",
        "cyan",
        "magenta",
        "teal",
    ]

    for color in color_terms:
        if color in text.lower():
            # Look for compound color descriptions
            color_pattern = rf"\b{color}\s+\w+\b"
            matches = re.findall(color_pattern, text.lower())

            if matches:
                potential_tags.extend(matches)
            else:
                potential_tags.append(color)

    return potential_tags


def normalize_tags(tags: List[str]) -> List[str]:
    """
    Clean and normalize tags.

    Args:
        tags: List of potential tags

    Returns:
        Cleaned and normalized tags
    """
    normalized = []
    seen = set()

    for tag in tags:
        # Clean the tag
        clean_tag = tag.strip().lower()

        # Skip empty or single-character tags
        if not clean_tag or len(clean_tag) < 2:
            continue

        # Skip common stop words when they appear alone
        if clean_tag in ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for"]:
            continue

        # Skip if already seen (case-insensitive)
        if clean_tag in seen:
            continue

        # Add to normalized list and mark as seen
        normalized.append(clean_tag)
        seen.add(clean_tag)

    return normalized


def natural_tags_to_text(tags: List[str]) -> str:
    """
    Convert a list of tags to a natural-sounding, fluent caption.

    This function produces high-quality natural language descriptions,
    with special handling for furry/anthro characters, anime styles,
    and other image generation specific content.

    Args:
        tags: List of tags to convert

    Returns:
        Natural language caption in complete sentences
    """
    if not tags:
        return ""

    # Categorize tags first
    categorized_tags = categorize_tags(tags)

    # Extract key components
    quality_terms = categorized_tags.get("quality", [])

    # Check for anthro/furry keywords in subject or other categories
    anthro_keywords = ["anthro", "anthropomorphic", "furry", "fursona", "kemono"]
    anthro_found = any(any(keyword in term.lower() for keyword in anthro_keywords) 
                    for term in categorized_tags.get("subject", []) + categorized_tags.get("other", []))

    # Extract subject and species terms
    subject_terms = categorized_tags.get("subject", [])
    species_terms = categorized_tags.get("species", [])

    # Process other components
    color_terms = categorized_tags.get("color", [])
    setting_terms = categorized_tags.get("setting", [])
    style_terms = categorized_tags.get("style", [])
    
    # Determine the main subject
    main_subject = ""
    if subject_terms:
        main_subject = subject_terms[0]  # Use the first subject term
    elif species_terms:
        if anthro_found:
            main_subject = f"anthro {species_terms[0]}"
        else:
            main_subject = species_terms[0]
    else:
        main_subject = "character"

    # Handle article properly
    starts_with_vowel = main_subject.lower()[0] in "aeiou"
    article = "An" if starts_with_vowel else "A"

    # Transform quality terms
    quality_adj = []
    for term in quality_terms:
        term_lower = term.lower()
        if term_lower == "masterpiece":
            quality_adj.append("masterfully crafted")
        elif term_lower == "high quality":
            quality_adj.append("high-quality")
        elif term_lower == "best quality":
            quality_adj.append("exceptional")
        elif term_lower == "detailed" or term_lower == "highly detailed":
            quality_adj.append("highly detailed")
        elif term_lower == "ultra detailed":
            quality_adj.append("incredibly detailed")
        else:
            quality_adj.append(term_lower)
    
    # Join quality adjectives
    quality_text = ""
    if quality_adj:
        if len(quality_adj) == 1:
            quality_text = quality_adj[0]
        elif len(quality_adj) == 2:
            quality_text = f"{quality_adj[0]} and {quality_adj[1]}"
        else:
            quality_text = ", ".join(quality_adj[:-1]) + f", and {quality_adj[-1]}"

    # Process color information
    color_desc = ""
    if color_terms:
        # Handle special color patterns
        processed_colors = []
        for color in color_terms:
            color_lower = color.lower()
            if "fur" in color_lower:
                processed_colors.append(f"{color.replace('fur', '').strip()}-furred")
            elif "eyes" in color_lower:
                processed_colors.append(f"{color.replace('eyes', '').strip()}-eyed")
            elif "hair" in color_lower:
                processed_colors.append(f"{color.replace('hair', '').strip()}-haired")
            else:
                processed_colors.append(color)
                
        if processed_colors:
            if len(processed_colors) == 1:
                color_desc = f"{processed_colors[0]} "
            else:
                color_desc = f"{' and '.join(processed_colors)} "

    # Process style information
    style_desc = ""
    if style_terms:
        style = style_terms[0]  # Just use the first style for clarity
        if "style" not in style.lower() and not any(x in style.lower() for x in ["art", "painting", "drawing"]):
            style_desc = f" in {style} style"
        else:
            style_desc = f" in {style}"

    # Process setting information
    setting_desc = ""
    if setting_terms:
        if len(setting_terms) == 1:
            setting_desc = f" in a {setting_terms[0]}"
        else:
            setting_desc = f" in a {' and '.join(setting_terms)}"

    # Process clothing
    clothing_desc = ""
    if categorized_tags.get("clothing", []):
        clothing = categorized_tags["clothing"]
        if len(clothing) == 1:
            clothing_desc = f", wearing {clothing[0]}"
        else:
            clothing_desc = f", wearing {' and '.join(clothing)}"

    # Process expression
    expression_desc = ""
    if categorized_tags.get("expression", []):
        expression = categorized_tags["expression"]
        if len(expression) == 1:
            expression_desc = f", with a {expression[0]} expression"
        else:
            expression_desc = f", with a {' and '.join(expression)} expression"

    # Process pose/action
    pose_action_desc = ""
    pose_action_terms = []
    if categorized_tags.get("pose", []):
        pose_action_terms.extend(categorized_tags["pose"])
    if categorized_tags.get("action", []):
        pose_action_terms.extend(categorized_tags["action"])
    
    if pose_action_terms:
        if len(pose_action_terms) == 1:
            pose_action_desc = f", {pose_action_terms[0]}"
        else:
            pose_action_desc = f", {' and '.join(pose_action_terms)}"

    # Process viewer interaction
    viewer_desc = ""
    if categorized_tags.get("viewer_interaction", []):
        interaction = categorized_tags["viewer_interaction"][0]  # Usually just one
        viewer_desc = f", {interaction}"

    # Process anatomical features
    anatomical_desc = ""
    if categorized_tags.get("anatomical", []):
        anatomical_terms = categorized_tags["anatomical"]
        if categorized_tags.get("nsfw_rating", []):  # If NSFW
            if len(anatomical_terms) == 1:
                anatomical_desc = f", showing {anatomical_terms[0]}"
            else:
                anatomical_desc = f", showing {', '.join(anatomical_terms[:-1])}, and {anatomical_terms[-1]}"
        else:
            if len(anatomical_terms) == 1:
                anatomical_desc = f", with visible {anatomical_terms[0]}"
            else:
                anatomical_desc = f", with visible {', '.join(anatomical_terms[:-1])}, and {anatomical_terms[-1]}"

    # Process other tags
    other_desc = ""
    filtered_other = []
    for term in categorized_tags.get("other", []):
        term_lower = term.lower()
        
        # Skip terms we've likely covered elsewhere
        if term_lower in ["interspecies"] or "lighting" in term_lower:
            filtered_other.append(term)
            
    if filtered_other:
        other_desc = f", {', '.join(filtered_other)}"

    # Add NSFW rating
    nsfw_desc = ""
    if categorized_tags.get("nsfw_rating", []):
        nsfw_desc = ", explicit content" if "explicit" in " ".join(categorized_tags["nsfw_rating"]).lower() else ", nsfw content"

    # Combine all parts into a fluent caption
    caption = f"{article} {quality_text} artwork of a {color_desc}{main_subject}{style_desc}{setting_desc}{clothing_desc}{expression_desc}{pose_action_desc}{viewer_desc}{anatomical_desc}{other_desc}{nsfw_desc}."

    # Clean up formatting issues
    caption = caption.replace("  ", " ")
    caption = caption.replace(" ,", ",")
    caption = caption.replace("a a", "a")
    caption = caption.replace("a an", "an")
    caption = caption.replace("of a anthro", "of an anthro")
    caption = caption.replace("A artwork", "An artwork")
    
    # Final grammar check
    caption = caption.replace("of a a", "of a")
    caption = caption.replace("of a an", "of an")
    
    return caption
