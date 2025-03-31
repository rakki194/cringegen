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
        "other": [],
    }

    for tag in tags:
        tag_lower = tag.lower()

        # Check quality tags first
        if any(prefix in tag_lower for prefix in TAG_PREFIXES):
            categories["quality"].append(tag)
            continue

        # Check for color descriptors
        if any(
            color in tag_lower
            for color in [
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
        ):
            categories["color"].append(tag)
            continue

        # Check for clothing items
        if any(
            item in tag_lower
            for item in [
                "shirt",
                "pants",
                "dress",
                "skirt",
                "hat",
                "jacket",
                "coat",
                "shoes",
                "boots",
                "gloves",
                "scarf",
                "outfit",
                "uniform",
                "clothes",
                "suit",
                "tie",
                "belt",
                "socks",
                "underwear",
            ]
        ):
            categories["clothing"].append(tag)
            continue

        # Check for setting/location
        if any(
            item in tag_lower
            for item in [
                "room",
                "forest",
                "beach",
                "city",
                "mountain",
                "field",
                "house",
                "building",
                "street",
                "park",
                "lake",
                "river",
                "ocean",
                "sky",
                "space",
                "indoor",
                "outdoor",
                "landscape",
                "scenery",
                "background",
            ]
        ):
            categories["setting"].append(tag)
            continue

        # Default to other category
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
    description_parts = []

    # Add quality descriptors
    if categorized_tags["quality"]:
        description_parts.append(", ".join(categorized_tags["quality"]))

    # Add subject/species
    subject_parts = []
    if categorized_tags["subject"]:
        subject_parts.extend(categorized_tags["subject"])
    if categorized_tags["species"]:
        subject_parts.extend(categorized_tags["species"])

    if subject_parts:
        description_parts.append(" ".join(subject_parts))

    # Add actions if any
    if categorized_tags["action"]:
        description_parts.append(" ".join(categorized_tags["action"]))

    # Add setting if any
    if categorized_tags["setting"]:
        location = " ".join(categorized_tags["setting"])
        description_parts.append(f"in {location}")

    # Add style if any
    if categorized_tags["style"]:
        style = ", ".join(categorized_tags["style"])
        description_parts.append(f"in {style} style")

    # Join all parts
    return ", ".join(description_parts)


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
        quality_desc = " ".join(categorized_tags["quality"]) + " of "

    # Build subject description
    subject_desc = ""
    if categorized_tags["subject"]:
        subject_desc = " ".join(categorized_tags["subject"])
    if categorized_tags["species"]:
        if subject_desc:
            subject_desc += " " + " ".join(categorized_tags["species"])
        else:
            subject_desc = " ".join(categorized_tags["species"])

    if not subject_desc:
        subject_desc = "subject"

    # Add features
    features = []

    # Add colors
    if categorized_tags["color"]:
        features.append("with " + " and ".join(categorized_tags["color"]) + " coloration")

    # Add clothing
    if categorized_tags["clothing"]:
        features.append("wearing " + " and ".join(categorized_tags["clothing"]))

    # Add accessories
    if categorized_tags["accessory"]:
        features.append("with " + " and ".join(categorized_tags["accessory"]))

    # Add pose/action
    if categorized_tags["pose"] or categorized_tags["action"]:
        pose_action = []
        if categorized_tags["pose"]:
            pose_action.extend(categorized_tags["pose"])
        if categorized_tags["action"]:
            pose_action.extend(categorized_tags["action"])
        features.append(" ".join(pose_action))

    # Add expression
    if categorized_tags["expression"]:
        features.append("with " + " ".join(categorized_tags["expression"]) + " expression")

    # Add setting/location
    if categorized_tags["setting"]:
        features.append("in " + " ".join(categorized_tags["setting"]))

    # Add style
    if categorized_tags["style"]:
        features.append("in " + " ".join(categorized_tags["style"]) + " style")

    # Add other tags
    if categorized_tags["other"]:
        features.append(", ".join(categorized_tags["other"]))

    # Combine all parts
    if features:
        return f"{quality_desc}{subject_desc} {', '.join(features)}"
    else:
        return f"{quality_desc}{subject_desc}"


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
    quality_terms = " ".join(categorized_tags["quality"]) if categorized_tags["quality"] else "a"
    subject_terms = []
    if categorized_tags["subject"]:
        subject_terms.extend(categorized_tags["subject"])
    if categorized_tags["species"]:
        subject_terms.extend(categorized_tags["species"])
    subject = " ".join(subject_terms) if subject_terms else "subject"

    opening = f"This is {quality_terms} {subject}"

    # Add style information to opening
    if categorized_tags["style"]:
        style = " ".join(categorized_tags["style"])
        opening += f" in {style} style"

    sentences.append(opening + ".")

    # Add appearance details
    appearance_details = []

    # Add colors
    if categorized_tags["color"]:
        colors = " and ".join(categorized_tags["color"])
        appearance_details.append(f"The subject has {colors} coloration")

    # Add clothing
    if categorized_tags["clothing"]:
        clothing = " and ".join(categorized_tags["clothing"])
        appearance_details.append(f"They are wearing {clothing}")

    # Add accessories
    if categorized_tags["accessory"]:
        accessories = " and ".join(categorized_tags["accessory"])
        appearance_details.append(f"They have {accessories}")

    # Add pose/action
    if categorized_tags["pose"] or categorized_tags["action"]:
        pose_action = []
        if categorized_tags["pose"]:
            pose_action.extend(categorized_tags["pose"])
        if categorized_tags["action"]:
            pose_action.extend(categorized_tags["action"])
        appearance_details.append(f"They are {' '.join(pose_action)}")

    # Add expression
    if categorized_tags["expression"]:
        expression = " ".join(categorized_tags["expression"])
        appearance_details.append(f"Their expression is {expression}")

    # Add setting/location
    if categorized_tags["setting"]:
        setting = " ".join(categorized_tags["setting"])
        appearance_details.append(f"The scene is set in {setting}")

    # Add other details
    if categorized_tags["other"]:
        others = ", ".join(categorized_tags["other"])
        appearance_details.append(f"Additional details include {others}")

    # Add appearance details as sentences
    sentences.extend([detail + "." for detail in appearance_details])

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

    # Check for anthro/furry keywords in other categories
    anthro_keywords = ["anthro", "anthropomorphic", "furry", "fursona", "kemono"]
    species_terms = []
    anthro_found = False

    # Check if we have anthro content
    for term in categorized_tags.get("other", []):
        if any(keyword in term.lower() for keyword in anthro_keywords):
            anthro_found = True
            break

    # Extract subject and species terms
    subject_terms = []
    if categorized_tags.get("subject", []):
        subject_terms.extend(categorized_tags["subject"])
    if categorized_tags.get("species", []):
        species_terms.extend(categorized_tags["species"])

    # Special handling for animal terms that might be subjects in furry content
    animal_terms = [
        "fox",
        "wolf",
        "cat",
        "dog",
        "tiger",
        "lion",
        "rabbit",
        "deer",
        "dragon",
        "horse",
        "raccoon",
        "otter",
        "bear",
        "bird",
        "mouse",
    ]

    found_animals = []
    for term in list(categorized_tags.get("other", [])):
        term_lower = term.lower()
        for animal in animal_terms:
            if animal in term_lower:
                found_animals.append(term)
                if term in categorized_tags["other"]:
                    categorized_tags["other"].remove(term)
                break

    # If we have anthro keywords and animals, combine them
    if anthro_found and found_animals:
        for animal in found_animals:
            anthro_term = f"anthro {animal}"
            if anthro_term not in subject_terms:
                subject_terms.append(anthro_term)
    # Otherwise just add the animals to subjects
    elif found_animals:
        for animal in found_animals:
            if animal not in subject_terms:
                subject_terms.append(animal)

    # Special handling for anime character types
    anime_char_types = [
        "neko",
        "kitsune",
        "kemonomimi",
        "catgirl",
        "foxgirl",
        "wolfgirl",
        "cat girl",
        "fox girl",
        "wolf girl",
        "bunny girl",
        "dragon girl",
    ]

    for term in list(categorized_tags.get("other", [])):
        if any(char_type in term.lower() for char_type in anime_char_types):
            subject_terms.append(term)
            if term in categorized_tags["other"]:
                categorized_tags["other"].remove(term)

    # Process other components
    color_terms = categorized_tags.get("color", [])
    setting_terms = categorized_tags.get("setting", [])

    # Special handling for art styles
    style_terms = categorized_tags.get("style", [])
    art_styles = [
        "digital art",
        "oil painting",
        "watercolor",
        "sketch",
        "anime style",
        "manga style",
        "cartoon",
        "realistic",
        "photorealistic",
        "stylized",
        "chibi",
        "pixel art",
        "cel shaded",
        "comic",
        "illustration",
    ]

    for term in list(categorized_tags.get("other", [])):
        if any(style in term.lower() for style in art_styles):
            style_terms.append(term)
            if term in categorized_tags["other"]:
                categorized_tags["other"].remove(term)

    # Create article and adjective phrase for the subject
    article = "A"
    if subject_terms and subject_terms[0][0].lower() in "aeiou":
        article = "An"

    # Build quality adjective string
    quality_adj = ""
    if quality_terms:
        # Transform quality terms into proper adjectives
        adj_mapping = {
            "masterpiece": "masterfully crafted",
            "high quality": "high-quality",
            "best quality": "exceptional",
            "detailed": "highly detailed",
            "ultra detailed": "incredibly detailed",
            "beautiful": "beautiful",
            "professional": "professional",
            "ultra high res": "ultra high-resolution",
            "hires": "high-resolution",
            "intricate": "intricately detailed",
            "4k": "4K resolution",
            "8k": "8K resolution",
        }

        quality_adjectives = []
        for term in quality_terms:
            term_lower = term.lower()
            if term_lower in adj_mapping:
                quality_adjectives.append(adj_mapping[term_lower])
            else:
                quality_adjectives.append(term_lower)

        # Join adjectives properly with commas and "and"
        if len(quality_adjectives) == 1:
            quality_adj = quality_adjectives[0]
        elif len(quality_adjectives) == 2:
            quality_adj = f"{quality_adjectives[0]} and {quality_adjectives[1]}"
        else:
            quality_adj = ", ".join(quality_adjectives[:-1]) + f", and {quality_adjectives[-1]}"

    # Build subject with color attributes
    subject = ""
    if subject_terms:
        # Join the subject terms
        if len(subject_terms) == 1:
            subject = subject_terms[0]
        else:
            subject = " ".join(subject_terms)

        # Add color information
        if color_terms:
            # Process color terms to make them adjectives
            processed_colors = []
            for color in color_terms:
                # Handle "X fur" pattern
                if "fur" in color:
                    color_parts = color.split()
                    if len(color_parts) > 1 and color_parts[-1] == "fur":
                        processed_colors.append(f"{'-'.join(color_parts[:-1])}-furred")
                    else:
                        processed_colors.append(color)
                # Handle "X eyes" pattern
                elif "eyes" in color:
                    color_parts = color.split()
                    if len(color_parts) > 1 and color_parts[-1] == "eyes":
                        processed_colors.append(f"{'-'.join(color_parts[:-1])}-eyed")
                    else:
                        processed_colors.append(color)
                # Handle "X hair" pattern
                elif "hair" in color:
                    color_parts = color.split()
                    if len(color_parts) > 1 and color_parts[-1] == "hair":
                        processed_colors.append(f"{'-'.join(color_parts[:-1])}-haired")
                    else:
                        processed_colors.append(color)
                # Handle other color patterns
                elif " " in color:
                    processed_colors.append(f"{color}")
                else:
                    processed_colors.append(f"{color}")

            if len(processed_colors) == 1:
                subject = f"{processed_colors[0]} {subject}"
            elif len(processed_colors) == 2:
                subject = f"{processed_colors[0]} and {processed_colors[1]} {subject}"
            else:
                subject = (
                    f"{', '.join(processed_colors[:-1])}, and {processed_colors[-1]} {subject}"
                )
    else:
        # Default subject based on presence of anthro keywords
        if anthro_found:
            subject = "anthropomorphic character"
        else:
            subject = "character"

    # Build style phrase
    style_phrase = ""
    if style_terms:
        style_words = []
        for style in style_terms:
            if (
                "art" not in style.lower()
                and "painting" not in style.lower()
                and "render" not in style.lower()
            ):
                style_words.append(f"{style} style")
            else:
                style_words.append(style)

        if len(style_words) == 1:
            style_phrase = style_words[0]
        elif len(style_words) == 2:
            style_phrase = f"{style_words[0]} and {style_words[1]}"
        else:
            style_phrase = ", ".join(style_words[:-1]) + f", and {style_words[-1]}"

    # Build setting phrase
    setting_phrase = ""
    if setting_terms:
        if len(setting_terms) == 1:
            setting = setting_terms[0]
            setting_phrase = f"in a {setting}"
        else:
            setting_list = []
            for setting in setting_terms:
                if "background" in setting:
                    setting = setting.replace("background", "").strip()
                setting_list.append(setting)

            if len(setting_list) == 1:
                setting_phrase = f"in a {setting_list[0]}"
            elif len(setting_list) == 2:
                setting_phrase = f"in a {setting_list[0]} and {setting_list[1]}"
            else:
                setting_phrase = f"in a {', '.join(setting_list[:-1])}, and {setting_list[-1]}"

    # Check for lighting, weather, and time in other tags
    lighting_terms = []
    weather_terms = []
    time_terms = []

    lighting_patterns = [
        "lighting",
        "illuminated",
        "illumination",
        "sunset",
        "sunrise",
        "moonlight",
        "sunlight",
        "dynamic lighting",
        "soft lighting",
        "dramatic lighting",
        "neon lights",
        "backlit",
    ]

    weather_patterns = [
        "rain",
        "rainy",
        "snow",
        "snowy",
        "cloudy",
        "foggy",
        "mist",
        "storm",
        "sunny",
        "clear sky",
        "overcast",
    ]

    time_patterns = [
        "day",
        "night",
        "twilight",
        "dawn",
        "dusk",
        "morning",
        "evening",
        "afternoon",
        "midnight",
        "noon",
    ]

    # Process remaining terms in other
    for term in list(categorized_tags.get("other", [])):
        term_lower = term.lower()
        # Check lighting
        if any(light in term_lower for light in lighting_patterns):
            lighting_terms.append(term)
            if term in categorized_tags["other"]:
                categorized_tags["other"].remove(term)
        # Check weather
        elif any(weather in term_lower for weather in weather_patterns):
            weather_terms.append(term)
            if term in categorized_tags["other"]:
                categorized_tags["other"].remove(term)
        # Check time
        elif any(time in term_lower for time in time_patterns):
            time_terms.append(term)
            if term in categorized_tags["other"]:
                categorized_tags["other"].remove(term)

    # Combine all parts into a fluent caption
    caption_parts = []

    # Start with quality and subject
    if style_phrase and quality_adj:
        subject_phrase = f"{article} {quality_adj} {style_phrase} of a {subject}"
    elif style_phrase:
        subject_phrase = f"{article} {style_phrase} of a {subject}"
    elif quality_adj:
        subject_phrase = f"{article} {quality_adj} artwork of a {subject}"
    else:
        subject_phrase = f"{article} artwork of a {subject}"

    # Clean up extra spaces and fix grammatical issues
    subject_phrase = subject_phrase.replace("  ", " ")
    subject_phrase = subject_phrase.replace(" of a a ", " of a ")

    caption_parts.append(subject_phrase)

    # Add setting, time, and weather details if available
    setting_details = []
    if setting_phrase:
        setting_details.append(setting_phrase)

    if time_terms:
        if len(time_terms) == 1:
            setting_details.append(f"during {time_terms[0]}")
        else:
            setting_details.append(f"during {' and '.join(time_terms)}")

    if weather_terms:
        if len(weather_terms) == 1:
            setting_details.append(f"with {weather_terms[0]} weather")
        else:
            setting_details.append(f"with {' and '.join(weather_terms)} weather")

    if lighting_terms:
        if len(lighting_terms) == 1:
            setting_details.append(f"with {lighting_terms[0]}")
        else:
            setting_details.append(f"with {' and '.join(lighting_terms)}")

    if setting_details:
        caption_parts.append(" " + ", ".join(setting_details))

    # Add other details like clothing, accessories, pose
    other_details = []

    if categorized_tags.get("clothing", []):
        clothing = categorized_tags["clothing"]
        if len(clothing) == 1:
            other_details.append(f"wearing {clothing[0]}")
        elif len(clothing) == 2:
            other_details.append(f"wearing {clothing[0]} and {clothing[1]}")
        else:
            other_details.append(f"wearing {', '.join(clothing[:-1])}, and {clothing[-1]}")

    if categorized_tags.get("accessory", []):
        accessories = categorized_tags["accessory"]
        if len(accessories) == 1:
            other_details.append(f"with {accessories[0]}")
        elif len(accessories) == 2:
            other_details.append(f"with {accessories[0]} and {accessories[1]}")
        else:
            other_details.append(f"with {', '.join(accessories[:-1])}, and {accessories[-1]}")

    if categorized_tags.get("pose", []) or categorized_tags.get("action", []):
        pose_actions = []
        if categorized_tags.get("pose", []):
            pose_actions.extend(categorized_tags["pose"])
        if categorized_tags.get("action", []):
            pose_actions.extend(categorized_tags["action"])

        if len(pose_actions) == 1:
            other_details.append(pose_actions[0])
        elif len(pose_actions) == 2:
            other_details.append(f"{pose_actions[0]} and {pose_actions[1]}")
        else:
            other_details.append(f"{', '.join(pose_actions[:-1])}, and {pose_actions[-1]}")

    # Add expression if available
    if categorized_tags.get("expression", []):
        expressions = categorized_tags["expression"]
        if len(expressions) == 1:
            other_details.append(f"with a {expressions[0]} expression")
        else:
            other_details.append(f"with a {' and '.join(expressions)} expression")

    # Add remaining "other" details
    if categorized_tags.get("other", []):
        # Filter out terms we've already handled
        remaining_terms = []
        for term in categorized_tags["other"]:
            if term not in subject_terms and term not in style_terms:
                remaining_terms.append(term)

        if remaining_terms:
            if len(remaining_terms) == 1:
                other_details.append(remaining_terms[0])
            else:
                other_details.append(", ".join(remaining_terms))

    if other_details:
        caption_parts.append(", " + ", ".join(other_details))

    # Finalize the caption with a period
    caption = "".join(caption_parts)

    # Clean up any remaining formatting issues
    caption = caption.replace("  ", " ").strip()
    if not caption.endswith("."):
        caption += "."

    # Capitalize the first letter
    caption = caption[0].upper() + caption[1:]

    return caption
