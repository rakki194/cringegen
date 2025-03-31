"""
General-purpose prompt enhancement utilities.
"""

import random
import re
from typing import Dict, List, Optional, Tuple, Union

# Descriptive adjectives by category
DESCRIPTIVE_ADJECTIVES = {
    "quality": [
        "masterpiece",
        "high quality",
        "detailed",
        "intricate",
        "professional",
        "pristine",
        "crisp",
        "sharp",
        "elegant",
        "exquisite",
        "photorealistic",
    ],
    "lighting": [
        "volumetric lighting",
        "rim lighting",
        "studio lighting",
        "soft lighting",
        "dramatic lighting",
        "cinematic lighting",
        "ambient lighting",
        "natural lighting",
        "atmospheric",
        "backlit",
        "golden hour",
        "blue hour",
    ],
    "style": [
        "trending on artstation",
        "award winning",
        "professional",
        "highly detailed",
        "concept art",
        "matte painting",
        "digital painting",
        "digital art",
        "illustration",
        "octane render",
        "realistic",
        "hyperrealistic",
        "stylized",
        "vibrant",
    ],
    "composition": [
        "dynamic composition",
        "rule of thirds",
        "centered composition",
        "wide angle",
        "telephoto",
        "panoramic",
        "portrait",
        "landscape",
        "close-up",
        "wide shot",
        "dramatic angle",
        "perspective",
        "foreshortening",
    ],
    "mood": [
        "atmospheric",
        "moody",
        "emotional",
        "serene",
        "peaceful",
        "tranquil",
        "chaotic",
        "energetic",
        "dreamy",
        "surreal",
        "ethereal",
        "mysterious",
        "fantastical",
        "magical",
        "whimsical",
    ],
}

# Action verbs by category
ACTION_VERBS = {
    "casual": [
        "standing",
        "sitting",
        "walking",
        "looking",
        "smiling",
        "talking",
        "reading",
        "drinking",
        "eating",
        "relaxing",
        "resting",
        "waiting",
        "watching",
    ],
    "dynamic": [
        "running",
        "jumping",
        "dancing",
        "flying",
        "swimming",
        "climbing",
        "diving",
        "swinging",
        "fighting",
        "racing",
        "spinning",
        "twirling",
        "flipping",
    ],
    "emotional": [
        "laughing",
        "crying",
        "shouting",
        "screaming",
        "celebrating",
        "cheering",
        "frowning",
        "sighing",
        "smirking",
        "grinning",
        "grimacing",
        "winking",
    ],
    "social": [
        "hugging",
        "holding hands",
        "kissing",
        "embracing",
        "meeting",
        "greeting",
        "waving",
        "bowing",
        "shaking hands",
        "high-fiving",
        "playing with",
        "talking to",
    ],
}

# Composition terms
COMPOSITION_TERMS = [
    "full body",
    "half body",
    "portrait",
    "from above",
    "from below",
    "side view",
    "front view",
    "3/4 view",
    "profile",
    "back view",
    "close-up",
    "extreme close-up",
    "wide shot",
    "medium shot",
]


def analyze_prompt(prompt: str) -> Dict[str, bool]:
    """Analyze a prompt to determine what enhancement it might need.

    Args:
        prompt: The original prompt text

    Returns:
        Dictionary indicating what enhancements are needed
    """
    prompt_lower = prompt.lower()

    # Check what the prompt already contains
    has_quality = any(term in prompt_lower for term in DESCRIPTIVE_ADJECTIVES["quality"])
    has_lighting = any(term in prompt_lower for term in DESCRIPTIVE_ADJECTIVES["lighting"])
    has_style = any(term in prompt_lower for term in DESCRIPTIVE_ADJECTIVES["style"])
    has_composition = any(term in prompt_lower for term in COMPOSITION_TERMS)
    has_mood = any(term in prompt_lower for term in DESCRIPTIVE_ADJECTIVES["mood"])
    has_action = any(
        term in prompt_lower for category in ACTION_VERBS.values() for term in category
    )

    return {
        "needs_quality": not has_quality,
        "needs_lighting": not has_lighting,
        "needs_style": not has_style,
        "needs_composition": not has_composition,
        "needs_mood": not has_mood,
        "needs_action": not has_action,
    }


def enhance_prompt_general(
    prompt: str,
    add_quality: bool = True,
    add_lighting: bool = True,
    add_style: bool = True,
    add_composition: bool = False,
    add_mood: bool = True,
    add_action: bool = False,
    enhancement_level: int = 2,
) -> str:
    """Enhance a general prompt with additional details.

    Args:
        prompt: The original prompt text
        add_quality: Whether to add quality-related terms
        add_lighting: Whether to add lighting-related terms
        add_style: Whether to add style-related terms
        add_composition: Whether to add composition-related terms
        add_mood: Whether to add mood-related terms
        add_action: Whether to add action-related terms
        enhancement_level: Level of enhancement (1-3)

    Returns:
        Enhanced prompt with additional details
    """
    # Analyze the prompt first
    analysis = analyze_prompt(prompt)

    # Override enhancement flags with analysis if they're True
    add_quality = add_quality and analysis["needs_quality"]
    add_lighting = add_lighting and analysis["needs_lighting"]
    add_style = add_style and analysis["needs_style"]
    add_composition = add_composition and analysis["needs_composition"]
    add_mood = add_mood and analysis["needs_mood"]
    add_action = add_action and analysis["needs_action"]

    # Determine number of terms to add for each category based on enhancement level
    num_terms = max(1, min(enhancement_level, 3))

    # Collect all enhancement terms
    enhancement_terms = []

    if add_quality:
        quality_terms = random.sample(DESCRIPTIVE_ADJECTIVES["quality"], num_terms)
        enhancement_terms.extend(quality_terms)

    if add_lighting:
        lighting_terms = random.sample(DESCRIPTIVE_ADJECTIVES["lighting"], num_terms)
        enhancement_terms.extend(lighting_terms)

    if add_style:
        style_terms = random.sample(DESCRIPTIVE_ADJECTIVES["style"], num_terms)
        enhancement_terms.extend(style_terms)

    if add_composition:
        # Only add one composition term to avoid conflicts
        composition_term = random.choice(COMPOSITION_TERMS)
        enhancement_terms.append(composition_term)

    if add_mood:
        mood_terms = random.sample(DESCRIPTIVE_ADJECTIVES["mood"], num_terms)
        enhancement_terms.extend(mood_terms)

    if add_action and "needs_action" in analysis and analysis["needs_action"]:
        # Choose a random action category
        action_category = random.choice(list(ACTION_VERBS.keys()))
        action_term = random.choice(ACTION_VERBS[action_category])
        enhancement_terms.append(action_term)

    # Combine the original prompt with the enhancement terms
    if enhancement_terms:
        enhanced_prompt = f"{prompt}, {', '.join(enhancement_terms)}"
    else:
        enhanced_prompt = prompt

    return enhanced_prompt


def enhance_prompt_with_details(
    prompt: str,
    subject: Optional[str] = None,
    action: Optional[str] = None,
    location: Optional[str] = None,
    style: Optional[str] = None,
    enhancement_level: int = 2,
) -> str:
    """Enhance a prompt with specific details about subject, action, location, and style.

    Args:
        prompt: The original prompt text
        subject: The main subject to emphasize
        action: The action the subject is performing
        location: The location or setting
        style: The art style to apply
        enhancement_level: Level of enhancement (1-3)

    Returns:
        Enhanced prompt with specific details
    """
    # Start with general enhancement
    enhanced_prompt = enhance_prompt_general(prompt, enhancement_level=enhancement_level)

    # Add specific details if provided
    details = []

    if subject:
        details.append(f"focus on {subject}")

    if action:
        # Check if the action is already in the prompt
        if action.lower() not in prompt.lower():
            details.append(action)

    if location and location.lower() not in prompt.lower():
        details.append(f"in {location}")

    if style and style.lower() not in prompt.lower():
        details.append(f"{style} style")

    # Add the details to the enhanced prompt
    if details:
        enhanced_prompt = f"{enhanced_prompt}, {', '.join(details)}"

    return enhanced_prompt


def simplify_prompt(prompt: str) -> str:
    """Simplify a prompt by removing redundant or excessive terms.

    Args:
        prompt: The original prompt text

    Returns:
        Simplified prompt
    """
    # Split the prompt into terms
    terms = [term.strip() for term in prompt.split(",")]

    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        term_lower = term.lower()
        if term_lower not in seen and term:
            seen.add(term_lower)
            unique_terms.append(term)

    # Rejoin the terms
    simplified = ", ".join(unique_terms)

    return simplified


def create_prompt_variations(prompt: str, num_variations: int = 3) -> List[str]:
    """Create variations of a prompt by adding different enhancements.

    Args:
        prompt: The original prompt text
        num_variations: Number of variations to create

    Returns:
        List of prompt variations
    """
    variations = []

    for _ in range(num_variations):
        # Randomize enhancement parameters
        add_quality = random.choice([True, False])
        add_lighting = random.choice([True, False])
        add_style = random.choice([True, False])
        add_composition = random.choice([True, False])
        add_mood = random.choice([True, False])
        add_action = random.choice([True, False])
        enhancement_level = random.randint(1, 3)

        # Create a variation
        variation = enhance_prompt_general(
            prompt,
            add_quality=add_quality,
            add_lighting=add_lighting,
            add_style=add_style,
            add_composition=add_composition,
            add_mood=add_mood,
            add_action=add_action,
            enhancement_level=enhancement_level,
        )

        # Ensure uniqueness
        if variation not in variations and variation != prompt:
            variations.append(variation)

    # If we couldn't generate enough unique variations, add some with specific details
    while len(variations) < num_variations:
        # Use enhance_prompt_with_details with random parameters
        subject = random.choice(["character", "subject", "figure", "person", "entity"])
        action = random.choice(
            ACTION_VERBS["casual"] + ACTION_VERBS["dynamic"] + ACTION_VERBS["emotional"]
        )
        enhancement_level = random.randint(1, 3)

        variation = enhance_prompt_with_details(
            prompt, subject=subject, action=action, enhancement_level=enhancement_level
        )

        # Ensure uniqueness
        if variation not in variations and variation != prompt:
            variations.append(variation)

    return variations
