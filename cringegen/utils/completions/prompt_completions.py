"""
Prompt completion utilities for shell auto-completion.
"""

import os
from typing import Dict, List, Optional, Set, Tuple

from ...utils.logger import get_logger
from ...utils.lora_metadata import (
    get_activation_text,
    get_lora_tag_frequencies,
)

logger = get_logger(__name__)

# Common quality terms for prompts
QUALITY_TERMS = [
    "masterpiece",
    "best quality",
    "high quality",
    "detailed",
    "intricate",
    "professional",
    "highly detailed",
    "sharp focus",
    "8k",
    "4k",
    "award winning",
    "stunning",
    "beautiful",
    "excellent",
    "amazing",
    "hires",
    "ultra detailed",
    "realistic",
]

# Common style terms for prompts
STYLE_TERMS = [
    "digital art",
    "digital painting",
    "illustration",
    "concept art",
    "anime",
    "manga",
    "oil painting",
    "watercolor",
    "sketch",
    "drawing",
    "realistic",
    "photorealistic",
    "cinematic",
    "dramatic",
    "fantasy",
    "sci-fi",
    "cyberpunk",
    "steampunk",
    "gothic",
    "surreal",
]

# Common composition terms for prompts
COMPOSITION_TERMS = [
    "full body",
    "portrait",
    "closeup",
    "wide shot",
    "dynamic pose",
    "action shot",
    "from above",
    "from below",
    "side view",
    "front view",
    "back view",
    "bokeh",
    "depth of field",
    "in focus",
    "blurred background",
]


def get_prompt_completions(
    partial_term: str = "",
    lora_names: Optional[List[str]] = None,
    checkpoint_name: Optional[str] = None,
    max_results: int = 30,
) -> List[Tuple[str, str]]:
    """
    Get smart prompt completions based on LoRA context.

    Args:
        partial_term: Partial prompt term to filter by
        lora_names: List of selected LoRA names (for context-aware completions)
        checkpoint_name: Selected checkpoint name (for context)
        max_results: Maximum number of results to return

    Returns:
        List of tuples (term, description) for shell completion
    """
    completions = []

    try:
        # Get all candidate terms
        terms = set()

        # Add LoRA-specific terms
        if lora_names:
            for lora_name in lora_names:
                terms.update(get_lora_keywords(lora_name))

        # Add common terms by category
        terms.update(QUALITY_TERMS)
        terms.update(STYLE_TERMS)
        terms.update(COMPOSITION_TERMS)

        # Filter by partial term and convert to list
        filtered_terms = [term for term in terms if partial_term.lower() in term.lower()]

        # Sort and limit results
        filtered_terms = sorted(filtered_terms)[:max_results]

        # Add descriptions
        for term in filtered_terms:
            if term in QUALITY_TERMS:
                completions.append((term, "Quality term"))
            elif term in STYLE_TERMS:
                completions.append((term, "Style term"))
            elif term in COMPOSITION_TERMS:
                completions.append((term, "Composition term"))
            else:
                # Check if it's from a LoRA
                is_lora_term = False
                if lora_names:
                    for lora_name in lora_names:
                        lora_keywords = get_lora_keywords(lora_name)
                        if term in lora_keywords:
                            is_lora_term = True
                            completions.append((term, f"From {os.path.basename(lora_name)}"))
                            break

                if not is_lora_term:
                    completions.append((term, "Prompt term"))

    except Exception as e:
        logger.error(f"Error generating prompt completions: {e}")

    return completions


def get_activation_completions(lora_name: str, partial_term: str = "") -> List[Tuple[str, str]]:
    """
    Get activation text completions for a specific LoRA.

    Args:
        lora_name: Name of the LoRA
        partial_term: Partial term to filter by

    Returns:
        List of tuples (term, description) for shell completion
    """
    completions = []

    try:
        # Get activation text
        activation_texts = get_activation_text(lora_name)

        if not activation_texts:
            return completions

        # Filter by partial term
        filtered_texts = [text for text in activation_texts if partial_term.lower() in text.lower()]

        # Add descriptions
        for text in filtered_texts:
            completions.append((text, f"Activation for {os.path.basename(lora_name)}"))

    except Exception as e:
        logger.error(f"Error getting activation completions for {lora_name}: {e}")

    return completions


def get_lora_keywords(lora_name: str, top_n: int = 20) -> Set[str]:
    """
    Get a set of keywords for a LoRA (activation terms + top tags).

    Args:
        lora_name: Name of the LoRA
        top_n: Number of top tags to include

    Returns:
        Set of keywords for the LoRA
    """
    keywords = set()

    try:
        # Add activation text
        activation_texts = get_activation_text(lora_name)
        if activation_texts:
            keywords.update(activation_texts)

        # Add top tags
        tag_frequencies = get_lora_tag_frequencies(lora_name)
        if tag_frequencies:
            # Sort by frequency and take top_n
            sorted_tags = sorted(tag_frequencies.items(), key=lambda x: x[1], reverse=True)
            top_tags = [tag for tag, _ in sorted_tags[:top_n]]
            keywords.update(top_tags)

    except Exception as e:
        logger.error(f"Error getting keywords for {lora_name}: {e}")

    return keywords


def generate_common_term_completions() -> List[Tuple[str, str]]:
    """
    Generate completions for common prompt terms.

    Returns:
        List of tuples (term, description) for shell completion
    """
    completions = []

    # Add quality terms
    for term in QUALITY_TERMS:
        completions.append((term, "Quality term"))

    # Add style terms
    for term in STYLE_TERMS:
        completions.append((term, "Style term"))

    # Add composition terms
    for term in COMPOSITION_TERMS:
        completions.append((term, "Composition term"))

    return completions
