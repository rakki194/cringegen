"""
Prompt Utility Functions

This module provides utility functions for working with image generation prompts.
It handles formatting, parsing, and cleaning operations for text prompts.
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union


def format_prompt(prompt_segments: Dict[str, str], template: Optional[str] = None) -> str:
    """
    Format a set of prompt segments according to a template.

    Args:
        prompt_segments: Dictionary of prompt segments
        template: Optional template string

    Returns:
        Formatted prompt string
    """
    if not template:
        # Default template just joins all segments with commas
        return ", ".join(filter(None, prompt_segments.values()))

    # Replace template placeholders with actual values
    result = template
    for key, value in prompt_segments.items():
        if value:  # Only substitute non-empty values
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, value)

    # Remove any remaining placeholders
    import re

    result = re.sub(r"\{[^}]+\}", "", result)

    # Clean up any double commas or trailing/leading commas
    result = result.replace(", ,", ",").strip(", ")

    return result


def print_prompt(prompt: str, title: Optional[str] = None, width: int = 80) -> None:
    """
    Print a formatted prompt to the console.

    Args:
        prompt: The prompt to print
        title: Optional title to display
        width: Width of the display box
    """
    if not prompt:
        return

    border = "-" * width
    print("\n" + border)

    if title:
        print(f"| {title.upper()}")
        print(border)

    # Word wrap the prompt
    import textwrap

    wrapped_prompt = textwrap.fill(prompt, width - 4)
    for line in wrapped_prompt.split("\n"):
        print(f"| {line}")

    print(border + "\n")


def parse_prompt_template(template: str) -> List[str]:
    """
    Parse a prompt template to extract placeholders.

    Args:
        template: Template string with {placeholders}

    Returns:
        List of placeholder names
    """
    import re

    placeholders = re.findall(r"\{([^}]+)\}", template)
    return placeholders


def combine_prompt_segments(segments: List[str], join_str: str = ", ") -> str:
    """
    Combine multiple prompt segments into a single prompt.

    Args:
        segments: List of prompt segments
        join_str: String to use for joining segments

    Returns:
        Combined prompt string
    """
    # Filter out None or empty segments
    filtered_segments = [s for s in segments if s]
    return join_str.join(filtered_segments)


def clean_prompt(prompt: str) -> str:
    """
    Clean a prompt by removing extra whitespace, normalizing commas, etc.

    Args:
        prompt: The prompt to clean

    Returns:
        Cleaned prompt string
    """
    # Remove extra whitespace
    cleaned = " ".join(prompt.split())

    # Normalize commas
    cleaned = cleaned.replace(" , ", ", ")
    cleaned = cleaned.replace(",,", ",")

    # Remove leading/trailing commas and whitespace
    cleaned = cleaned.strip(", ")

    return cleaned
