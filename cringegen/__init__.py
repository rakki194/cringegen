"""
CringeGen: AI Image Prompt Generation Toolkit

A comprehensive toolkit for generating, analyzing, and enhancing image generation prompts,
with specialized support for anthro/furry content.

This package provides utilities for building, analyzing, and refining prompts for
text-to-image generation models like Stable Diffusion.
"""

__version__ = "0.1.0"

# Import key components for easier access
from .data import (
    # Most commonly used data structures
    SPECIES_TAXONOMY,
    ANTHRO_SPECIES,
    POPULAR_ANTHRO_SPECIES,
    FANTASY_SPECIES,
    BASIC_COLORS,
    COLOR_PATTERNS,
    BACKGROUND_SETTINGS,
)

# Add all modules that should be exposed at the package level
__all__ = [
    "data",
    "utils",
    "prompt",
    "analyze",
    # Commonly used data structures
    "SPECIES_TAXONOMY",
    "ANTHRO_SPECIES",
    "POPULAR_ANTHRO_SPECIES",
    "FANTASY_SPECIES",
    "BASIC_COLORS",
    "COLOR_PATTERNS",
    "BACKGROUND_SETTINGS",
]

# Make these modules directly accessible - moved to the end to avoid circular imports
from . import data
from . import utils

# These imports might cause circular dependencies, so we'll import them only when needed
# Instead of importing them here, users can access them through the namespace
# from . import prompt
# from . import analyze
