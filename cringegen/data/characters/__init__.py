"""
Character Data Module

This module organizes character templates in individual files for easier maintenance.
It provides a centralized way to access character data while keeping the data separate.
"""

import os
import importlib
from typing import Dict, List, Any, Optional

from ..character_taxonomy import (
    CharacterTemplate,
    FAMOUS_CHARACTER_TEMPLATES,
    get_character_by_name,
)

# Directory where character data files are stored
CHARACTERS_DIR = os.path.dirname(os.path.abspath(__file__))

# Dictionary to store character data loaded from individual files
INDIVIDUAL_CHARACTER_TEMPLATES: Dict[str, CharacterTemplate] = {}


def load_all_characters() -> Dict[str, CharacterTemplate]:
    """
    Load all individual character data from files in the characters directory.

    Returns:
        Dictionary mapping character names to their templates
    """
    global INDIVIDUAL_CHARACTER_TEMPLATES

    # Get all Python files in the characters directory
    for filename in os.listdir(CHARACTERS_DIR):
        if filename.endswith(".py") and filename != "__init__.py":
            # Extract module name (remove .py extension)
            module_name = filename[:-3]

            try:
                # Import the module
                module = importlib.import_module(f".{module_name}", package=__name__)

                # If module has a CHARACTER_TEMPLATE attribute, add it to the dictionary
                if hasattr(module, "CHARACTER_TEMPLATE"):
                    character_name = module_name.lower()
                    INDIVIDUAL_CHARACTER_TEMPLATES[character_name] = module.CHARACTER_TEMPLATE

                    # Also update the main templates dictionary for compatibility
                    FAMOUS_CHARACTER_TEMPLATES[character_name] = module.CHARACTER_TEMPLATE
            except ImportError as e:
                print(f"Error importing character module {module_name}: {e}")

    return INDIVIDUAL_CHARACTER_TEMPLATES


def get_individual_character(name: str) -> Optional[CharacterTemplate]:
    """
    Get a character template from individual character files.

    Args:
        name: Name of the character to retrieve

    Returns:
        CharacterTemplate for the requested character, or None if not found
    """
    name_lower = name.lower()

    # If character not already loaded, try to load it specifically
    if name_lower not in INDIVIDUAL_CHARACTER_TEMPLATES:
        try:
            # Try to import the module with this name
            module = importlib.import_module(f".{name_lower}", package=__name__)

            # If module has a CHARACTER_TEMPLATE attribute, add it
            if hasattr(module, "CHARACTER_TEMPLATE"):
                INDIVIDUAL_CHARACTER_TEMPLATES[name_lower] = module.CHARACTER_TEMPLATE
                FAMOUS_CHARACTER_TEMPLATES[name_lower] = module.CHARACTER_TEMPLATE
        except ImportError:
            # Character file doesn't exist, fall back to main dictionary
            return get_character_by_name(name)

    return INDIVIDUAL_CHARACTER_TEMPLATES.get(name_lower)


def get_available_characters() -> List[str]:
    """
    Get a list of all available character names from individual files.

    Returns:
        List of available character names
    """
    # If characters haven't been loaded yet, load them
    if not INDIVIDUAL_CHARACTER_TEMPLATES:
        load_all_characters()

    return list(INDIVIDUAL_CHARACTER_TEMPLATES.keys())


# Load all characters when the module is imported
load_all_characters()

__all__ = [
    "INDIVIDUAL_CHARACTER_TEMPLATES",
    "load_all_characters",
    "get_individual_character",
    "get_available_characters",
]
