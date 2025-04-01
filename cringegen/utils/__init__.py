"""
cringegen Utilities

This module provides utility functions for working with cringegen data structures.
"""

# Import utilities for working with styles
from .style_utils import (
    get_style_suggestions,
    generate_style_prompt,
    find_style_by_mood,
    get_complementary_styles,
    get_random_style_combination,
)

# Import utilities for working with LoRAs
from .lora_utils import (
    select_random_lora,
    create_balanced_lora_set,
    extract_lora_info_from_filename,
    find_loras_for_prompt,
    generate_lora_prompt_addition,
    upgrade_legacy_lora_selection,
)

# Import utilities for working with characters
from .character_utils import (
    generate_random_character,
    character_to_prompt,
    find_character_by_description,
    is_character_compatible,
    filter_incompatible_elements,
    enhance_character_prompt,
    suggest_character_improvements,
    preset_fantasy_adventurer,
    preset_modern_casual,
    migrate_legacy_character_data,
    generate_canon_character_prompt,
)

# Import utilities for working with prompts
from .prompt_utils import (
    format_prompt,
    print_prompt,
    parse_prompt_template,
    combine_prompt_segments,
    clean_prompt,
)

# Import model utilities
from . import model_utils

# Export model utility functions
from .model_utils import (
    ModelOptimizer,
    detect_model,
    optimize_prompt,
    get_model_optimal_parameters,
    get_model_optimal_resolution,
)

__all__ = [
    # Style utilities
    "get_style_suggestions",
    "generate_style_prompt",
    "find_style_by_mood",
    "get_complementary_styles",
    "get_random_style_combination",
    # LoRA utilities
    "select_random_lora",
    "create_balanced_lora_set",
    "extract_lora_info_from_filename",
    "find_loras_for_prompt",
    "generate_lora_prompt_addition",
    "upgrade_legacy_lora_selection",
    # Character utilities
    "generate_random_character",
    "character_to_prompt",
    "find_character_by_description",
    "is_character_compatible",
    "filter_incompatible_elements",
    "enhance_character_prompt",
    "suggest_character_improvements",
    "preset_fantasy_adventurer",
    "preset_modern_casual",
    "migrate_legacy_character_data",
    "generate_canon_character_prompt",
    # Prompt utilities
    "format_prompt",
    "print_prompt",
    "parse_prompt_template",
    "combine_prompt_segments",
    "clean_prompt",
    # Model utilities
    "ModelOptimizer",
    "detect_model",
    "optimize_prompt",
    "get_model_optimal_parameters",
    "get_model_optimal_resolution",
]
