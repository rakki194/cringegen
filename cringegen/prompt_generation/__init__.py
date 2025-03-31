"""
Prompt generation components for CringeGen.
"""

# Import base components
from .base import (
    CharacterComponent,
    NegativePromptComponent,
    PoseComponent,
    PromptComponent,
    PromptGenerator,
    QualityComponent,
    SettingComponent,
    StyleComponent,
)

# Import from generators
from .generators.furry_generator import FurryPromptGenerator, NsfwFurryPromptGenerator

# Import LLM generator
from .llm_generator import LLMPromptGenerator

# Import NLP utilities
from .nlp import (  # Species utilities; Background utilities; General enhancer utilities; Color utilities
    analyze_prompt,
    create_prompt_variations,
    enhance_prompt_general,
    enhance_prompt_with_anatomy,
    enhance_prompt_with_background,
    enhance_prompt_with_details,
    generate_background_description,
    generate_color_description,
    generate_scene_description,
    generate_species_description,
    get_anatomical_terms,
    get_body_covering_type,
    get_complementary_colors,
    get_complementary_locations,
    get_random_colors,
    get_random_marking,
    get_random_pattern,
    get_species_accessories,
    get_species_colors,
    parse_color_input,
    simplify_prompt,
)

__all__ = [
    # Generators
    "FurryPromptGenerator",
    "NsfwFurryPromptGenerator",
    "LLMPromptGenerator",
    # NLP utilities - Species
    "get_anatomical_terms",
    "enhance_prompt_with_anatomy",
    "get_species_accessories",
    "get_species_colors",
    "generate_species_description",
    # NLP utilities - Background
    "generate_background_description",
    "generate_scene_description",
    "get_complementary_locations",
    "enhance_prompt_with_background",
    # NLP utilities - General enhancer
    "enhance_prompt_general",
    "enhance_prompt_with_details",
    "simplify_prompt",
    "create_prompt_variations",
    "analyze_prompt",
    # NLP utilities - Color
    "get_body_covering_type",
    "get_random_colors",
    "get_complementary_colors",
    "get_random_pattern",
    "get_random_marking",
    "generate_color_description",
    "parse_color_input",
    # Base components
    "PromptComponent",
    "PromptGenerator",
    "CharacterComponent",
    "PoseComponent",
    "SettingComponent",
    "StyleComponent",
    "QualityComponent",
    "NegativePromptComponent",
]
