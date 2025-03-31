"""
Natural language processing utilities for prompt generation.
"""

from .background_utils import (
    enhance_prompt_with_background,
    generate_background_description,
    generate_scene_description,
    get_complementary_locations,
)
from .color_utils import (
    generate_color_description,
    get_body_covering_type,
    get_complementary_colors,
    get_random_colors,
    get_random_marking,
    get_random_pattern,
    parse_color_input,
)

# Import from new data utilities module
from .data_utils import (
    analyze_tags_with_data_structures,
    classify_tag,
    enhance_tag_description,
    get_all_anime_character_types,
    get_all_art_styles,
    get_all_game_character_types,
    get_all_species,
    get_anime_character_details,
    get_art_style_details,
    get_compatible_accessories,
    get_game_character_details,
    get_tag_suggestions,
    get_taxonomy_group,
)
from .general_enhancer import (
    analyze_prompt,
    create_prompt_variations,
    enhance_prompt_general,
    enhance_prompt_with_details,
    simplify_prompt,
)
from .nltk_utils import (
    analyze_sentiment,
    analyze_text_pos,
    compute_text_similarity,
    extract_entities,
    extract_keywords_with_pos,
    extract_phrases,
    find_collocations,
    generate_ngrams,
    get_antonyms,
    get_hypernyms,
    get_hyponyms,
    get_synonyms,
    lemmatize_text,
)
from .prompt_analyzer import analyze_prompt as analyze_prompt_structure
from .prompt_analyzer import (
    compare_prompts,
    detect_prompt_style,
    extract_keywords,
    get_prompt_structure,
)
from .prompt_analyzer import simplify_prompt as simplify_prompt_structure
from .prompt_analyzer import (
    suggest_improvements,
)
from .species_utils import (
    enhance_prompt_with_anatomy,
    generate_species_description,
    get_anatomical_terms,
    get_species_accessories,
    get_species_colors,
)

# Import from new modules
from .tag_converter import (
    categorize_tags,
    natural_tags_to_text,
    tags_to_text,
    text_to_tags,
)

__all__ = [
    # Species NLP utilities
    "get_anatomical_terms",
    "enhance_prompt_with_anatomy",
    "get_species_accessories",
    "get_species_colors",
    "generate_species_description",
    # Background NLP utilities
    "generate_background_description",
    "generate_scene_description",
    "get_complementary_locations",
    "enhance_prompt_with_background",
    # General enhancer utilities
    "enhance_prompt_general",
    "enhance_prompt_with_details",
    "simplify_prompt",
    "create_prompt_variations",
    "analyze_prompt",
    # Color utilities
    "get_body_covering_type",
    "get_random_colors",
    "get_complementary_colors",
    "get_random_pattern",
    "get_random_marking",
    "generate_color_description",
    "parse_color_input",
    # Tag converter utilities
    "tags_to_text",
    "text_to_tags",
    "categorize_tags",
    "natural_tags_to_text",
    # Prompt analyzer utilities
    "analyze_prompt_structure",
    "get_prompt_structure",
    "compare_prompts",
    "suggest_improvements",
    "simplify_prompt_structure",
    "extract_keywords",
    "detect_prompt_style",
    # NLTK utilities
    "analyze_text_pos",
    "extract_phrases",
    "get_synonyms",
    "get_antonyms",
    "get_hypernyms",
    "get_hyponyms",
    "analyze_sentiment",
    "extract_keywords_with_pos",
    "lemmatize_text",
    "compute_text_similarity",
    "extract_entities",
    "generate_ngrams",
    "find_collocations",
    # Data utilities
    "get_all_species",
    "get_all_anime_character_types",
    "get_all_game_character_types",
    "get_all_art_styles",
    "get_taxonomy_group",
    "get_anime_character_details",
    "get_game_character_details",
    "get_art_style_details",
    "classify_tag",
    "enhance_tag_description",
    "get_compatible_accessories",
    "get_tag_suggestions",
    "analyze_tags_with_data_structures",
]
