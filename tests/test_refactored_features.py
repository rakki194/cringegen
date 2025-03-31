#!/usr/bin/env python3
"""
Test script to verify all refactored components in CringeGen.
"""

import random
import sys
from typing import Dict, List, Optional

# Set the random seed for reproducibility
random.seed(42)

# Import data structures using new centralized data module
from cringegen.data import taxonomy, accessories, colors, backgrounds, habitats

# Original imports for reference:
# from cringegen.data.species_data import SPECIES_TAXONOMY, SPECIES_ACCESSORIES, SPECIES_COLORS
# from cringegen.data.background_data import BACKGROUND_SETTINGS, SPECIES_HABITATS

# Map old variables to new module structure
SPECIES_TAXONOMY = taxonomy.SPECIES_TAXONOMY
SPECIES_ACCESSORIES = accessories.SPECIES_ACCESSORIES
SPECIES_COLORS = colors.SPECIES_COLORS
BACKGROUND_SETTINGS = backgrounds.BACKGROUND_SETTINGS
SPECIES_HABITATS = habitats.SPECIES_HABITATS

# Import NLP utilities
from cringegen.prompt_generation.nlp.species_utils import (
    get_anatomical_terms,
    enhance_prompt_with_anatomy,
    get_species_accessories,
    get_species_colors,
    generate_species_description,
)

from cringegen.prompt_generation.nlp.background_utils import (
    generate_background_description,
    generate_scene_description,
    get_complementary_locations,
    enhance_prompt_with_background,
)

from cringegen.prompt_generation.nlp.general_enhancer import (
    enhance_prompt_general,
    enhance_prompt_with_details,
    simplify_prompt,
    create_prompt_variations,
    analyze_prompt,
)

# Import generators
from cringegen.prompt_generation.generators.furry_generator import (
    FurryPromptGenerator,
    NsfwFurryPromptGenerator,
)


def test_data_structures() -> None:
    """Test the data structures from the data directory."""
    print("\n" + "=" * 50)
    print("TESTING DATA STRUCTURES")
    print("=" * 50)

    # Test species taxonomy
    print("\nSpecies Taxonomy:")
    fox_taxonomy = SPECIES_TAXONOMY.get("fox", "unknown")
    cat_taxonomy = SPECIES_TAXONOMY.get("cat", "unknown")
    print(f"Fox taxonomy: {fox_taxonomy}")
    print(f"Cat taxonomy: {cat_taxonomy}")

    # Test species accessories
    print("\nSpecies Accessories:")
    fox_accessories = SPECIES_ACCESSORIES.get("canine", {}).get("male", [])
    cat_accessories = SPECIES_ACCESSORIES.get("feline", {}).get("female", [])
    print(f"Fox (male) accessories: {fox_accessories}")
    print(f"Cat (female) accessories: {cat_accessories}")

    # Test background settings
    print("\nBackground Settings:")
    forest_data = BACKGROUND_SETTINGS.get("forest", {})
    city_data = BACKGROUND_SETTINGS.get("city", {})
    print(f"Forest descriptors: {forest_data.get('descriptors', [])[:3]}...")
    print(f"City features: {city_data.get('features', [])[:3]}...")

    # Test species habitats
    print("\nSpecies Habitats:")
    wolf_habitats = SPECIES_HABITATS.get("wolf", [])
    otter_habitats = SPECIES_HABITATS.get("otter", [])
    print(f"Wolf habitats: {wolf_habitats}")
    print(f"Otter habitats: {otter_habitats}")


def test_species_nlp() -> None:
    """Test the species NLP utilities."""
    print("\n" + "=" * 50)
    print("TESTING SPECIES NLP UTILITIES")
    print("=" * 50)

    # Test anatomical terms
    print("\nAnatomical Terms:")
    wolf_male_terms = get_anatomical_terms("wolf", "male", 2)
    cat_female_terms = get_anatomical_terms("cat", "female", 2)
    print(f"Wolf (male) anatomical terms: {wolf_male_terms}")
    print(f"Cat (female) anatomical terms: {cat_female_terms}")

    # Test prompt enhancement with anatomy
    print("\nEnhance Prompt with Anatomy:")
    base_prompt = "a majestic wolf standing on a cliff"
    enhanced = enhance_prompt_with_anatomy(base_prompt, "wolf", "male", 2)
    print(f"Original: {base_prompt}")
    print(f"Enhanced: {enhanced}")

    # Test species accessories
    print("\nSpecies-specific Accessories:")
    wolf_accessories = get_species_accessories("wolf", "male", 3)
    tiger_accessories = get_species_accessories("tiger", "female", 3)
    print(f"Wolf (male) accessories: {wolf_accessories}")
    print(f"Tiger (female) accessories: {tiger_accessories}")

    # Test species colors
    print("\nSpecies-specific Colors:")
    wolf_colors = get_species_colors("wolf", 3)
    fox_colors = get_species_colors("fox", 3)
    print(f"Wolf colors: {wolf_colors}")
    print(f"Fox colors: {fox_colors}")

    # Test species description
    print("\nSpecies Description Generation:")
    wolf_desc = generate_species_description("wolf", "male")
    fox_desc = generate_species_description("fox", "female")
    dragon_desc = generate_species_description("dragon", "male")
    print(f"Wolf description: {wolf_desc}")
    print(f"Fox description: {fox_desc}")
    print(f"Dragon description: {dragon_desc}")


def test_background_nlp() -> None:
    """Test the background NLP utilities."""
    print("\n" + "=" * 50)
    print("TESTING BACKGROUND NLP UTILITIES")
    print("=" * 50)

    # Test background description
    print("\nBackground Description Generation:")
    forest_desc = generate_background_description("forest", "morning", "misty")
    city_desc = generate_background_description("city", "night", "rainy", "autumn", "mysterious")
    print(f"Forest description: {forest_desc}")
    print(f"City description: {city_desc}")

    # Test scene description
    print("\nScene Description Generation:")
    scene_desc = generate_scene_description(
        "beach",
        "sunset",
        "clear",
        ["a male wolf in a Hawaiian shirt", "a female fox with sunglasses"],
        "playing volleyball",
    )
    print(f"Scene description: {scene_desc}")

    # Test complementary locations
    print("\nComplementary Locations:")
    wolf_locations = get_complementary_locations("wolf")
    otter_locations = get_complementary_locations("otter")
    rabbit_locations = get_complementary_locations("rabbit")
    print(f"Wolf complementary locations: {wolf_locations}")
    print(f"Otter complementary locations: {otter_locations}")
    print(f"Rabbit complementary locations: {rabbit_locations}")

    # Test prompt enhancement with background
    print("\nEnhance Prompt with Background:")
    base_prompt = "a wolf and a fox having a picnic"
    enhanced = enhance_prompt_with_background(base_prompt, "forest", "afternoon", "sunny")
    print(f"Original: {base_prompt}")
    print(f"Enhanced: {enhanced}")


def test_general_enhancer() -> None:
    """Test the general prompt enhancer utilities."""
    print("\n" + "=" * 50)
    print("TESTING GENERAL PROMPT ENHANCER")
    print("=" * 50)

    # Test prompt analysis
    print("\nPrompt Analysis:")
    prompt1 = "a wolf"
    prompt2 = "a wolf, masterpiece, trending on artstation, dramatic lighting"
    analysis1 = analyze_prompt(prompt1)
    analysis2 = analyze_prompt(prompt2)
    print(f"Analysis of '{prompt1}': {analysis1}")
    print(f"Analysis of '{prompt2}': {analysis2}")

    # Test general prompt enhancement
    print("\nGeneral Prompt Enhancement:")
    basic_prompt = "a wolf standing in a forest"
    enhanced1 = enhance_prompt_general(basic_prompt, enhancement_level=1)
    enhanced2 = enhance_prompt_general(basic_prompt, enhancement_level=3)
    print(f"Original: {basic_prompt}")
    print(f"Enhanced (level 1): {enhanced1}")
    print(f"Enhanced (level 3): {enhanced2}")

    # Test enhancement with specific details
    print("\nEnhancement with Specific Details:")
    detailed = enhance_prompt_with_details(
        basic_prompt,
        subject="the wolf's eyes",
        action="howling",
        location="moonlit forest",
        style="realistic",
    )
    print(f"Original: {basic_prompt}")
    print(f"With details: {detailed}")

    # Test prompt simplification
    print("\nPrompt Simplification:")
    redundant_prompt = "wolf, wolf, forest, masterpiece, high quality, forest, dramatic"
    simplified = simplify_prompt(redundant_prompt)
    print(f"Original: {redundant_prompt}")
    print(f"Simplified: {simplified}")

    # Test prompt variations
    print("\nPrompt Variations:")
    variations = create_prompt_variations("a wolf in the forest", 3)
    print(f"Original: a wolf in the forest")
    for i, variation in enumerate(variations, 1):
        print(f"Variation {i}: {variation}")


def test_generators() -> None:
    """Test the prompt generators."""
    print("\n" + "=" * 50)
    print("TESTING PROMPT GENERATORS")
    print("=" * 50)

    # Test FurryPromptGenerator
    print("\nFurry Prompt Generator:")
    # With default random species and gender
    furry_gen1 = FurryPromptGenerator()
    prompt1 = furry_gen1.generate()
    neg_prompt1 = furry_gen1.get_negative_prompt()
    print(f"Default prompt: {prompt1}")
    print(f"Default negative prompt: {neg_prompt1}")

    # With specified species and gender
    furry_gen2 = FurryPromptGenerator(species="fox", gender="male", use_nlp=True)
    prompt2 = furry_gen2.generate()
    print(f"Fox (male) prompt: {prompt2}")

    # Test NsfwFurryPromptGenerator
    print("\nNSFW Furry Prompt Generator:")
    # Level 1 (mild)
    nsfw_gen1 = NsfwFurryPromptGenerator(
        species="wolf", gender="female", explicit_level=1, use_anatomical_terms=True
    )
    nsfw_prompt1 = nsfw_gen1.generate()
    nsfw_neg1 = nsfw_gen1.get_negative_prompt()
    print(f"NSFW Level 1 prompt: {nsfw_prompt1}")
    print(f"NSFW Level 1 negative prompt: {nsfw_neg1}")

    # Level 3 (explicit)
    nsfw_gen3 = NsfwFurryPromptGenerator(
        species="tiger", gender="male", explicit_level=3, use_anatomical_terms=True
    )
    nsfw_prompt3 = nsfw_gen3.generate()
    print(f"NSFW Level 3 prompt: {nsfw_prompt3}")


def main() -> None:
    """Run all tests."""
    print("\nTESTING REFACTORED CRINGEGEN COMPONENTS")
    print("======================================\n")

    test_data_structures()
    test_species_nlp()
    test_background_nlp()
    test_general_enhancer()
    test_generators()

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
