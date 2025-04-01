#!/usr/bin/env python3
"""
Demo for the data structures and utilities in cringegen.

This script demonstrates:
1. Species data structures and taxonomy
2. Anime character data structures and utilities
3. Game character data structures and utilities
4. Art style and media data structures
5. Tag classification and enhancement
6. Tag suggestions and analysis
"""

import sys
import os
from pprint import pprint

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cringegen.prompt_generation.nlp import (
    # Data utilities
    get_all_species,
    get_all_anime_character_types,
    get_all_game_character_types,
    get_all_art_styles,
    get_taxonomy_group,
    get_anatomical_terms,
    get_species_accessories,
    get_species_colors,
    get_anime_character_details,
    get_game_character_details,
    get_art_style_details,
    classify_tag,
    enhance_tag_description,
    get_compatible_accessories,
    get_tag_suggestions,
    analyze_tags_with_data_structures,
    # NLTK utilities
    extract_entities,
)


def separator(title):
    """Print a separator with title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def demo_species_data():
    """Demonstrate species data structures and taxonomy"""
    separator("SPECIES DATA STRUCTURES DEMO")

    # Get all species
    all_species = get_all_species()
    print(f"Total number of species: {len(all_species)}")
    print("Sample species:")
    for species in all_species[:10]:  # Show first 10
        print(f"  - {species}")

    # Get taxonomy for specific species
    test_species = ["fox", "cat", "wolf", "dragon", "otter"]
    print("\nTaxonomy groups:")
    for species in test_species:
        taxonomy = get_taxonomy_group(species)
        print(f"  {species}: {taxonomy}")

    # Get anatomical terms
    print("\nAnatomical terms for a male fox:")
    pprint(get_anatomical_terms("fox", "male"))

    print("\nAnatomical terms for a female wolf:")
    pprint(get_anatomical_terms("wolf", "female"))

    # Get species accessories
    print("\nAccessories for a male cat:")
    pprint(get_species_accessories("cat", "male"))

    print("\nAccessories for a female dragon:")
    pprint(get_species_accessories("dragon", "female"))

    # Get species colors
    print("\nSuggested colors for a fox:")
    pprint(get_species_colors("fox"))

    print("\nSuggested colors for a wolf:")
    pprint(get_species_colors("wolf"))


def demo_anime_data():
    """Demonstrate anime character data structures and utilities"""
    separator("ANIME DATA STRUCTURES DEMO")

    # Get all anime character types (sample)
    all_types = get_all_anime_character_types()
    print(f"Total number of anime character types: {len(all_types)}")
    print("Sample character types:")
    for character_type in all_types[:15]:  # Show first 15
        print(f"  - {character_type}")

    # Get details for specific anime character types
    test_characters = ["cat girl", "kitsune", "magical girl", "maid", "tsundere"]
    for character in test_characters:
        print(f"\nDetails for '{character}':")
        details = get_anime_character_details(character)
        pprint(details)


def demo_game_data():
    """Demonstrate game character data structures and utilities"""
    separator("GAME DATA STRUCTURES DEMO")

    # Get all game character types (sample)
    all_types = get_all_game_character_types()
    print(f"Total number of game character types: {len(all_types)}")
    print("Sample character types:")
    for character_type in all_types[:15]:  # Show first 15
        print(f"  - {character_type}")

    # Get details for specific game character types
    test_characters = ["warrior", "mage", "rogue", "paladin", "ranger"]
    for character in test_characters:
        print(f"\nDetails for '{character}':")
        details = get_game_character_details(character)
        pprint(details)


def demo_art_style_data():
    """Demonstrate art style and media data structures"""
    separator("ART STYLE DATA STRUCTURES DEMO")

    # Get all art styles (sample)
    all_styles = get_all_art_styles()
    print(f"Total number of art styles: {len(all_styles)}")
    print("Sample art styles:")
    for style in all_styles[:15]:  # Show first 15
        print(f"  - {style}")

    # Get details for specific art styles
    test_styles = ["digital painting", "anime", "oil painting", "pixel art", "cyberpunk"]
    for style in test_styles:
        print(f"\nDetails for '{style}':")
        details = get_art_style_details(style)
        pprint(details)


def demo_tag_classification():
    """Demonstrate tag classification and enhancement"""
    separator("TAG CLASSIFICATION DEMO")

    # Classify various tags
    test_tags = [
        "fox",
        "cat girl",
        "warrior",
        "digital painting",
        "anthro wolf",
        "feral tiger",
        "anime style",
        "pixel art",
        "masterpiece",
        "detailed",
        "forest",
        "school uniform",
    ]

    for tag in test_tags:
        print(f"\nClassification for '{tag}':")
        classification = classify_tag(tag)

        # Print main classification details
        print(f"  Category: {classification['category']}")
        if classification["is_species"]:
            print(f"  Species: Yes (Taxonomy: {classification['taxonomy_group']})")
        if classification["is_anime_character"]:
            print(f"  Anime Character: Yes")
        if classification["is_game_character"]:
            print(f"  Game Character: Yes")
        if classification["is_art_style"]:
            print(f"  Art Style: Yes")

        # Enhanced description
        print(f"\n  Enhanced description: {enhance_tag_description(tag)}")

        # With gender for species
        if classification["is_species"]:
            print(f"  Enhanced with gender: {enhance_tag_description(tag, 'female')}")


def demo_tag_suggestions():
    """Demonstrate tag suggestions and compatibility"""
    separator("TAG SUGGESTIONS DEMO")

    # Get tag suggestions for various base tags
    test_tags = ["fox", "cat girl", "warrior", "digital painting", "anthro"]

    for tag in test_tags:
        print(f"\nSuggestions for '{tag}':")
        suggestions = get_tag_suggestions(tag, 8)
        print(f"  {', '.join(suggestions)}")

    # Get compatible accessories
    print("\nCompatible accessories for a fox as a warrior:")
    accessories = get_compatible_accessories("fox", "male", "warrior")
    print(f"  {', '.join(accessories)}")

    print("\nCompatible accessories for a cat as a cat girl:")
    accessories = get_compatible_accessories("cat", "female", "cat girl")
    print(f"  {', '.join(accessories)}")


def demo_tag_analysis():
    """Demonstrate tag analysis with data structures"""
    separator("TAG ANALYSIS DEMO")

    # Analyze various tag sets
    test_tag_sets = [
        # Anthro/furry set
        ["masterpiece", "detailed", "anthro", "fox", "red fur", "forest", "sunset", "digital art"],
        # Anime set
        [
            "high quality",
            "anime style",
            "cat girl",
            "pink hair",
            "school uniform",
            "cherry blossoms",
        ],
        # Game character set
        ["detailed", "warrior", "plate armor", "sword", "shield", "fantasy", "battle"],
        # Mixed set
        ["masterpiece", "fox girl", "kimono", "forest", "digital painting", "studio ghibli style"],
    ]

    for i, tags in enumerate(test_tag_sets):
        print(f"\nAnalysis for tag set #{i+1}:")
        print(f"Tags: {', '.join(tags)}")

        analysis = analyze_tags_with_data_structures(tags)

        print(f"\n  Dominant category: {analysis['dominant_category']}")

        if analysis["species"]:
            print(f"  Species: {', '.join(analysis['species'])}")
        if analysis["taxonomy_groups"]:
            print(f"  Taxonomy groups: {', '.join(analysis['taxonomy_groups'])}")
        if analysis["anime_characters"]:
            print(f"  Anime characters: {', '.join(analysis['anime_characters'])}")
        if analysis["game_characters"]:
            print(f"  Game characters: {', '.join(analysis['game_characters'])}")
        if analysis["art_styles"]:
            print(f"  Art styles: {', '.join(analysis['art_styles'])}")

        print(f"  Anthro content: {'Yes' if analysis['has_anthro'] else 'No'}")
        print(f"  Feral content: {'Yes' if analysis['has_feral'] else 'No'}")
        print(f"  Anime content: {'Yes' if analysis['has_anime'] else 'No'}")
        print(f"  Game content: {'Yes' if analysis['has_game'] else 'No'}")

        if analysis["suggested_additions"]:
            print(f"  Suggested additions: {', '.join(analysis['suggested_additions'])}")


def demo_entity_detection():
    """Demonstrate entity detection with our data structures"""
    separator("ENTITY DETECTION DEMO")

    # Test with various texts
    test_texts = [
        "An anthro fox warrior with a blue sword battling alongside human allies in a fantasy castle.",
        "A cute anime-style cat girl with pink hair wearing a school uniform under cherry blossoms.",
        "A digital painting of a feral wolf howling at the moon in a snowy forest.",
        "A cyberpunk cityscape with robot citizens, anthro raccoons, and human hackers at night.",
        "A Studio Ghibli style fox girl in a traditional kimono exploring an enchanted forest.",
    ]

    for i, text in enumerate(test_texts):
        print(f"\nText #{i+1}:")
        print(f'"{text}"\n')

        # Extract entities
        entities = extract_entities(text)
        print("Detected entities:")
        for category, items in entities.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")

        # Convert to potential tags
        tags = text.replace(".", "").replace(",", "").split()
        analysis = analyze_tags_with_data_structures(tags)

        print("\nTag-based analysis:")
        if analysis["dominant_category"]:
            print(f"  Dominant category: {analysis['dominant_category']}")
        if analysis["suggested_additions"]:
            print(f"  Suggested additions: {', '.join(analysis['suggested_additions'])}")


def run_all_demos():
    """Run all demo functions"""
    print("CRINGEGEN DATA STRUCTURES AND UTILITIES DEMO")
    print("This demo will showcase the various data structures and utilities")
    print("available in cringegen for handling different types of content.")

    # Run all demos
    demo_species_data()
    demo_anime_data()
    demo_game_data()
    demo_art_style_data()
    demo_tag_classification()
    demo_tag_suggestions()
    demo_tag_analysis()
    demo_entity_detection()

    separator("DEMO COMPLETED")


if __name__ == "__main__":
    run_all_demos()
