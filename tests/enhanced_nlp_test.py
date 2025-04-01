#!/usr/bin/env python3
"""
Test enhanced NLP features:
1. Improved entity detection
2. Natural caption generation
"""

import sys
import os
from pprint import pprint

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cringegen.prompt_generation.nlp import (
    categorize_tags,
    tags_to_text,
    natural_tags_to_text,
    extract_entities,
)


def test_entity_detection():
    """Test improved entity detection with various inputs"""
    print("\n===== TESTING IMPROVED ENTITY DETECTION =====\n")

    test_cases = [
        "John Smith visited New York City and met with Google executives in August.",
        "The red fox jumped over the brown dog in the forest near London.",
        "Apple released a new iPhone at their headquarters in California.",
        "Neil Armstrong was the first person to walk on the Moon in July 1969.",
        "The Eiffel Tower in Paris, France is one of the most famous landmarks in Europe.",
        "Leonardo da Vinci painted the Mona Lisa, which is displayed at the Louvre Museum.",
        "Mount Everest, located in the Himalayas, is the tallest mountain on Earth.",
        "Amazon, Microsoft, and Google are some of the largest technology companies.",
        "The character Harry Potter was created by J.K. Rowling in her popular book series.",
        "During the COVID-19 pandemic, the World Health Organization issued health guidelines.",
    ]

    for i, text in enumerate(test_cases):
        print(f"Test case #{i+1}:")
        print(f"Text: {text}")

        entities = extract_entities(text)
        print("Detected entities:")

        if not entities or all(len(entity_list) == 0 for entity_list in entities.values()):
            print("  No entities detected! This might indicate a problem.")
        else:
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"  {entity_type.capitalize()}: {', '.join(entity_list)}")

        print("\n" + "-" * 50 + "\n")


def test_natural_caption_generation():
    """Test enhanced natural caption generation"""
    print("\n===== TESTING NATURAL CAPTION GENERATION =====\n")

    # Test sets of tags
    test_tag_sets = [
        ["masterpiece", "detailed", "fox", "red fur", "forest", "digital art"],
        [
            "best quality",
            "highly detailed",
            "wolf",
            "snowy mountains",
            "night scene",
            "blue eyes",
            "oil painting",
        ],
        [
            "masterpiece",
            "anthropomorphic",
            "rabbit",
            "wearing blue dress",
            "garden",
            "flowers",
            "watercolor",
        ],
        [
            "high quality",
            "detailed",
            "tiger",
            "orange fur",
            "black stripes",
            "jungle",
            "sunset",
            "realistic",
        ],
        [
            "masterpiece",
            "fox girl",
            "red hair",
            "green eyes",
            "smiling",
            "beach",
            "summer",
            "anime style",
        ],
    ]

    for i, tags in enumerate(test_tag_sets):
        print(f"Test set #{i+1}:")
        print(f"Tags: {', '.join(tags)}")

        print("\nCategorized tags:")
        categories = categorize_tags(tags)
        for category, items in categories.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")

        print("\nStandard tags_to_text output (descriptive style):")
        standard_text = tags_to_text(tags, style="descriptive")
        print(f'  "{standard_text}"')

        print("\nEnhanced natural caption:")
        natural_text = natural_tags_to_text(tags)
        print(f'  "{natural_text}"')

        print("\n" + "-" * 50 + "\n")


def test_complex_prompts():
    """Test more complex prompt scenarios"""
    print("\n===== TESTING COMPLEX PROMPTS =====\n")

    complex_tag_sets = [
        [
            "masterpiece",
            "best quality",
            "detailed",
            "anthro",
            "female fox",
            "red fur",
            "blue eyes",
            "wearing casual outfit",
            "smiling",
            "sitting on bench",
            "city park",
            "sunset",
            "digital painting",
        ],
        [
            "best quality",
            "ultra detailed",
            "4k",
            "wolf",
            "male",
            "warrior",
            "gray fur",
            "battle armor",
            "sword",
            "standing",
            "medieval castle",
            "foggy",
            "dynamic pose",
            "fantasy",
            "dramatic lighting",
        ],
        [
            "masterpiece",
            "high quality",
            "detailed",
            "deer",
            "anthro",
            "female",
            "brown fur",
            "white spots",
            "green dress",
            "forest clearing",
            "flowers",
            "spring",
            "soft lighting",
            "relaxed pose",
            "peaceful",
        ],
    ]

    for i, tags in enumerate(complex_tag_sets):
        print(f"Complex test set #{i+1}:")
        print(f"Tags: {', '.join(tags)}")

        print("\nStandard styles output:")
        print(f"  Concise: \"{tags_to_text(tags, style='concise')}\"")
        print(f"  Descriptive: \"{tags_to_text(tags, style='descriptive')}\"")
        print(f"  Detailed: \"{tags_to_text(tags, style='detailed')}\"")

        print("\nEnhanced natural caption:")
        natural_text = natural_tags_to_text(tags)
        print(f'  "{natural_text}"')

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    print("=====================================")
    print("ENHANCED NLP FEATURES TEST")
    print("=====================================")

    # Test improved entity detection
    test_entity_detection()

    # Test natural caption generation
    test_natural_caption_generation()

    # Test complex prompts
    test_complex_prompts()

    print("\n=====================================")
    print("ALL TESTS COMPLETED")
    print("=====================================")
