#!/usr/bin/env python3
"""
Tests for entity detection with a focus on furry characters.
This script demonstrates the enhanced entity detection for anthropomorphic characters,
including famous characters from games, anime, and other media.
"""

import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cringegen.prompt_generation.nlp import extract_entities


def separator(title):
    """Print a separator with title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def test_basic_entity_detection():
    """Test basic entity detection for anthro characters"""
    separator("BASIC ANTHRO ENTITY DETECTION")

    test_texts = [
        "An anthro fox warrior with a blue sword battling in a fantasy castle.",
        "A digital painting of a fox girl wearing a school uniform.",
        "A 3D render of an anthropomorphic wolf in cyberpunk clothing.",
        "A cartoon drawing of a cat girl with pink hair.",
        "A realistic painting of a feral wolf howling at the moon.",
    ]

    for i, text in enumerate(test_texts):
        print(f"\nText #{i+1}:")
        print(f'"{text}"\n')

        entities = extract_entities(text)
        print("Detected Entities:")
        for category, items in entities.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")


def test_famous_character_detection():
    """Test detection of famous anthro characters"""
    separator("FAMOUS ANTHRO CHARACTER DETECTION")

    test_texts = [
        "Blaidd the Half-Wolf is a character from Elden Ring who serves Ranni.",
        "Maliketh the Black Blade is a fearsome boss in Elden Ring.",
        "Sonic the Hedgehog runs fast and battles Dr. Robotnik.",
        "Nick Wilde and Judy Hopps are the main characters in Zootopia.",
        "Legoshi from Beastars is a wolf struggling with his carnivore instincts.",
        "Robin Hood the fox archer is a classic Disney character.",
        "Rocket Raccoon is a member of the Guardians of the Galaxy.",
        "Blaidd fights with great skill while Maliketh uses powerful magic.",
        "A painting showing Sonic, Tails, and Knuckles running through Green Hill Zone.",
        "A crossover artwork featuring Legoshi, Nick Wilde, and Robin Hood.",
    ]

    for i, text in enumerate(test_texts):
        print(f"\nText #{i+1}:")
        print(f'"{text}"\n')

        entities = extract_entities(text)
        print("Detected Entities:")
        for category, items in entities.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")


def test_mixed_entity_types():
    """Test detection of mixed entity types including anthro characters"""
    separator("MIXED ENTITY TYPES DETECTION")

    test_texts = [
        "Judy Hopps and Nick Wilde investigate a case in Zootopia while Sonic watches from afar.",
        "In the game, the player meets Blaidd in the Mistwood before finding Maliketh in Crumbling Farum Azula.",
        "The artist drew Legoshi from Beastars alongside a generic anthro wolf character.",
        "Robin Hood the fox archer meets with Little John and Maid Marian in Sherwood Forest.",
        "Rocket Raccoon uses advanced technology while a regular raccoon eats trash nearby.",
    ]

    for i, text in enumerate(test_texts):
        print(f"\nText #{i+1}:")
        print(f'"{text}"\n')

        entities = extract_entities(text)
        print("Detected Entities:")
        for category, items in entities.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")


def run_all_tests():
    """Run all entity detection tests"""
    separator("FURRY ENTITY DETECTION TESTS")

    print(
        "This test will demonstrate the enhanced entity detection for furry characters, with special focus on:"
    )
    print("1. Basic detection of anthro vs. feral characters")
    print("2. Detection of famous anthro characters from games, anime, and other media")
    print("3. Mixed entity types including humans, places, and anthro characters")

    # Run all tests
    test_basic_entity_detection()
    test_famous_character_detection()
    test_mixed_entity_types()

    separator("ALL TESTS COMPLETED")


if __name__ == "__main__":
    run_all_tests()
