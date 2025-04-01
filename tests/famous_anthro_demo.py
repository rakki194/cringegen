#!/usr/bin/env python3
"""
Demo script for testing famous anthro character detection in the cringegen toolkit.
This script allows users to explore the database of famous furry characters and test
the entity detection on their own text.
"""

import sys
import os
from pprint import pprint

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cringegen.prompt_generation.nlp import extract_entities
from cringegen.data import characters

# Map old variables to new module structure
FAMOUS_FURRY_CHARACTERS = characters.FAMOUS_FURRY_CHARACTERS
ALL_ANTHRO_CHARACTERS = characters.ALL_ANTHRO_CHARACTERS
CHARACTER_TO_FULL_NAME = characters.CHARACTER_TO_FULL_NAME


def print_separator(title):
    """Print a separator with title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def display_available_characters():
    """Display all available anthro characters in the database"""
    print_separator("FAMOUS ANTHRO CHARACTERS DATABASE")

    print(f"Total characters in database: {len(ALL_ANTHRO_CHARACTERS)}")
    print("Characters by category:\n")

    for category, sources in FAMOUS_FURRY_CHARACTERS.items():
        print(f"--- {category.upper()} ---")
        for source, characters in sources.items():
            # Display only full names, skip shortened versions
            full_names = (
                [c for i, c in enumerate(characters) if i % 2 == 0]
                if len(characters) > 1
                else characters
            )
            print(f"  {source.replace('_', ' ').title()}: {len(full_names)} characters")
            # Print a sample of characters (first 3)
            if full_names:
                print(
                    f"    Sample: {', '.join(full_names[:3])}"
                    + (", ..." if len(full_names) > 3 else "")
                )
        print()


def test_entity_detection():
    """Interactive test for entity detection"""
    print_separator("ANTHRO CHARACTER DETECTION TEST")

    print("This tool will detect famous anthro characters in your text.")
    print("Enter some text that includes characters like Blaidd, Sonic, Nick Wilde, etc.")
    print("Enter 'quit' to exit.")
    print()

    while True:
        text = input("Enter text to analyze: ")
        if text.lower() in ["quit", "exit", "q"]:
            break

        if not text:
            continue

        print("\nDetected entities:")
        entities = extract_entities(text)

        for category, items in entities.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")
        print()


def search_character():
    """Search for specific characters in the database"""
    print_separator("CHARACTER SEARCH")

    print("This tool allows you to search for characters in the database.")
    print("Enter a search term (or 'quit' to exit).")
    print()

    while True:
        search = input("Search for character: ")
        if search.lower() in ["quit", "exit", "q"]:
            break

        if not search or len(search) < 2:
            print("Please enter a valid search term (at least 2 characters).")
            continue

        results = []
        for character in ALL_ANTHRO_CHARACTERS:
            if search.lower() in character.lower():
                # If this is a short form of a character, find the full name
                full_name = CHARACTER_TO_FULL_NAME.get(character, character)
                if full_name not in results:
                    results.append(full_name)

        if results:
            print(f"\nFound {len(results)} matching characters:")
            for character in results:
                print(f"  - {character}")
        else:
            print("\nNo characters found matching that search term.")
        print()


def main():
    """Main function with menu"""
    print_separator("FAMOUS ANTHRO CHARACTER DEMO")

    print("This demo showcases the anthro character detection in cringegen.")
    print("Choose an option:")

    while True:
        print("\n1. View available characters")
        print("2. Test entity detection on your text")
        print("3. Search for characters")
        print("4. Quit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            display_available_characters()
        elif choice == "2":
            test_entity_detection()
        elif choice == "3":
            search_character()
        elif choice == "4":
            print("\nExiting demo. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    main()
