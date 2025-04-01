#!/usr/bin/env python3
"""
Test script for NSFW tag categorization with furry Stable Diffusion prompts
Focused on e621-like tags for anatomical features and poses
"""

import sys
import os
import json
from pprint import pprint

# Add cringegen directory to the Python path
sys.path.append(os.path.join(os.getcwd(), "cringegen"))

try:
    from cringegen.prompt_generation.nlp.tag_converter import (
        categorize_tags,
        tags_to_text,
        natural_tags_to_text,
    )

    print("Successfully imported tag categorization modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def print_categorization(tags, category_name=""):
    """
    Print the categorization results for a set of tags

    Args:
        tags: List of tags to categorize
        category_name: Name of the test category for organization
    """
    print(f"\n=== Testing {category_name} Tags: {tags} ===")
    categories = categorize_tags(tags)

    # Print only non-empty categories for readability
    non_empty = {k: v for k, v in categories.items() if v}
    print(json.dumps(non_empty, indent=2))

    # Generate text representations
    descriptive = tags_to_text(tags, style="descriptive")
    concise = tags_to_text(tags, style="concise")
    detailed = tags_to_text(tags, style="detailed")
    natural = natural_tags_to_text(tags)

    print(f"\nDescriptive: {descriptive}")
    print(f"Concise: {concise}")
    print(f"Detailed: {detailed}")
    print(f"Natural: {natural}")

    print("\n" + "-" * 80)


def test_nsfw_presenting_tags():
    """Test tags related to presenting poses"""
    print("\n===== TESTING PRESENTING POSE TAGS =====\n")

    # Basic presenting
    print_categorization(["presenting"], "Basic Presenting")

    # Specific presenting poses
    print_categorization(["presenting hindquarters"], "Presenting Hindquarters")

    print_categorization(["presenting sheath"], "Presenting Sheath")

    # With species
    print_categorization(["anthro wolf", "presenting"], "Species + Presenting")

    # Complex presenting scenarios
    print_categorization(
        ["anthro fox", "presenting hindquarters", "looking back at viewer"],
        "Complex Presenting",
    )


def test_genital_tags():
    """Test genital/anatomical tags"""
    print("\n===== TESTING GENITAL TAGS =====\n")

    # Basic anatomical terms
    print_categorization(["canine genitalia"], "Canine Genitalia")

    print_categorization(["equine genitalia"], "Equine Genitalia")

    # Species-specific terms
    print_categorization(["knot", "sheath", "canine penis"], "Canine Male Anatomy")

    print_categorization(
        ["flared penis", "horse cock", "equine penis"], "Equine Male Anatomy"
    )

    # Female anatomy
    print_categorization(["canine pussy", "animal pussy"], "Canine Female Anatomy")

    # Mixed with species
    print_categorization(
        ["anthro dragon", "genital slit", "hemipenes"], "Reptile Anatomy + Species"
    )


def test_viewer_interaction_tags():
    """Test tags related to viewer interaction"""
    print("\n===== TESTING VIEWER INTERACTION TAGS =====\n")

    print_categorization(["looking at viewer"], "Looking At Viewer")

    print_categorization(["looking back at viewer"], "Looking Back At Viewer")

    print_categorization(["eye contact"], "Eye Contact")

    print_categorization(
        ["anthro wolf", "looking at viewer", "smiling"], "Wolf Looking At Viewer"
    )


def test_nsfw_rating_tags():
    """Test rating/content tags"""
    print("\n===== TESTING NSFW RATING TAGS =====\n")

    print_categorization(["nsfw"], "NSFW")

    print_categorization(["explicit"], "Explicit")

    print_categorization(["rating:explicit"], "Rating:Explicit")

    print_categorization(["questionable content"], "Questionable")


def test_complex_nsfw_prompts():
    """Test complex NSFW prompts combining multiple tag types"""
    print("\n===== TESTING COMPLEX NSFW PROMPTS =====\n")

    # Full NSFW prompt 1
    print_categorization(
        [
            "masterpiece",
            "high quality",
            "detailed",
            "anthro male wolf",
            "muscular",
            "red fur",
            "presenting",
            "canine genitalia",
            "knot",
            "sheath",
            "looking back at viewer",
            "bedroom",
            "evening",
            "nsfw",
            "explicit",
        ],
        "Full Male NSFW Prompt",
    )

    # Full NSFW prompt 2
    print_categorization(
        [
            "best quality",
            "highly detailed",
            "anthro female fox",
            "white fur",
            "blue eyes",
            "lying on bed",
            "spread legs",
            "animal pussy",
            "bedroom",
            "soft lighting",
            "looking at viewer",
            "nsfw",
            "explicit",
        ],
        "Full Female NSFW Prompt",
    )

    # Mixed species prompt
    print_categorization(
        [
            "masterpiece",
            "detailed",
            "anthro horse",
            "anthro wolf",
            "interspecies",
            "equine penis",
            "canine pussy",
            "mating",
            "forest",
            "night",
            "moonlight",
            "nsfw",
            "explicit",
        ],
        "Mixed Species NSFW Prompt",
    )


def main():
    """Main function to run all tests"""
    print("=== NSFW TAG CATEGORIZATION TESTS ===")
    print("Testing furry NSFW tag categorization for Stable Diffusion\n")

    # Run all test categories
    test_nsfw_presenting_tags()
    test_genital_tags()
    test_viewer_interaction_tags()
    test_nsfw_rating_tags()
    test_complex_nsfw_prompts()

    print("\n===== SUMMARY OF ENHANCEMENTS =====\n")
    print("The tag converter has been enhanced with:")
    print(
        "1. New categories for anatomical features, NSFW ratings, and viewer interactions"
    )
    print("2. Comprehensive lists of anatomical terms for various species")
    print("3. Better categorization of presenting poses")
    print(
        "4. Recognition of e621-style tags like 'looking at viewer', 'presenting', etc."
    )
    print("5. Enhanced natural language representation of NSFW content")
    print(
        "\nThese improvements enable more accurate tagging and description of furry NSFW content."
    )


if __name__ == "__main__":
    main()
