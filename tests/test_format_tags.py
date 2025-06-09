#!/usr/bin/env python
"""
Test script to verify tags processor handling of anatomical terms and proper tag formatting
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import cringegen
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from cringegen.utils.tags_processor import TagsProcessor
from cringegen.utils.ollama_api import (
    default_client,
    TAGS_PROCESSOR_AVAILABLE,
    SPECIES_UTILS_AVAILABLE,
)


def format_tag(tag: str) -> str:
    """
    Format a tag to replace underscores with spaces and escape parentheses

    Args:
        tag: Original tag

    Returns:
        Formatted tag
    """
    # Replace underscores with spaces
    formatted = tag.replace("_", " ")

    # Escape parentheses
    formatted = formatted.replace("(", "\\(").replace(")", "\\)")

    return formatted


def test_anatomical_tags():
    """Test species-specific anatomical tags and formatting"""
    tags_file = "dummy_tags_file.txt"
    processor = TagsProcessor(tags_file)
    TAGS_PROCESSOR_AVAILABLE = False
    SPECIES_UTILS_AVAILABLE = False

    # Print import module information
    print("\nModule Information:")
    print("=" * 80)
    print(f"Using TagsProcessor from: {processor.__class__.__module__}")
    print(f"Using OllamaAPIClient from: {default_client.__class__.__module__}")
    print(f"Tags processor available in Ollama API: {TAGS_PROCESSOR_AVAILABLE}")
    print(f"Species utils available in Ollama API: {SPECIES_UTILS_AVAILABLE}")

    # Print module paths to debug import issues
    import cringegen
    from cringegen.prompt_generation.nlp import species_utils
    from cringegen.data import anatomy

    print("\nModule Paths:")
    print(f"cringegen: {cringegen.__file__}")
    print(f"species_utils: {species_utils.__file__}")
    print(f"anatomy: {anatomy.__file__}")

    # Print information about the tags processed
    if processor.loaded:
        print(f"\nSuccessfully loaded tags dictionary with {len(processor.tags_dict)} tags")
        # Print a few sample tags as verification
        print(f"Sample tags: {list(processor.tags_dict.keys())[:5]}")
    else:
        print("\nWarning: Tags dictionary not loaded successfully")
        if tags_file:
            # Check if the specified file exists
            if not os.path.exists(tags_file):
                print(f"  Error: Specified tags file '{tags_file}' does not exist")
            else:
                print(f"  Tags file exists but could not be loaded: '{tags_file}'")
        else:
            # List the directories that were searched
            search_dirs = [
                Path(__file__).parent.parent / "tags.json",
                Path(__file__).parent.parent / "cringegen" / "data" / "tags.json",
            ]
            print("  Searched for tags.json in:")
            for path in search_dirs:
                status = "EXISTS" if path.exists() else "NOT FOUND"
                print(f"    {path}: {status}")

    # Test with various species and explicit levels
    species_to_test = ["wolf", "fox", "dragon", "horse"]
    genders = ["male", "female"]

    # Print table header
    print("\nAnatomical Terms Test")
    print("=" * 80)
    print(f"{'Species':<12} {'Gender':<8} {'Explicit Level':<15} {'Anatomical Terms'}")
    print("-" * 80)

    # Test each species/gender/explicitness combination
    for species in species_to_test:
        for gender in genders:
            for level in range(1, 4):
                from cringegen.prompt_generation.nlp.species_utils import get_anatomical_terms
                terms = get_anatomical_terms(species, gender, level)
                formatted_terms = [format_tag(term) for term in terms]
                print(f"{species:<12} {gender:<8} {level:<15} {', '.join(formatted_terms)}")

    print("\nTag Formatting Test")
    print("=" * 80)

    # Test formatting with underscores and parentheses
    test_tags = [
        "canine_penis",
        "fox_tail",
        "krystal_(star_fox)",
        "renamon_(digimon)",
        "claws_out",
        "holding_hands",
        "red_fox",
        "wolf_(canine)",
        "dragon_(western)",
    ]

    print(f"{'Original Tag':<25} {'Formatted Tag'}")
    print("-" * 80)

    for tag in test_tags:
        formatted = format_tag(tag)
        print(f"{tag:<25} {formatted}")

    # If requested, test with LLM (disabled by default)
    with_llm = False
    if with_llm:
        test_species = "dragon"
        test_gender = "male"
        print(f"Using default test case: {test_gender} {test_species}")
        original_tags = processor.generate_species_specific_tags(
            test_species, test_gender, nsfw=True, explicit_level=3
        )
        print("\nRaw tags from TagsProcessor:")
        for category, tags in original_tags.items():
            if tags:
                formatted_tags = [format_tag(t) for t in tags]
                print(f"  {category.capitalize()}: {', '.join(formatted_tags)}")
        from cringegen.utils.ollama_api import generate_species_specific_tags
        internal_tags = generate_species_specific_tags(
            test_species, test_gender, nsfw=True, explicit_level=3
        )
        print("\nRaw tags from internal generate_species_specific_tags:")
        for category, tags in internal_tags.items():
            if tags:
                print(f"  {category.capitalize()}: {', '.join(tags)}")
        TAGS_PROCESSOR_AVAILABLE = True
        SPECIES_UTILS_AVAILABLE = True
        print(f"\nImportant Status:")
        print(f"- TagsProcessor available in Ollama API: {TAGS_PROCESSOR_AVAILABLE}")
        print(f"- Species utils available in Ollama API: {SPECIES_UTILS_AVAILABLE}")
        if not TAGS_PROCESSOR_AVAILABLE:
            if SPECIES_UTILS_AVAILABLE:
                print(
                    "Ollama API is using the internal generate_species_specific_tags function instead of TagsProcessor."
                )
                print("Tags are still being sent to the LLM, just generated differently.")
            else:
                print(
                    "WARNING: Neither TagsProcessor nor species_utils are available to the Ollama API."
                )
                print("The tags above will not be passed to the LLM in the API call.")
        try:
            print("\nGenerating LLM caption with anatomical terms...")
            caption = default_client.generate_nsfw_caption(
                subject="character",
                species=test_species,
                gender=test_gender,
                nsfw_intensity="explicit",
                temperature=0.7,
                show_thinking=False,
            )
            print("\nGenerated Caption:")
            print("-" * 80)
            print(caption)
            anatomical_terms = set()
            for category, tags in internal_tags.items():
                if category == "nsfw":
                    anatomical_terms.update(tags)
            print("\nAnatomical Terms Usage Analysis:")
            found_terms = []
            missing_terms = []
            for term in anatomical_terms:
                clean_term = term.replace("\\", "")
                if clean_term.lower() in caption.lower():
                    found_terms.append(term)
                else:
                    missing_terms.append(term)
            if found_terms:
                print(f"Terms found in caption: {', '.join(found_terms)}")
            if missing_terms:
                print(f"Terms not found in caption: {', '.join(missing_terms)}")
            percentage = len(found_terms) / max(1, len(anatomical_terms)) * 100
            print(
                f"Anatomical terms usage: {percentage:.1f}% ({len(found_terms)}/{len(anatomical_terms)})"
            )
        except Exception as e:
            print(f"\nError generating caption: {str(e)}")
            print("\nCheck if Ollama is running with: ollama serve")
            print("If it's not running, start it with that command in another terminal")


if __name__ == "__main__":
    test_anatomical_tags()
