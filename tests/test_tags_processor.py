#!/usr/bin/env python
"""
Test script for the tags processor
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import cringegen
sys.path.append(str(Path(__file__).parent.parent))

from cringegen.utils.tags_processor import TagsProcessor
from cringegen.utils.ollama_api import default_client


def test_tags_processor():
    """Test the tags processor"""
    tags_file = "dummy_tags_file.txt"
    processor = TagsProcessor(tags_file)
    species = "wolf"
    gender = "male"
    nsfw = True
    explicit_level = 2
    # Generate species-specific tags
    species_tags = processor.generate_species_specific_tags(
        species=species, gender=gender, nsfw=nsfw, explicit_level=explicit_level
    )

    # Print the results
    print(f"\nSpecies-specific tags for {species} ({gender}):")
    print("==================================================")

    for category, tags in species_tags.items():
        if tags:
            print(f"\n{category.capitalize()}:")
            print(", ".join(tags))

    with_llm = False
    if with_llm:
        print("\n\nGenerating LLM caption...")
        print("==========================")

        caption = default_client.generate_nsfw_caption(
            subject="character",
            species=species,
            gender=gender,
            nsfw_intensity="explicit" if explicit_level > 2 else "moderate",
            temperature=0.7,
            show_thinking=False,
        )

        print(caption)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the tags processor")
    parser.add_argument("--tags-file", help="Path to tags.json file")
    parser.add_argument("--species", default="fox", help="Species of the character")
    parser.add_argument("--gender", default="female", help="Gender of the character")
    parser.add_argument("--nsfw", action="store_true", help="Include NSFW tags")
    parser.add_argument("--explicit-level", type=int, default=1, help="Explicit level (1-3)")
    parser.add_argument("--with-llm", action="store_true", help="Test with LLM caption generation")

    args = parser.parse_args()
    test_tags_processor()


if __name__ == "__main__":
    main()
