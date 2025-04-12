#!/usr/bin/env python3
"""
Test script to verify the tag enhancement for all the new species.
"""

import sys
import os
from pathlib import Path

# Add the cringegen directory to the path so we can import from it
REPO_ROOT = Path("/home/kade/code/cringe.live")
CRINGEGEN_DIR = REPO_ROOT / "cringegen"
sys.path.append(str(CRINGEGEN_DIR))

try:
    from cringegen.commands.llm_noobai_nsfw import enhance_prompt_with_species_tags
    from cringegen.data import taxonomy

    # Create a basic prompt
    BASE_PROMPT = "anthro, standing, muscular build"

    # New species to test
    NEW_SPECIES = [
        # Dog breeds
        "dobermann",
        "border_collie",
        "golden_retriever",
        "corgi",
        "dachshund",
        # Cat breeds
        "siamese",
        "bengal",
        # Other new species
        "penguin",
        "turtle",
        "ferret",
        "octopus",
        "crow",
        "chinchilla",
    ]

    def main():
        print("Testing tag enhancement for new species")
        print("=" * 50)

        for species in NEW_SPECIES:
            # Get taxonomy
            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species, "unknown")

            # Test for male
            enhanced_male = enhance_prompt_with_species_tags(
                BASE_PROMPT, species, "male", 2
            )
            male_tags = enhanced_male.replace(BASE_PROMPT + ", ", "")

            # Test for female
            enhanced_female = enhance_prompt_with_species_tags(
                BASE_PROMPT, species, "female", 2
            )
            female_tags = enhanced_female.replace(BASE_PROMPT + ", ", "")

            print(f"\nSpecies: {species} (Taxonomy: {taxonomy_key})")
            print(f"  Male tags: {male_tags}")
            print(f"  Female tags: {female_tags}")

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
