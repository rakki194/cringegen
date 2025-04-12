#!/usr/bin/env python3
"""
Test script to verify that the newly added species are correctly mapped to their taxonomy groups.
"""

import sys
import os
from pathlib import Path
import random

# Add the cringegen directory to the path so we can import from it
REPO_ROOT = Path("/home/kade/code/cringe.live")
CRINGEGEN_DIR = REPO_ROOT / "cringegen"
sys.path.append(str(CRINGEGEN_DIR))

try:
    from cringegen.data import taxonomy
    from cringegen.prompt_generation.nlp.species_utils import get_anatomical_terms
    from cringegen.commands.llm_noobai_nsfw import enhance_prompt_with_species_tags

    def test_dog_breeds():
        """Test that all dog breeds are correctly mapped to 'canine'."""
        print("\n=== Testing Dog Breeds ===\n")

        dog_breeds = [
            "border_collie",
            "german_shepherd",
            "dobermann",
            "labrador",
            "golden_retriever",
            "dalmatian",
            "corgi",
            "shiba_inu",
            "poodle",
            "beagle",
            "dachshund",
            "pitbull",
            "boxer",
            "australian_shepherd",
            "malamute",
            "collie",
            "greyhound",
            "rottweiler",
            "pomeranian",
            "bulldog",
            "sheepdog",
            "terrier",
        ]

        for breed in dog_breeds:
            actual_taxonomy = taxonomy.SPECIES_TAXONOMY.get(breed)
            if actual_taxonomy == "canine":
                print(f"✓ {breed:<20} → {actual_taxonomy}")
            else:
                print(
                    f"✗ {breed:<20} → {actual_taxonomy or 'MISSING'} (expected: canine)"
                )

    def test_new_species_anatomical_terms():
        """Test that new species get appropriate anatomical terms."""
        print("\n=== Testing Anatomical Terms for New Species ===\n")

        test_cases = [
            # Dog breeds
            ("border_collie", "male", 2),
            ("dobermann", "male", 3),
            ("corgi", "female", 2),
            # Cat breeds
            ("siamese", "female", 2),
            ("bengal", "male", 3),
            # Other new species
            ("ferret", "male", 2),
            ("penguin", "male", 2),
            ("turtle", "female", 2),
        ]

        for species, gender, explicit_level in test_cases:
            terms = get_anatomical_terms(species, gender, explicit_level)

            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species.lower(), "accessory")
            print(
                f"{species:<20} {gender:<6} (level {explicit_level}) → Taxonomy: {taxonomy_key:<15} Terms: {', '.join(terms)}"
            )

    def test_prompt_enhancement_for_new_species():
        """Test that prompts are correctly enhanced with tags for new species."""
        print("\n=== Testing Prompt Enhancement for New Species ===\n")

        BASE_PROMPT = "anthro, muscular, standing, dimly lit room"

        test_cases = [
            ("border_collie", "male", "explicit"),
            ("dobermann", "male", "hardcore"),
            ("siamese", "female", "explicit"),
            ("penguin", "male", "explicit"),
            ("ferret", "male", "hardcore"),
            ("octopus", "male", "explicit"),
        ]

        for species, gender, intensity in test_cases:
            explicit_level = 2 if intensity == "explicit" else 3

            # Get enhanced prompt
            enhanced = enhance_prompt_with_species_tags(
                BASE_PROMPT, species, gender, explicit_level
            )

            # Calculate what tags were added
            tags_added = enhanced.replace(BASE_PROMPT + ", ", "")

            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species.lower(), "accessory")
            print(
                f"{species:<20} {gender:<6} {intensity:<9} → Taxonomy: {taxonomy_key:<15} Tags: {tags_added}"
            )

    def test_random_new_species():
        """Test random selection of new species from the taxonomy dictionary."""
        print("\n=== Testing Random Selection of New Species ===\n")

        new_species = [
            "dobermann",
            "border_collie",
            "siamese",
            "bengal",
            "penguin",
            "ferret",
            "turtle",
            "octopus",
            "weasel",
            "corgi",
        ]

        # Try a random subset of the new species
        random_species = random.sample(new_species, min(5, len(new_species)))

        BASE_PROMPT = "anthro, muscular, standing, dimly lit room"

        for species in random_species:
            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species.lower())

            for gender in ["male", "female"]:
                # Get terms for explicit level 2
                terms = get_anatomical_terms(species, gender, 2)

                # Check enhanced prompt
                enhanced = enhance_prompt_with_species_tags(
                    BASE_PROMPT, species, gender, 2
                )

                tags_added = enhanced.replace(BASE_PROMPT + ", ", "")

                print(
                    f"{species:<20} {gender:<6} → Taxonomy: {taxonomy_key:<15} Terms: {', '.join(terms)}"
                )
                print(f"  Enhanced with: {tags_added}")

    # Run all tests
    if __name__ == "__main__":
        print("TESTING NEW SPECIES IN TAXONOMY AND TAG PROCESSING")
        print("=" * 70)

        test_dog_breeds()
        test_new_species_anatomical_terms()
        test_prompt_enhancement_for_new_species()
        test_random_new_species()

        print("\nAll tests completed.")

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
