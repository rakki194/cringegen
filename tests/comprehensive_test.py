#!/usr/bin/env python3
"""
Comprehensive test for species taxonomy and anatomical tag generation.
This script tests that species are correctly mapped to taxonomies and that
anatomical tags are appropriate for each species/gender combination.
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
    # Import the necessary modules
    from cringegen.commands.llm_noobai_nsfw import (
        enhance_prompt_with_species_tags,
        get_explicit_level,
    )
    from cringegen.prompt_generation.nlp.species_utils import get_anatomical_terms
    from cringegen.data import taxonomy

    # Set up a basic prompt for testing
    BASE_PROMPT = "anthro, muscular, standing, dimly lit room"

    def test_species_taxonomy_mapping():
        """Test that species are correctly mapped to taxonomies."""
        print("\n=== Testing Species Taxonomy Mapping ===\n")

        # Test cases: list of tuples (species, expected_taxonomy)
        test_cases = [
            # Canines
            ("fox", "canine"),
            ("wolf", "canine"),
            ("dog", "canine"),
            ("coyote", "canine"),
            # Felines
            ("cat", "feline"),
            ("tiger", "feline"),
            ("lion", "feline"),
            ("leopard", "feline"),
            # Equines
            ("horse", "equine"),
            ("pony", "equine"),
            ("zebra", "equine"),
            # Other mammals
            ("deer", "deer"),
            ("raccoon", "procyonid"),
            ("bear", "ursid_humanoid"),
            ("rabbit", "lagomorph"),
            # Reptiles, Avians, Fantasy
            (
                "dragon",
                "reptile",
            ),  # Note: dragon maps to both reptile and dragon in the taxonomy
            ("lizard", "reptile"),
            ("snake", "reptile"),
            ("bird", "avian"),
            ("sergal", "sergal"),
            ("protogen", "protogen"),
        ]

        # Check each mapping
        for species, expected_taxonomy in test_cases:
            actual_taxonomy = taxonomy.SPECIES_TAXONOMY.get(species.lower())

            if actual_taxonomy == expected_taxonomy:
                print(f"✓ {species:<10} → {actual_taxonomy}")
            else:
                print(
                    f"✗ {species:<10} → {actual_taxonomy} (expected: {expected_taxonomy})"
                )

    def test_anatomical_terms():
        """Test that anatomical terms are appropriate for each species/gender combination."""
        print("\n=== Testing Anatomical Terms ===\n")

        # Test cases: list of tuples (species, gender, explicit_level)
        test_cases = [
            # Canines
            ("fox", "male", 1),
            ("fox", "male", 2),
            ("fox", "male", 3),
            ("wolf", "male", 3),
            ("dog", "male", 2),
            ("coyote", "male", 1),
            ("fox", "female", 2),
            # Felines
            ("cat", "male", 2),
            ("tiger", "male", 3),
            ("cat", "female", 2),
            # Equines
            ("horse", "male", 2),
            ("pony", "male", 3),
            ("zebra", "female", 2),
            # Other mammals
            ("deer", "male", 2),
            ("raccoon", "male", 2),
            ("rabbit", "female", 2),
            # Reptiles, Avians, Fantasy
            ("dragon", "male", 2),
            ("lizard", "male", 3),
            ("bird", "female", 2),
            ("sergal", "male", 3),
            ("protogen", "male", 2),
            # Edge cases
            ("unknown_species", "male", 2),  # Should default to "accessory"
            ("hyena", "female", 3),  # Specific taxonomy
        ]

        # Check anatomical terms for each case
        for species, gender, explicit_level in test_cases:
            terms = get_anatomical_terms(species, gender, explicit_level)

            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species.lower(), "accessory")
            print(
                f"{species:<12} {gender:<6} (level {explicit_level}) → Taxonomy: {taxonomy_key:<15} Terms: {', '.join(terms)}"
            )

    def test_prompt_enhancement():
        """Test that prompts are correctly enhanced with species-specific tags."""
        print("\n=== Testing Prompt Enhancement ===\n")

        # Test cases: list of tuples (species, gender, intensity)
        test_cases = [
            # Canines
            ("fox", "male", "explicit"),
            ("wolf", "male", "hardcore"),
            ("dog", "male", "explicit"),
            ("fox", "female", "explicit"),
            # Felines
            ("cat", "male", "explicit"),
            ("tiger", "male", "hardcore"),
            # Equines
            ("horse", "male", "explicit"),
            ("pony", "female", "explicit"),
            # Other mammals
            ("deer", "male", "explicit"),
            ("rabbit", "female", "mild"),
            # Reptiles, Avians, Fantasy
            ("dragon", "male", "hardcore"),
            ("lizard", "male", "explicit"),
            ("bird", "female", "explicit"),
            ("sergal", "male", "hardcore"),
            ("protogen", "male", "explicit"),
            # Edge cases
            ("unknown_species", "male", "explicit"),  # Should default to "accessory"
        ]

        # Check enhanced prompts for each case
        for species, gender, intensity in test_cases:
            explicit_level = get_explicit_level(intensity)

            # Get enhanced prompt
            enhanced = enhance_prompt_with_species_tags(
                BASE_PROMPT, species, gender, explicit_level
            )

            # Calculate what tags were added
            tags_added = enhanced.replace(BASE_PROMPT + ", ", "")

            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species.lower(), "accessory")
            print(
                f"{species:<12} {gender:<6} {intensity:<9} → Taxonomy: {taxonomy_key:<15} Tags: {tags_added}"
            )

    def test_repeated_cases():
        """Test specific cases that were previously problematic."""
        print("\n=== Testing Previously Problematic Cases ===\n")

        # Case 1: Male fox should get canine-specific tags
        species, gender, level = "fox", "male", 2
        terms = get_anatomical_terms(species, gender, level)
        should_contain = ["animal penis", "canine penis"]
        contains_expected = all(term in terms for term in should_contain)
        print(f"Male fox tags: {', '.join(terms)}")
        print(f"Contains expected terms: {'Yes' if contains_expected else 'No'}")

        # Case 2: Male wolf should get canine-specific tags at level 3
        species, gender, level = "wolf", "male", 3
        terms = get_anatomical_terms(species, gender, level)
        should_contain = ["knot", "sheath"]
        contains_expected = all(term in terms for term in should_contain)
        print(f"Male wolf tags (level 3): {', '.join(terms)}")
        print(f"Contains expected terms: {'Yes' if contains_expected else 'No'}")

        # Case 3: Check that hemipenes for dragons is handled correctly
        enhanced = enhance_prompt_with_species_tags(
            "anthro, dragon, male, hemipenes, spread wings", "dragon", "male", 3
        )
        print(f"Dragon with hemipenes: {enhanced}")
        print(
            f"Didn't add redundant tags: {'Yes' if enhanced == 'anthro, dragon, male, hemipenes, spread wings' else 'No'}"
        )

    def test_random_species():
        """Test random species from the SPECIES_TAXONOMY dictionary."""
        print("\n=== Testing Random Species from Taxonomy ===\n")

        # Get 10 random species from the taxonomy dictionary
        all_species = list(taxonomy.SPECIES_TAXONOMY.keys())
        random_species = random.sample(all_species, min(10, len(all_species)))

        for species in random_species:
            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species.lower())

            # Try both genders
            for gender in ["male", "female"]:
                # Get terms for explicit level 2
                terms = get_anatomical_terms(species, gender, 2)

                # Check enhanced prompt
                enhanced = enhance_prompt_with_species_tags(
                    BASE_PROMPT, species, gender, 2
                )

                tags_added = enhanced.replace(BASE_PROMPT + ", ", "")

                print(
                    f"{species:<15} {gender:<6} → Taxonomy: {taxonomy_key:<15} Terms: {', '.join(terms)}"
                )
                print(f"  Enhanced with: {tags_added}")

    # Run all tests
    if __name__ == "__main__":
        print("COMPREHENSIVE TESTING OF SPECIES TAXONOMY AND ANATOMICAL TAGS")
        print("=" * 70)

        test_species_taxonomy_mapping()
        test_anatomical_terms()
        test_prompt_enhancement()
        test_repeated_cases()
        test_random_species()

        print("\nAll tests completed.")

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
