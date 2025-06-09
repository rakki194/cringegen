#!/usr/bin/env python3
"""
Test script to verify that the newly added species are correctly recognized.
"""

import sys
import os
from pathlib import Path

# Add the cringegen directory to the path so we can import from it
REPO_ROOT = Path("/home/kade/code/cringe.live")
CRINGEGEN_DIR = REPO_ROOT / "cringegen"
sys.path.append(str(CRINGEGEN_DIR))

try:
    from cringegen.data import taxonomy
    from cringegen.data.anatomy import MALE_ANATOMY, FEMALE_ANATOMY
    from cringegen.prompt_generation.nlp.species_utils import get_anatomical_terms

    def test_new_species_taxonomy():
        """Test that all new species are correctly mapped in the taxonomy."""
        print("\n=== Testing New Species Taxonomy Mapping ===\n")

        new_species = [
            # Additional canines
            "dingo",
            "african_wild_dog",
            "maned_wolf",
            # Additional felines
            "caracal",
            "serval",
            "jaguar",
            "panther",
            "scottish_fold",
            "maine_coon",
            # Fantasy/original species
            "dutch_angel_dragon",
            "eastern_dragon",
            "primagen",
            "synx",
            "manticore",
            "kitsune",
            "drekkubus",
            "sparkledog",
            "latex_creature",
            "goo_creature",
            # Special hybrids
            "fox_wolf_hybrid",
            "cat_dog_hybrid",
            "liger",
            "tigon",
            "dragon_wolf",
            # Miscellaneous species
            "aardwolf",
            "mongoose",
            "possum",
            "opossum",
            "axolotl",
            # Additional avians
            "hawk",
            "falcon",
            "parrot",
            "macaw",
            "corvid",
            "duck",
        ]

        for species in new_species:
            taxonomy_group = taxonomy.SPECIES_TAXONOMY.get(species)
            if taxonomy_group:
                print(f"✓ {species:<20} → {taxonomy_group}")
            else:
                print(f"✗ {species:<20} → MISSING")

    def test_anatomical_terms_for_new_species():
        """Test that anatomical terms are retrieved for new species."""
        print("\n=== Testing Anatomical Terms for New Species ===\n")

        test_cases = [
            # Canines
            ("dingo", "male", 2),
            ("african_wild_dog", "female", 2),
            # Felines
            ("caracal", "male", 2),
            ("serval", "female", 2),
            # Fantasy species
            ("kitsune", "male", 2),
            ("eastern_dragon", "female", 2),
            # Hybrids
            ("fox_wolf_hybrid", "male", 2),
            ("dragon_wolf", "female", 2),
            # Other
            ("axolotl", "male", 2),
            ("corvid", "female", 2),
        ]

        for species, gender, explicit_level in test_cases:
            terms = get_anatomical_terms(species, gender, explicit_level)
            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species.lower(), "accessory")

            print(
                f"{species:<20} {gender:<6} (level {explicit_level}) → Taxonomy: {taxonomy_key:<15} Terms: {', '.join(terms)}"
            )

    def test_species_categorizations():
        """Test that new species are in the appropriate lists."""
        print("\n=== Testing Species Categorization Lists ===\n")

        SPECIES_TAXONOMY = taxonomy.SPECIES_TAXONOMY
        species_categories = {
            "ANTHRO_SPECIES": taxonomy.ANTHRO_SPECIES,
            "COMMON_SPECIES": set(),  # Define as needed
            "RARE_SPECIES": set(),    # Define as needed
            "FANTASY_SPECIES": taxonomy.FANTASY_SPECIES,
            "FANTASTICAL_SPECIES": set(),  # Define as needed
        }

        test_species = {
            "Popular fantasy": ["kitsune", "eastern_dragon"],
            "Common animals": ["husky", "caracal", "serval"],
            "Fantasy": ["dutch_angel_dragon", "synx", "primagen"],
            "Hybrids": ["fox_wolf_hybrid", "dragon_wolf"],
            "Rare animals": ["axolotl", "mongoose", "corvid"],
        }

        for category_name, species_list in test_species.items():
            print(f"\n{category_name}:")
            for species in species_list:
                found_in = []
                for list_name, category_list in species_categories.items():
                    if species in category_list:
                        found_in.append(list_name)

                if found_in:
                    print(f"✓ {species:<20} → Found in: {', '.join(found_in)}")
                else:
                    print(f"✗ {species:<20} → Not found in any category list")

    def test_taxonomy_groups():
        """Test that new taxonomy groups are correctly defined."""
        print("\n=== Testing Taxonomy Groups ===\n")

        # New taxonomy groups we expect to see
        new_groups = [
            "eastern_dragon",
            "synx",
            "manticore",
            "kitsune",
            "drekkubus",
            "sparkledog",
            "latex_creature",
            "goo_creature",
            "hybrid",
            "marsupial",
            "amphibian",
        ]

        # Check which taxonomy groups are defined in MALE_ANATOMY and FEMALE_ANATOMY
        for group in new_groups:
            male_anatomy = group in MALE_ANATOMY
            female_anatomy = group in FEMALE_ANATOMY
            taxonomy_groups = False

            # Check if the group is in any taxonomy group
            for group_category, group_members in taxonomy.TAXONOMY_GROUPS.items():
                if group in group_members:
                    taxonomy_groups = True
                    break

            status = []
            if male_anatomy:
                status.append("MALE_ANATOMY")
            if female_anatomy:
                status.append("FEMALE_ANATOMY")
            if taxonomy_groups:
                status.append("TAXONOMY_GROUPS")

            if status:
                print(f"✓ {group:<20} → Defined in: {', '.join(status)}")
            else:
                print(f"✗ {group:<20} → Not defined in anatomy or taxonomy groups")

    # Run all tests
    if __name__ == "__main__":
        print("TESTING NEW SPECIES IN TAXONOMY")
        print("=" * 70)

        test_new_species_taxonomy()
        test_anatomical_terms_for_new_species()
        test_species_categorizations()
        test_taxonomy_groups()

        print("\nAll tests completed.")

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
