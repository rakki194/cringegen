#!/usr/bin/env python3
"""
Test script to verify that the newly added fictional species are correctly recognized.
"""

import sys
from pathlib import Path

# Add the cringegen directory to the path so we can import from it
REPO_ROOT = Path("/home/kade/code/cringe.live")
CRINGEGEN_DIR = REPO_ROOT / "cringegen"
sys.path.append(str(CRINGEGEN_DIR))

try:
    from cringegen.data import taxonomy
    from cringegen.prompt_generation.nlp.species_utils import get_anatomical_terms
    from cringegen.commands.llm_noobai_nsfw import enhance_prompt_with_species_tags
    from cringegen.data.taxonomy.descriptors import SPECIES_DESCRIPTORS
    from cringegen.data.colors import SPECIES_COLORS

    def test_fictional_species_taxonomy():
        """Test that all new fictional species are correctly mapped in the taxonomy."""
        print("\n=== Testing Fictional Species Taxonomy Mapping ===\n")

        # Define species to test by category
        test_species = {
            "Pokemon": [
                "lucario",
                "zoroark",
                "delphox",
                "charizard",
                "braixen",
                "incineroar",
                "zeraora",
                "lycanroc",
                "umbreon",
                "sylveon",
            ],
            "Digimon": [
                "renamon",
                "guilmon",
                "garurumon",
                "veemon",
                "agumon",
                "flamedramon",
                "weregarurumon",
            ],
            "Video Game Characters": [
                "blaidd",
                "maliketh",
                "sif",
                "krystal",
                "wolf_o'donnell",
                "judy_hopps",
                "nick_wilde",
                "legoshi",
                "sonic",
                "tails",
            ],
        }

        # Test each category
        for category, species_list in test_species.items():
            print(f"\n{category}:")
            for species in species_list:
                taxonomy_group = taxonomy.SPECIES_TAXONOMY.get(species)
                if taxonomy_group:
                    print(f"✓ {species:<20} → {taxonomy_group}")
                else:
                    print(f"✗ {species:<20} → MISSING")

    def test_fictional_anatomical_terms():
        """Test that anatomical terms are retrieved for fictional species."""
        print("\n=== Testing Anatomical Terms for Fictional Species ===\n")

        test_cases = [
            # Pokemon
            ("lucario", "male", 2),
            ("zoroark", "female", 2),
            # Digimon
            ("renamon", "male", 2),
            ("guilmon", "female", 2),
            # Video Game Characters
            ("blaidd", "male", 2),
            ("krystal", "female", 2),
        ]

        for species, gender, explicit_level in test_cases:
            terms = get_anatomical_terms(species, gender, explicit_level)
            taxonomy_key = taxonomy.SPECIES_TAXONOMY.get(species.lower(), "accessory")

            print(
                f"{species:<20} {gender:<6} (level {explicit_level}) → Taxonomy: {taxonomy_key:<20} Terms: {', '.join(terms)}"
            )

    def test_fictional_species_prompts():
        """Test prompt enhancement with fictional species."""
        print("\n=== Testing Prompt Enhancement for Fictional Species ===\n")

        base_prompt = "anthro, masterpiece, best quality, (standing:1.2), looking at viewer"

        test_cases = [
            # Pokemon
            ("lucario", "male", 2),
            ("zoroark", "female", 2),
            # Digimon
            ("renamon", "male", 2),
            ("guilmon", "female", 2),
            # Video Game Characters
            ("blaidd", "male", 2),
            ("krystal", "female", 2),
        ]

        for species, gender, explicit_level in test_cases:
            enhanced = enhance_prompt_with_species_tags(
                base_prompt, species, gender, explicit_level
            )

            print(f"\n{species} ({gender}):")
            print(f"Base prompt: {base_prompt}")
            print(f"Enhanced:    {enhanced}")

    def test_species_categorization():
        """Test proper categorization in species lists."""
        print("\n=== Testing Fictional Species Categorization ===\n")

        SPECIES_TAXONOMY = taxonomy.SPECIES_TAXONOMY
        categories = {
            "ANTHRO_SPECIES": taxonomy.ANTHRO_SPECIES,
            "POPULAR_ANTHRO_SPECIES": taxonomy.POPULAR_ANTHRO_SPECIES,
            "POKEMON_SPECIES": {k for k, v in SPECIES_TAXONOMY.items() if v == "pokemon"},
            "DIGIMON_SPECIES": {k for k, v in SPECIES_TAXONOMY.items() if v == "digimon"},
            "VIDEO_GAME_CHARACTERS": set(),  # Define as needed
            "FANTASY_SPECIES": taxonomy.FANTASY_SPECIES,
            "FANTASTICAL_SPECIES": set(),  # Define as needed
            "ALL_SPECIES": set(SPECIES_TAXONOMY.keys()),
        }

        test_species = [
            "lucario",  # Pokemon
            "renamon",  # Digimon
            "blaidd",  # Video Game
            "zoroark",  # Pokemon
            "guilmon",  # Digimon
            "krystal",  # Video Game
        ]

        for species in test_species:
            found_in = []
            for category_name, category_list in categories.items():
                if species in category_list:
                    found_in.append(category_name)

            print(f"{species:<20} → Found in: {', '.join(found_in) if found_in else 'NONE'}")

    def test_species_attributes():
        """Test species-specific attributes."""
        print("\n=== Testing Fictional Species Attributes ===\n")

        test_species = [
            "lucario",  # Pokemon
            "zoroark",  # Pokemon
            "renamon",  # Digimon
            "blaidd",  # Video Game
            "maliketh",  # Video Game
        ]

        print("Adjectives:")
        for species in test_species:
            adjectives = SPECIES_DESCRIPTORS.get(species, [])
            print(f"{species:<20} → {', '.join(adjectives) if adjectives else 'NONE'}")

        print("\nColors:")
        for species in test_species:
            colors = SPECIES_COLORS.get(species, [])
            print(f"{species:<20} → {', '.join(colors) if colors else 'NONE'}")

    def test_taxonomy_group_assignments():
        """Test that the taxonomy groups for fictional species are properly defined."""
        print("\n=== Testing Taxonomy Group Assignments ===\n")

        # Check if the fictional taxonomy groups exist
        taxonomy_group_categories = ["pokemon", "digimon", "video_game_character"]

        for group in taxonomy_group_categories:
            found_in_groups = False
            for category, members in taxonomy.TAXONOMY_GROUPS.items():
                if group in members:
                    found_in_groups = True
                    print(f"✓ {group:<20} → Found in category: {category}")
                    break

            if not found_in_groups:
                print(f"✗ {group:<20} → Not found in any taxonomy group category")

    # Run all tests
    if __name__ == "__main__":
        print("TESTING FICTIONAL SPECIES IN TAXONOMY")
        print("=" * 70)

        test_fictional_species_taxonomy()
        test_fictional_anatomical_terms()
        test_fictional_species_prompts()
        test_species_categorization()
        test_species_attributes()
        test_taxonomy_group_assignments()

        print("\nAll tests completed.")

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
