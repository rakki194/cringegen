#!/usr/bin/env python3
"""
Simple test script to manually test a new species with the enhanced prompt system.
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
    from cringegen.prompt_generation.nlp.species_utils import get_anatomical_terms
    from cringegen.commands.llm_noobai_nsfw import enhance_prompt_with_species_tags

    print("=== Testing New Species Integration ===\n")

    # Select a few test species
    test_species = ["kitsune", "eastern_dragon", "axolotl", "dutch_angel_dragon", "dingo"]

    # Loop through each species
    for species in test_species:
        print(f"\nTesting species: {species}")

        # Get the taxonomy group
        taxonomy_group = taxonomy.SPECIES_TAXONOMY.get(species, "Not found")
        print(f"  Taxonomy group: {taxonomy_group}")

        # Test with different genders
        for gender in ["male", "female"]:
            # Get anatomical terms (explicit level 2)
            terms = get_anatomical_terms(species, gender, 2)
            print(
                f"  {gender.capitalize()} anatomical terms: {', '.join(terms) if terms else 'None'}"
            )

            # Test enhancing a prompt
            base_prompt = "anthro, masterpiece, best quality, (standing:1.2), looking at viewer"
            enhanced = enhance_prompt_with_species_tags(base_prompt, species, gender, 2)
            print(f"  Enhanced prompt ({gender}): {enhanced}")

        print("-" * 50)

    print("\nTest completed.")

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
