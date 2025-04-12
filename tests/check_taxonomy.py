#!/usr/bin/env python3
"""
Script to check if species_taxonomy_override is necessary or if SPECIES_TAXONOMY already has correct mappings.
"""

import sys
import os
from pathlib import Path

# Add the cringegen directory to the path
REPO_ROOT = Path("/home/kade/code/cringe.live")
CRINGEGEN_DIR = REPO_ROOT / "cringegen"
sys.path.append(str(CRINGEGEN_DIR))

try:
    from cringegen.data import taxonomy

    # Define the override mapping for comparison
    species_taxonomy_override = {
        "fox": "canine",
        "wolf": "canine",
        "dog": "canine",
        "coyote": "canine",
        "cat": "feline",
        "tiger": "feline",
        "lion": "feline",
        "leopard": "feline",
        "horse": "equine",
        "pony": "equine",
        "zebra": "equine",
    }

    # Check if SPECIES_TAXONOMY contains all the mappings in species_taxonomy_override
    print("=== Checking Taxonomy Mappings ===\n")

    # First, show what's in SPECIES_TAXONOMY for these species
    print("Mappings in SPECIES_TAXONOMY:")

    inconsistencies = []
    missing = []

    for species, expected_taxonomy in species_taxonomy_override.items():
        actual_taxonomy = taxonomy.SPECIES_TAXONOMY.get(species.lower())

        if actual_taxonomy is None:
            print(f"  - {species}: MISSING")
            missing.append(species)
        elif actual_taxonomy != expected_taxonomy:
            print(f"  - {species}: {actual_taxonomy} (expected: {expected_taxonomy})")
            inconsistencies.append(species)
        else:
            print(f"  - {species}: {actual_taxonomy} ✓")

    # Summary
    print("\nSummary:")
    if not inconsistencies and not missing:
        print(
            "✓ All mappings in species_taxonomy_override are already correctly defined in SPECIES_TAXONOMY."
        )
        print("Recommendation: The override is redundant and can be removed.")
    else:
        if inconsistencies:
            print(
                f"✗ Found {len(inconsistencies)} inconsistent mappings: {', '.join(inconsistencies)}"
            )
        if missing:
            print(f"✗ Found {len(missing)} missing mappings: {', '.join(missing)}")
        print("Recommendation: Keep the override to maintain correct mappings.")

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
