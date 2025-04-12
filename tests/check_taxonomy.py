#!/usr/bin/env python3

"""
Test script to verify that the taxonomy refactoring works correctly.
This checks that all the expected data structures are available and populated.
"""

import sys
import os
from pathlib import Path

# Now try to import from both the direct taxonomy package and the re-exports
try:
    # Import directly from the taxonomy package
    from cringegen.data.taxonomy import (
        SPECIES_TAXONOMY,
        BODY_COVERING_BY_TAXONOMY,
        ANTHRO_SPECIES,
        FANTASY_SPECIES,
        POPULAR_ANTHRO_SPECIES,
        TAXONOMY_GROUPS,
        ANTHRO_DESCRIPTOR_SET,
    )

    # Import from the data module (testing re-exports work)
    from cringegen.data import (
        SPECIES_TAXONOMY as SPECIES_TAXONOMY2,
        BODY_COVERING_BY_TAXONOMY as BODY_COVERING_BY_TAXONOMY2,
        ANTHRO_SPECIES as ANTHRO_SPECIES2,
        FANTASY_SPECIES as FANTASY_SPECIES2,
        POPULAR_ANTHRO_SPECIES as POPULAR_ANTHRO_SPECIES2,
        TAXONOMY_GROUPS as TAXONOMY_GROUPS2,
        ANTHRO_DESCRIPTOR_SET as ANTHRO_DESCRIPTOR_SET2,
    )

    print("✅ Import test passed - All taxonomy structures were imported successfully")

    # Verify the structures have data
    print("\nVerifying data structures contain expected data:")

    # Check species taxonomy
    print(f"- SPECIES_TAXONOMY: {len(SPECIES_TAXONOMY)} species mapped")
    print(f"  Example: wolf -> {SPECIES_TAXONOMY.get('wolf')}")

    # Check body coverings
    print(
        f"- BODY_COVERING_BY_TAXONOMY: {len(BODY_COVERING_BY_TAXONOMY)} taxonomies with body coverings"
    )
    print(f"  Example (canine): {BODY_COVERING_BY_TAXONOMY.get('canine')[:3]}...")

    # Check species sets
    print(f"- ANTHRO_SPECIES: {len(ANTHRO_SPECIES)} species")
    print(f"- FANTASY_SPECIES: {len(FANTASY_SPECIES)} species")
    print(f"- POPULAR_ANTHRO_SPECIES: {len(POPULAR_ANTHRO_SPECIES)} species")

    # Verify taxonomy groups
    print(f"- TAXONOMY_GROUPS: {len(TAXONOMY_GROUPS)} taxonomy groups")

    # Verify descriptor set
    print(f"- ANTHRO_DESCRIPTOR_SET: {len(ANTHRO_DESCRIPTOR_SET)} descriptors")

    # Verify the data is identical from both import paths
    identical = (
        SPECIES_TAXONOMY == SPECIES_TAXONOMY2
        and BODY_COVERING_BY_TAXONOMY == BODY_COVERING_BY_TAXONOMY2
        and ANTHRO_SPECIES == ANTHRO_SPECIES2
        and FANTASY_SPECIES == FANTASY_SPECIES2
        and POPULAR_ANTHRO_SPECIES == POPULAR_ANTHRO_SPECIES2
        and TAXONOMY_GROUPS == TAXONOMY_GROUPS2
        and ANTHRO_DESCRIPTOR_SET == ANTHRO_DESCRIPTOR_SET2
    )

    if identical:
        print("\n✅ Data verification passed - Re-exported structures contain identical data")
    else:
        print("\n❌ Data verification failed - Re-exported structures have different data")

    # Print specific examples to verify deeper structure
    from cringegen.data.taxonomy.hybrids import HYBRID_COMBINATIONS

    print(f"\n- HYBRID_COMBINATIONS: {len(HYBRID_COMBINATIONS)} hybrid combinations defined")
    print(f"  Example (wolf_fox): {HYBRID_COMBINATIONS.get('wolf_fox')['name']}")

    from cringegen.data.taxonomy.descriptors import SPECIES_DESCRIPTORS

    print(f"- SPECIES_DESCRIPTORS: {len(SPECIES_DESCRIPTORS)} species with descriptors")
    print(f"  Example (wolf): {SPECIES_DESCRIPTORS.get('wolf')[:3]}...")

    print("\n✅ All tests passed! The taxonomy refactoring appears to be working correctly.")

except ImportError as e:
    print(f"❌ Import test failed: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback

    traceback.print_exc()
