"""
Taxonomy package for cringegen

This package contains structured taxonomic data used for generating image prompts.
It organizes species data, body types, anatomical references, and classification hierarchies.
"""

# Import from submodules to provide a unified interface
from .species import (
    SPECIES_TAXONOMY,
    ANTHRO_SPECIES,
    FANTASY_SPECIES,
    POPULAR_ANTHRO_SPECIES,
)

from .body_coverings import (
    BODY_COVERING_BY_TAXONOMY,
)

from .groups import (
    TAXONOMY_GROUPS,
    TAXONOMY_HIERARCHY,
)

from .descriptors import (
    ANTHRO_DESCRIPTOR_SET,
    SPECIES_DESCRIPTORS,
)

from .hybrids import (
    HYBRID_COMBINATIONS,
    COMPATIBLE_HYBRIDS,
)

# Re-export everything for backward compatibility
__all__ = [
    # Species taxonomy data
    "SPECIES_TAXONOMY",
    "ANTHRO_SPECIES",
    "FANTASY_SPECIES",
    "POPULAR_ANTHRO_SPECIES",
    # Body covering data
    "BODY_COVERING_BY_TAXONOMY",
    # Taxonomy group data
    "TAXONOMY_GROUPS",
    "TAXONOMY_HIERARCHY",
    # Descriptor data
    "ANTHRO_DESCRIPTOR_SET",
    "SPECIES_DESCRIPTORS",
    # Hybrid data
    "HYBRID_COMBINATIONS",
    "COMPATIBLE_HYBRIDS",
]
