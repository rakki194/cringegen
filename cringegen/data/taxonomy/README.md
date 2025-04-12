# Taxonomy Package

This package organizes the taxonomic data used by cringegen for prompt generation. It was refactored from a monolithic `taxonomy.py` file to a modular structure for better maintainability.

## Package Structure

- `__init__.py` - Re-exports all components for backward compatibility
- `species.py` - Contains core species taxonomy mappings and species lists
- `body_coverings.py` - Defines body covering types and patterns by taxonomy
- `groups.py` - Organizes taxonomy into broader groups and hierarchies
- `descriptors.py` - Provides descriptive terms for species and taxonomic groups
- `hybrids.py` - Defines hybrid species combinations and trait inheritance

## Usage

The package maintains the same interface as the original taxonomy.py file. All data structures are imported and re-exported through the __init__.py file, so existing code should continue to work without modifications:

```python
# Import at package level
from cringegen.data.taxonomy import SPECIES_TAXONOMY, BODY_COVERING_BY_TAXONOMY

# Or through the parent package (backward compatible)
from cringegen.data import SPECIES_TAXONOMY, BODY_COVERING_BY_TAXONOMY
```

## Data Organization

- __Species Data__: Species are mapped to their taxonomic groups and organized into sets like `ANTHRO_SPECIES` and `FANTASY_SPECIES`.
- __Body Coverings__: Defines appropriate body coverings (fur, scales, etc.) for each taxonomic group.
- __Taxonomy Groups__: Organizes species into higher-level taxonomic categories.
- __Descriptors__: Provides descriptive terms and cultural traits for species.
- __Hybrids__: Defines valid hybrid combinations and trait inheritance patterns.

## Extending the Taxonomy

To add new species or taxonomic data:

1. Identify the appropriate module(s) for your additions
2. Add the new data to the relevant dictionaries/sets
3. If creating a new taxonomy category, update multiple files as needed:
   - Add to `SPECIES_TAXONOMY` in `species.py`
   - Add appropriate body coverings in `body_coverings.py`
   - Add to taxonomy groups in `groups.py`
   - Add descriptors in `descriptors.py`
   - Consider hybrid compatibility in `hybrids.py`
