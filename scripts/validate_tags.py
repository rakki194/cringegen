#!/usr/bin/env python
"""
Script to validate tags used in data files against tags.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add the parent directory to the path so we can import cringegen
sys.path.append(str(Path(__file__).parent.parent))

# Import data modules
from cringegen.data import anatomy, accessories, colors, taxonomy


def load_tags_json(tags_file: str) -> Dict[str, int]:
    """Load tags from tags.json file

    Args:
        tags_file: Path to tags.json file

    Returns:
        Dictionary of tags
    """
    try:
        with open(tags_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading tags file: {e}")
        return {}


def extract_tags_from_data() -> Dict[str, Set[str]]:
    """Extract tags from data modules

    Returns:
        Dictionary of tags grouped by data module
    """
    data_tags = {
        "anatomy": set(),
        "accessories": set(),
        "colors": set(),
        "taxonomy": set(),
    }

    # Extract tags from MALE_ANATOMY
    for taxonomy_group, terms in anatomy.MALE_ANATOMY.items():
        data_tags["anatomy"].add(taxonomy_group)
        for term in terms:
            data_tags["anatomy"].add(term)

    # Extract tags from FEMALE_ANATOMY
    for taxonomy_group, terms in anatomy.FEMALE_ANATOMY.items():
        data_tags["anatomy"].add(taxonomy_group)
        for term in terms:
            data_tags["anatomy"].add(term)

    # Extract tags from ANTHRO_FEATURES
    for taxonomy_group, features in anatomy.ANTHRO_FEATURES.items():
        data_tags["anatomy"].add(taxonomy_group)
        for feature in features:
            data_tags["anatomy"].add(feature)

    # Extract tags from accessories
    for taxonomy_group, accessory_groups in accessories.SPECIES_ACCESSORIES.items():
        data_tags["accessories"].add(taxonomy_group)
        for form_type, gender_accessories in accessory_groups.items():
            for gender, items in gender_accessories.items():
                for item in items:
                    data_tags["accessories"].add(item)

    # Extract tags from colors
    for species, color_list in colors.SPECIES_COLORS.items():
        data_tags["colors"].add(species)
        for color in color_list:
            data_tags["colors"].add(color)

    # Extract basic colors
    for color in colors.BASIC_COLORS:
        data_tags["colors"].add(color)

    # Extract tags from taxonomy
    for species, taxonomy_group in taxonomy.SPECIES_TAXONOMY.items():
        data_tags["taxonomy"].add(species)
        data_tags["taxonomy"].add(taxonomy_group)

    return data_tags


def validate_tags(
    tags_json: Dict[str, int], data_tags: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    """Validate tags from data modules against tags.json

    Args:
        tags_json: Dictionary of tags from tags.json
        data_tags: Dictionary of tags from data modules

    Returns:
        Dictionary of missing tags grouped by data module
    """
    missing_tags = {
        "anatomy": [],
        "accessories": [],
        "colors": [],
        "taxonomy": [],
    }

    # Check each data module for missing tags
    for module, tag_set in data_tags.items():
        for tag in tag_set:
            # Skip tags that are too short (likely not real tags)
            if len(tag) < 3:
                continue

            # Skip anatomical terms that are clearly composite phrases
            if (
                " " in tag
                and module == "anatomy"
                and ("penis" in tag or "sheath" in tag or "pussy" in tag or "genitalia" in tag)
            ):
                # These are composite terms, not individual tags
                continue

            if tag not in tags_json:
                # Check for tag with underscores
                underscored_tag = tag.replace(" ", "_")
                if underscored_tag not in tags_json:
                    missing_tags[module].append(tag)

    return missing_tags


def format_tag_suggestions(missing_tags: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """Format tag suggestions with potential replacement

    Args:
        missing_tags: Dictionary of missing tags

    Returns:
        Dictionary of tag suggestions with potential replacements
    """
    suggestions = {
        "anatomy": [],
        "accessories": [],
        "colors": [],
        "taxonomy": [],
    }

    # Load tags from tags.json to find similar tags
    tags_json = load_tags_json(str(Path(__file__).parent.parent / "tags.json"))

    for module, tags in missing_tags.items():
        for tag in tags:
            similar_tags = []

            # Find similar tags
            for existing_tag in tags_json.keys():
                # Convert underscores to spaces for comparison
                existing_tag_spaces = existing_tag.replace("_", " ")

                # Check for substring matches
                if tag in existing_tag_spaces or existing_tag_spaces in tag:
                    similar_tags.append(existing_tag)

                # Check for word overlap (at least half the words match)
                tag_words = set(tag.lower().split())
                existing_words = set(existing_tag_spaces.lower().split())
                if len(tag_words.intersection(existing_words)) >= len(tag_words) / 2:
                    if existing_tag not in similar_tags:
                        similar_tags.append(existing_tag)

            # Find the best replacement tag
            if similar_tags:
                suggestions[module].append((tag, ", ".join(similar_tags[:3])))
            else:
                suggestions[module].append((tag, "No similar tags found"))

    return suggestions


def main():
    """Main function"""
    # Load tags from tags.json
    tags_file = str(Path(__file__).parent.parent / "tags.json")

    if not os.path.exists(tags_file):
        print(f"Error: tags.json not found at {tags_file}")
        sys.exit(1)

    print(f"Loading tags from {tags_file}...")
    tags_json = load_tags_json(tags_file)
    print(f"Loaded {len(tags_json)} tags from tags.json\n")

    # Extract tags from data modules
    print("Extracting tags from data modules...")
    data_tags = extract_tags_from_data()

    # Print count of tags from each module
    for module, tag_set in data_tags.items():
        print(f"Found {len(tag_set)} tags in {module} module")
    print()

    # Validate tags
    print("Validating tags...")
    missing_tags = validate_tags(tags_json, data_tags)

    # Print missing tags
    print("\nMissing Tags Report:")
    print("=" * 80)

    for module, tags in missing_tags.items():
        if tags:
            print(f"\n{module.capitalize()} Module Missing Tags ({len(tags)}):")
            print("-" * 80)
            for tag in sorted(tags):
                print(f"- {tag}")

    # Generate suggestions
    suggestions = format_tag_suggestions(missing_tags)

    # Print suggestions
    print("\nTag Replacement Suggestions:")
    print("=" * 80)

    for module, tag_suggestions in suggestions.items():
        if tag_suggestions:
            print(f"\n{module.capitalize()} Module Suggestions:")
            print("-" * 80)
            for original, replacement in tag_suggestions:
                print(f"- {original} → {replacement}")


if __name__ == "__main__":
    main()
