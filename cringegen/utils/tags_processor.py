"""
Tags processor for loading and using the e621 tags data with NLTK for prompt generation.
"""

import json
import os
from typing import Dict, List, Optional
import nltk
from nltk.corpus import wordnet
from pathlib import Path

from ..data import anatomy, taxonomy
from ..prompt_generation.nlp.species_utils import (
    get_anatomical_terms,
    get_species_accessories,
    get_species_colors,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TagsProcessor:
    """Process and utilize e621 tags for prompt generation"""

    def __init__(self, tags_file: Optional[str] = None):
        """Initialize the tags processor

        Args:
            tags_file: Path to tags.json file (default: None, will use package default)
        """
        # Try to find tags file if not specified
        if not tags_file:
            # Look in standard locations
            possible_paths = [
                Path(__file__).parent.parent.parent / "tags.json",  # /cringegen/tags.json
                Path(__file__).parent.parent
                / "data"
                / "tags.json",  # /cringegen/cringegen/data/tags.json
            ]

            for path in possible_paths:
                if path.exists():
                    tags_file = str(path)
                    break

        if not tags_file or not os.path.exists(tags_file):
            logger.warning("Tags file not found. Limited functionality will be available.")
            self.tags_dict = {}
            self.id_to_tag = {}
            self.loaded = False
        else:
            # Load the tags file
            self.load_tags(tags_file)
            self.loaded = True

        # Download WordNet data if not already downloaded
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

    def load_tags(self, tags_file: str) -> None:
        """Load tags from the file

        Args:
            tags_file: Path to tags.json file
        """
        try:
            with open(tags_file, "r") as f:
                self.tags_dict = json.load(f)

            # Create reverse mapping (id to tag)
            self.id_to_tag = {v: k for k, v in self.tags_dict.items()}
            logger.info(f"Loaded {len(self.tags_dict)} tags from {tags_file}")
        except Exception as e:
            logger.error(f"Error loading tags file: {e}")
            self.tags_dict = {}
            self.id_to_tag = {}

    def format_tag(self, tag: str) -> str:
        """Format tag for use in prompts

        - Replaces underscores with spaces
        - Escapes parentheses with backslashes

        Args:
            tag: Original tag

        Returns:
            Formatted tag
        """
        # Replace underscores with spaces
        formatted = tag.replace("_", " ")

        # Escape parentheses with backslash
        formatted = formatted.replace("(", "\\(").replace(")", "\\)")

        return formatted

    def format_tags(self, tags: List[str]) -> List[str]:
        """Format a list of tags

        Args:
            tags: List of original tags

        Returns:
            List of formatted tags
        """
        return [self.format_tag(tag) for tag in tags]

    def get_related_tags(
        self, species: str, gender: str, nsfw: bool = False, explicit_level: int = 1
    ) -> List[str]:
        """Get related tags for a species and gender

        Args:
            species: The species name (e.g., "fox", "wolf")
            gender: The gender ("male", "female")
            nsfw: Whether to include NSFW tags
            explicit_level: Level of explicitness (1-3)

        Returns:
            List of related tags
        """
        if not self.loaded:
            logger.warning("Tags file not loaded. Cannot get related tags.")
            return []

        related_tags = []

        # Add species tag if it exists
        if species.lower() in self.tags_dict:
            related_tags.append(species.lower())

        # Add gender tag
        if gender.lower() in self.tags_dict:
            related_tags.append(gender.lower())

        # Add anthro tag
        if "anthro" in self.tags_dict:
            related_tags.append("anthro")

        # Add anatomical tags if NSFW
        if nsfw:
            anatomical_terms = get_anatomical_terms(species, gender, explicit_level)

            # Map these to e621 tags where possible
            for term in anatomical_terms:
                term = term.lower()
                # Check if the term or its components are in the tags dictionary
                if term in self.tags_dict:
                    related_tags.append(term)
                else:
                    # Try to find components (e.g., "canine penis" -> "canine" + "penis")
                    words = term.split()
                    for word in words:
                        if word in self.tags_dict and word not in related_tags:
                            related_tags.append(word)

        # Add species-specific tags
        # Add colors
        colors = get_species_colors(species, 1)
        for color in colors:
            if color in self.tags_dict and color not in related_tags:
                related_tags.append(color)

        # Add accessories
        accessories = get_species_accessories(species, gender, 1)
        for accessory in accessories:
            if accessory in self.tags_dict and accessory not in related_tags:
                related_tags.append(accessory)

        # Format all tags before returning
        return self.format_tags(related_tags)

    def get_species_taxonomy_tags(self, species: str) -> List[str]:
        """Get taxonomy-related tags for a species

        Args:
            species: The species name

        Returns:
            List of taxonomy tags (e.g. "canine", "feline")
        """
        if not self.loaded:
            return []

        # Get the taxonomy group for this species
        taxonomy_group = taxonomy.SPECIES_TAXONOMY.get(species.lower(), "default")

        # If taxonomy_group is in tags, add it
        if taxonomy_group in self.tags_dict:
            return [self.format_tag(taxonomy_group)]

        # Get alternative synonyms using WordNet
        try:
            syns = wordnet.synsets(taxonomy_group)
            alt_terms = []
            for syn in syns:
                alt_terms.extend(syn.lemma_names())

            # Check if any of these are in the tags dictionary
            for term in alt_terms:
                if term.lower() in self.tags_dict:
                    return [self.format_tag(term.lower())]
        except Exception:
            pass

        return []

    def expand_tag_with_synonyms(self, tag: str) -> List[str]:
        """Use WordNet to find synonyms for a tag

        Args:
            tag: The tag to expand

        Returns:
            List of synonyms that exist in the tags dictionary
        """
        if not self.loaded:
            return []

        try:
            syns = wordnet.synsets(tag)
            synonyms = []
            for syn in syns:
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name().lower())

            # Filter to only ones in our tags dictionary
            valid_tags = [syn for syn in synonyms if syn in self.tags_dict]

            # Format tags before returning
            return self.format_tags(valid_tags)
        except Exception:
            return []

    def generate_species_specific_tags(
        self, species: str, gender: str, nsfw: bool = False, explicit_level: int = 1
    ) -> Dict[str, List[str]]:
        """Generate comprehensive species-specific tags by category

        Args:
            species: The species name
            gender: The gender
            nsfw: Whether to include NSFW tags
            explicit_level: Level of explicitness (1-3)

        Returns:
            Dictionary of categorized tags
        """
        result = {
            "identity": [],
            "physical": [],
            "accessories": [],
            "nsfw": [],
        }

        # Basic identity tags
        result["identity"] = ["anthro", species.lower(), gender.lower()]

        # Add taxonomy group
        taxonomy_tags = self.get_species_taxonomy_tags(species)
        if taxonomy_tags:
            result["identity"].extend(taxonomy_tags)

        # Physical characteristics
        colors = get_species_colors(species, 2)
        for color in colors:
            if color in self.tags_dict:
                result["physical"].append(f"{color} fur")

        # Species-specific physical traits
        taxonomy_group = taxonomy.SPECIES_TAXONOMY.get(species.lower(), "default")
        if taxonomy_group in anatomy.ANTHRO_FEATURES:
            for feature in anatomy.ANTHRO_FEATURES[taxonomy_group]:
                if feature in self.tags_dict:
                    result["physical"].append(feature)

        # Accessories
        accessories = get_species_accessories(species, gender, 2)
        result["accessories"] = [acc for acc in accessories if acc in self.tags_dict]

        # NSFW tags
        if nsfw:
            anatomical_terms = get_anatomical_terms(species, gender, explicit_level)
            result["nsfw"] = [
                term
                for term in anatomical_terms
                if term in self.tags_dict or any(word in self.tags_dict for word in term.split())
            ]

        # Format all tags in the result
        for category in result:
            result[category] = self.format_tags(result[category])

        return result


# Create a default instance for ease of use
default_processor = TagsProcessor()
