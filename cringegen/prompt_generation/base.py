"""
Base classes for prompt generation
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

from .nlp.background_utils import (
    generate_background_description,
    generate_scene_description,
    get_complementary_locations,
)
from .nlp.color_utils import (
    generate_color_description,
    get_body_covering_type,
    get_complementary_colors,
    get_random_colors,
)
from .nlp.species_utils import (
    generate_species_description,
    get_species_accessories,
    get_species_colors,
)


class PromptComponent(ABC):
    """Abstract base class for all prompt components"""

    @abstractmethod
    def get_tags(self) -> List[str]:
        """Get the tags for this component

        Returns:
            A list of tags
        """
        pass


class QualityComponent(PromptComponent):
    """Component for quality-related tags"""

    def __init__(self, quality_level: int = 5):
        """Initialize a quality component

        Args:
            quality_level: Quality level from 1 to 5
        """
        self.quality_level = max(1, min(5, quality_level))

    def get_tags(self) -> List[str]:
        """Get quality-related tags

        Returns:
            A list of quality tags
        """
        # Basic quality tags used in most models
        tags = ["masterpiece", "best quality"]

        # Additional tags based on quality level
        if self.quality_level >= 2:
            tags.append("high quality")
        if self.quality_level >= 3:
            tags.extend(["detailed", "sharp focus"])
        if self.quality_level >= 4:
            tags.extend(["high detail", "intricate details"])
        if self.quality_level >= 5:
            tags.extend(["ultra high res", "perfect lighting"])

        return tags


class CharacterComponent(PromptComponent):
    """Component for character-related tags"""

    def __init__(
        self,
        species: str = "human",
        gender: str = None,
        body_type: str = None,
        features: List[str] = None,
        colors: List[str] = None,
        color_pattern: str = None,
        use_nlp_description: bool = False,
        is_anthro: bool = True,
    ):
        """Initialize a character component

        Args:
            species: The species of the character
            gender: The gender of the character
            body_type: The body type of the character
            features: Additional features or descriptions
            colors: List of colors for the character
            color_pattern: Pattern for the colors (e.g., "spotted", "striped")
            use_nlp_description: Whether to use NLP for enhanced descriptions
            is_anthro: Whether the character is anthro (True) or feral (False)
        """
        self.species = species
        self.gender = gender
        self.body_type = body_type
        self.features = features or []
        self.colors = colors or []
        self.color_pattern = color_pattern
        self.use_nlp_description = use_nlp_description
        self.is_anthro = is_anthro

        # Generate species-specific accessories if none provided in features
        self._add_species_specific_features()

    def _add_species_specific_features(self) -> None:
        """Add species-specific features and accessories using NLP"""
        if not self.use_nlp_description or not self.species or not self.gender:
            return

        # Don't add accessories if features already has some
        has_accessory = any(
            word in " ".join(self.features).lower()
            for word in ["collar", "necklace", "hat", "glasses", "wearing"]
        )

        if not has_accessory:
            # Get 1-2 species-specific accessories
            count = random.randint(1, 2)
            accessories = get_species_accessories(self.species, self.gender, count, self.is_anthro)
            self.features.extend(accessories)

        # Add species-specific colors if no colors specified
        if not self.colors:
            colors = get_species_colors(self.species, 1)
            if colors:
                self.colors = colors

    def get_tags(self) -> List[str]:
        """Get character-related tags

        Returns:
            A list of character tags
        """
        tags = []

        # If using NLP description and we have species and gender
        if self.use_nlp_description and self.species and self.gender:
            # Generate a rich species description
            species_desc = generate_species_description(self.species, self.gender)
            tags.append(species_desc)

            # Generate color description if colors are specified
            if self.colors:
                color_desc = generate_color_description(
                    self.species, colors=self.colors, pattern=self.color_pattern
                )
                tags.append(color_desc)

            # Add any features not covered in the description
            for feature in self.features:
                # Only add if not likely to be in the description
                if feature.lower() not in species_desc.lower():
                    tags.append(feature)

            # Add gender and body type if specified
            if self.gender and self.gender.lower() not in species_desc.lower():
                tags.append(self.gender)

            if self.body_type:
                tags.append(self.body_type)

            return tags

        # Traditional tag generation if not using NLP
        # Add species
        if self.species and self.species.lower() != "human":
            # If anthro/furry character
            if self.species.lower() in ["anthro", "furry"]:
                tags.append("anthro")
            else:
                tags.append(self.species)
                # Check if it's an anthro animal
                if "human" not in self.features and "anthro" not in self.features:
                    tags.append("anthro")

        # Add gender
        if self.gender:
            tags.append(self.gender)

        # Add body type
        if self.body_type:
            tags.append(self.body_type)

        # Add color description if colors specified
        if self.colors and self.species.lower() != "human":
            covering_type = get_body_covering_type(self.species)
            if len(self.colors) == 1:
                tags.append(f"{self.colors[0]} {covering_type}")
            elif len(self.colors) > 1:
                color_str = " and ".join(self.colors)
                tags.append(f"{color_str} {covering_type}")

        # Add features
        tags.extend(self.features)

        return tags


class MultiSubjectComponent(PromptComponent):
    """Component for multiple subjects/characters"""

    def __init__(
        self,
        characters: List[CharacterComponent],
        arrangement: str = None,
        interaction: str = None,
    ):
        """Initialize a multi-subject component

        Args:
            characters: List of character components
            arrangement: How the characters are arranged (e.g., "side by side")
            interaction: How the characters are interacting (e.g., "talking")
        """
        self.characters = characters
        self.arrangement = arrangement
        self.interaction = interaction

    def get_tags(self) -> List[str]:
        """Get tags for multiple subjects

        Returns:
            A list of tags for multiple subjects
        """
        if not self.characters:
            return []

        tags = []

        # First, collect all character tags
        character_tag_groups = [char.get_tags() for char in self.characters]

        # Add arrangement if specified
        if self.arrangement:
            tags.append(self.arrangement)

        # Add interaction if specified
        if self.interaction:
            tags.append(self.interaction)

        # Add a group descriptor based on number of characters
        if len(self.characters) == 2:
            tags.append("duo")
            tags.append("two characters")
        elif len(self.characters) > 2:
            tags.append("group")
            tags.append(f"{len(self.characters)} characters")

        # Combine all character tags
        for char_tags in character_tag_groups:
            tags.extend(char_tags)

        return tags


class PoseComponent(PromptComponent):
    """Component for pose-related tags"""

    def __init__(
        self,
        pose: str = None,
        action: str = None,
        expression: str = None,
        additional_poses: List[str] = None,
    ):
        """Initialize a pose component

        Args:
            pose: The basic pose
            action: The action being performed
            expression: The facial expression
            additional_poses: Additional pose specifications
        """
        self.pose = pose
        self.action = action
        self.expression = expression
        self.additional_poses = additional_poses or []

    def get_tags(self) -> List[str]:
        """Get pose-related tags

        Returns:
            A list of pose tags
        """
        tags = []

        # Add pose
        if self.pose:
            tags.append(self.pose)

        # Add action
        if self.action:
            tags.append(self.action)

        # Add expression
        if self.expression:
            tags.append(self.expression)

        # Add additional poses
        tags.extend(self.additional_poses)

        return tags


class SettingComponent(PromptComponent):
    """Component for setting-related tags"""

    def __init__(
        self,
        location: str = None,
        time_of_day: str = None,
        weather: str = None,
        additional_settings: List[str] = None,
        use_nlp_description: bool = False,
        mood: str = None,
        season: str = None,
    ):
        """Initialize a setting component

        Args:
            location: The location
            time_of_day: The time of day
            weather: The weather
            additional_settings: Additional setting specifications
            use_nlp_description: Whether to use NLP for generating rich descriptions
            mood: The mood/atmosphere of the scene
            season: The season setting
        """
        self.location = location
        self.time_of_day = time_of_day
        self.weather = weather
        self.additional_settings = additional_settings or []
        self.use_nlp_description = use_nlp_description
        self.mood = mood
        self.season = season

    def get_tags(self) -> List[str]:
        """Get setting-related tags

        Returns:
            A list of setting tags
        """
        tags = []

        # If using NLP description and we have location
        if self.use_nlp_description and self.location:
            # Generate a rich background description
            background_desc = generate_background_description(
                self.location,
                self.time_of_day,
                self.weather,
                self.season,
                self.mood,
            )
            tags.append(background_desc)

            # Add any additional settings not covered in the description
            for setting in self.additional_settings:
                # Only add if not likely to be in the description
                if setting.lower() not in background_desc.lower():
                    tags.append(setting)

            return tags

        # Traditional tag generation if not using NLP
        # Add location
        if self.location:
            tags.append(self.location)

        # Add time of day
        if self.time_of_day:
            tags.append(self.time_of_day)

        # Add weather
        if self.weather:
            tags.append(self.weather)

        # Add season
        if self.season:
            tags.append(self.season)

        # Add mood
        if self.mood:
            tags.append(self.mood)

        # Add additional settings
        tags.extend(self.additional_settings)

        return tags

    def suggest_locations_for_species(self, species: str) -> List[str]:
        """Suggest appropriate locations for a given species

        Args:
            species: The species to find locations for

        Returns:
            A list of suitable locations
        """
        return get_complementary_locations(species)


class StyleComponent(PromptComponent):
    """Component for style-related tags"""

    def __init__(
        self,
        art_style: str = None,
        artist: str = None,
        medium: str = None,
        additional_styles: List[str] = None,
        checkpoint_name: str = None,
    ):
        """Initialize a style component

        Args:
            art_style: The art style
            artist: The artist to emulate
            medium: The medium (e.g., digital, oil painting)
            additional_styles: Additional style specifications
            checkpoint_name: Name of the checkpoint model (used for formatting artist)
        """
        self.art_style = art_style
        self.artist = artist
        self.medium = medium
        self.additional_styles = additional_styles or []
        self.checkpoint_name = checkpoint_name

    def get_tags(self) -> List[str]:
        """Get style-related tags

        Returns:
            A list of style tags
        """
        tags = []

        # Add art style
        if self.art_style:
            tags.append(self.art_style)

        # Add artist
        if self.artist:
            # Check if we need to add "by " prefix based on checkpoint
            if self.checkpoint_name and "noobaiXLVpredv10" in self.checkpoint_name:
                # NoobAI model doesn't need "by " prefix
                tags.append(self.artist)
            else:
                # Other models including ponyDiffusion need "by " prefix
                if not self.artist.lower().startswith("by "):
                    tags.append(f"by {self.artist}")
                else:
                    tags.append(self.artist)

        # Add medium
        if self.medium:
            tags.append(self.medium)

        # Add additional styles
        tags.extend(self.additional_styles)

        return tags


class NegativePromptComponent(PromptComponent):
    """Component for negative prompt tags"""

    def __init__(self, is_nsfw: bool = False, custom_negative: List[str] = None):
        """Initialize a negative prompt component

        Args:
            is_nsfw: Whether this is for NSFW content (adds censorship negations)
            custom_negative: Custom negative tags to add
        """
        self.is_nsfw = is_nsfw
        self.custom_negative = custom_negative or []

    def get_tags(self) -> List[str]:
        """Get negative prompt tags

        Returns:
            A list of negative prompt tags
        """
        # Standard negative tags for quality issues
        tags = [
            "low quality",
            "worst quality",
            "bad anatomy",
            "bad proportions",
            "deformed",
            "disfigured",
            "malformed limbs",
            "missing limbs",
            "extra limbs",
            "poorly drawn face",
            "poorly drawn hands",
            "blurry",
            "fuzzy",
            "grainy",
        ]

        # Add NSFW-specific negative tags if this is for NSFW content
        if self.is_nsfw:
            tags.extend(
                [
                    "censored",
                    "mosaic censoring",
                    "bar censor",
                    "blur censor",
                ]
            )

        # Add custom negative tags
        tags.extend(self.custom_negative)

        return tags


class PromptGenerator:
    """Base class for prompt generators"""

    def __init__(self):
        """Initialize a prompt generator"""
        self.components = []
        self.negative_component = None
        self.checkpoint_name = None
        self.lora_name = None
        # Note: The random seed is set at the CLI level before generator creation
        # This ensures deterministic prompt generation with the same seed

    def add_component(self, component: PromptComponent) -> None:
        """Add a component to the generator

        Args:
            component: The component to add
        """
        self.components.append(component)

    def set_models(self, checkpoint_name: str, lora_name: str = None) -> None:
        """Set the checkpoint and LoRA models to use

        Args:
            checkpoint_name: Name of the checkpoint
            lora_name: Name of the LoRA
        """
        self.checkpoint_name = checkpoint_name
        self.lora_name = lora_name

    def set_negative_component(self, component: NegativePromptComponent) -> None:
        """Set the negative prompt component

        Args:
            component: The negative prompt component
        """
        self.negative_component = component

    def get_model_prefix(self) -> str:
        """Get the appropriate prompt prefix based on the checkpoint model

        Returns:
            A model-specific prefix string
        """
        if not self.checkpoint_name:
            # Default generic prefix
            return "masterpiece, best quality, high quality, detailed"

        # NoobAI models
        if "noobai" in self.checkpoint_name.lower():
            return "masterpiece, best quality, newest, absurdres, highres"

        # PonyDiffusion models
        if "ponydiffusion" in self.checkpoint_name.lower():
            return "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"

        # Default prefix for other models
        return "masterpiece, best quality, high quality, detailed"

    def generate(self) -> str:
        """Generate a prompt

        Returns:
            A complete prompt string
        """
        all_tags = []

        # Add model-specific prefix
        model_prefix = self.get_model_prefix()
        if model_prefix:
            all_tags.append(model_prefix)

        # Add component tags
        for component in self.components:
            component_tags = component.get_tags()
            all_tags.extend(component_tags)

        # Create a comma-separated prompt string
        prompt = ", ".join(all_tags)
        return prompt

    def get_negative_prompt(self) -> str:
        """Get the negative prompt

        Returns:
            A negative prompt string
        """
        if not self.negative_component:
            return ""

        negative_tags = self.negative_component.get_tags()
        negative_prompt = ", ".join(negative_tags)
        return negative_prompt
