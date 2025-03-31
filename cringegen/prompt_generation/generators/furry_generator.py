"""
Furry-specific prompt generators
"""

import random
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..base import (
    CharacterComponent,
    MultiSubjectComponent,
    NegativePromptComponent,
    PoseComponent,
    PromptComponent,
    PromptGenerator,
    QualityComponent,
    SettingComponent,
    StyleComponent,
)
from ..nlp.background_utils import get_complementary_locations
from ..nlp.color_utils import get_random_pattern, parse_color_input
from ..nlp.species_utils import enhance_prompt_with_anatomy

# Create logger
logger = logging.getLogger(__name__)

class FurryPromptGenerator(PromptGenerator):
    """Generator for SFW furry content"""

    def __init__(
        self,
        species: str = None,
        gender: str = None,
        colors: str = None,
        pattern: str = None,
        species2: str = None,
        gender2: str = None,
        colors2: str = None,
        pattern2: str = None,
        use_duo: bool = False,
        group_size: int = 0,
        use_nlp: bool = True,
        use_art_style: bool = True,
        is_anthro: bool = True,
        is_feral: bool = False,
    ):
        """Initialize a furry prompt generator

        Args:
            species: The species of the character
            gender: The gender of the character
            colors: Comma-separated list of colors for the character
            pattern: Pattern type for the character
            species2: The species of the second character (for duo scenes)
            gender2: The gender of the second character (for duo scenes)
            colors2: Comma-separated list of colors for the second character
            pattern2: Pattern type for the second character
            use_duo: Whether to create a scene with two characters
            group_size: Number of characters for a group scene (overrides duo if > 0)
            use_nlp: Whether to use NLP features for enhanced descriptions
            use_art_style: Whether to include random art style in the prompt
            is_anthro: Whether to use anthro-specific accessories (defaults to True)
            is_feral: Whether to use feral-specific accessories (overrides is_anthro)
        """
        super().__init__()

        # Store user-specified species and gender (may be None)
        self._user_species = species
        self._user_gender = gender

        # Initialize with random values or user-specified values
        self.species = species if species is not None else self._random_species()
        self.gender = gender if gender is not None else self._random_gender()
        self.use_nlp = use_nlp
        self.use_art_style = use_art_style

        # Determine if anthro or feral (feral flag takes precedence)
        self.is_anthro = not is_feral if is_feral else is_anthro

        # Parse color inputs
        self.colors = parse_color_input(colors) if colors else []
        self.pattern = pattern

        # Setup for duo or group scenes
        self.use_duo = use_duo
        self.group_size = max(0, group_size)  # Ensure non-negative

        # Setup for second character if needed
        if self.use_duo or self.group_size > 0:
            self._user_species2 = species2
            self._user_gender2 = gender2
            self.species2 = species2 if species2 is not None else self._random_species()
            self.gender2 = gender2 if gender2 is not None else self._random_gender()
            self.colors2 = parse_color_input(colors2) if colors2 else []
            self.pattern2 = pattern2
        else:
            self._user_species2 = None
            self._user_gender2 = None
            self.species2 = None
            self.gender2 = None
            self.colors2 = []
            self.pattern2 = None

        # Add default components
        self._add_default_components()

    def _add_default_components(self) -> None:
        """Add default components for furry content"""
        # Quality component is no longer needed as we use model-specific prefixes
        # quality = QualityComponent(quality_level=5)
        # self.add_component(quality)

        # Determine environment/location that complements the species
        random_location = self._choose_location_for_species()

        # Handle single character, duo, or group
        if self.group_size > 0:
            # Generate a group scene
            self._add_group_components(random_location)
        elif self.use_duo:
            # Generate a duo scene
            self._add_duo_components(random_location)
        else:
            # Generate a single character scene
            self._add_single_character_components(random_location)

        # Add style component if enabled
        if self.use_art_style:
            style = StyleComponent(
                art_style=self._random_art_style(), checkpoint_name=self.checkpoint_name
            )
            self.add_component(style)

        # Set negative component
        negative = NegativePromptComponent(is_nsfw=False)
        self.set_negative_component(negative)

    def _add_single_character_components(self, location: str) -> None:
        """Add components for a single character scene

        Args:
            location: The location for the scene
        """
        # Get features and add "solo" tag for single character scenes
        features = self._random_features()
        features.append("solo")

        # Add anthro or feral tag based on settings
        if self.is_anthro:
            features.append("anthro")
        else:
            features.append("feral")

        # Log current species and gender
        logger.debug(f"Creating character with species={self.species}, gender={self.gender}")

        # Add character component with NLP if enabled
        character = CharacterComponent(
            species=self.species,
            gender=self.gender,
            features=features,
            colors=self.colors,
            color_pattern=self.pattern,
            use_nlp_description=self.use_nlp,
            is_anthro=self.is_anthro,
        )
        self.add_component(character)

        # Add pose component
        pose = PoseComponent(pose=self._random_pose(), expression=self._random_expression())
        self.add_component(pose)

        # Add setting component with NLP if enabled
        setting = SettingComponent(
            location=location,
            time_of_day=self._random_time_of_day(),
            weather=self._random_weather() if random.random() < 0.3 else None,
            mood=self._random_mood() if random.random() < 0.3 else None,
            season=self._random_season() if random.random() < 0.3 else None,
            use_nlp_description=self.use_nlp,
        )
        self.add_component(setting)

    def _add_duo_components(self, location: str) -> None:
        """Add components for a duo scene

        Args:
            location: The location for the scene
        """
        # Get features for first character
        features1 = self._random_features()
        if self.is_anthro:
            features1.append("anthro")
        else:
            features1.append("feral")

        # Create first character
        character1 = CharacterComponent(
            species=self.species,
            gender=self.gender,
            features=features1,
            colors=self.colors,
            color_pattern=self.pattern,
            use_nlp_description=self.use_nlp,
            is_anthro=self.is_anthro,
        )

        # Get features for second character
        features2 = self._random_features()
        if self.is_anthro:
            features2.append("anthro")
        else:
            features2.append("feral")

        # Create second character
        character2 = CharacterComponent(
            species=self.species2,
            gender=self.gender2,
            features=features2,
            colors=self.colors2,
            color_pattern=self.pattern2,
            use_nlp_description=self.use_nlp,
            is_anthro=self.is_anthro,
        )

        # Create interaction elements
        interaction = self._random_interaction()
        arrangement = self._random_arrangement()

        # Create multi-subject component
        duo = MultiSubjectComponent(
            characters=[character1, character2],
            interaction=interaction,
            arrangement=arrangement,
        )
        self.add_component(duo)

        # Add setting component with NLP if enabled
        setting = SettingComponent(
            location=location,
            time_of_day=self._random_time_of_day(),
            weather=self._random_weather() if random.random() < 0.3 else None,
            mood=self._random_mood() if random.random() < 0.3 else None,
            season=self._random_season() if random.random() < 0.3 else None,
            use_nlp_description=self.use_nlp,
        )
        self.add_component(setting)

    def _add_group_components(self, location: str) -> None:
        """Add components for a group scene

        Args:
            location: The location for the scene
        """
        # Limit group size to reasonable number
        group_size = min(max(3, self.group_size), 6)

        # Create characters for the group
        characters = []

        # Get features for first character
        features1 = self._random_features()
        if self.is_anthro:
            features1.append("anthro")
        else:
            features1.append("feral")

        # First character (the main one specified by user)
        characters.append(
            CharacterComponent(
                species=self.species,
                gender=self.gender,
                features=features1,
                colors=self.colors,
                color_pattern=self.pattern,
                use_nlp_description=self.use_nlp,
                is_anthro=self.is_anthro,
            )
        )

        # Get features for second character if specified
        if self.species2:
            features2 = self._random_features()
            if self.is_anthro:
                features2.append("anthro")
            else:
                features2.append("feral")

            characters.append(
                CharacterComponent(
                    species=self.species2,
                    gender=self.gender2,
                    features=features2,
                    colors=self.colors2,
                    color_pattern=self.pattern2,
                    use_nlp_description=self.use_nlp,
                    is_anthro=self.is_anthro,
                )
            )
            remaining = group_size - 2
        else:
            remaining = group_size - 1

        # Generate remaining random characters
        for _ in range(remaining):
            random_species = self._random_species()
            random_features = self._random_features()

            if self.is_anthro:
                random_features.append("anthro")
            else:
                random_features.append("feral")

            characters.append(
                CharacterComponent(
                    species=random_species,
                    gender=self._random_gender(),
                    features=random_features,
                    use_nlp_description=self.use_nlp,
                    is_anthro=self.is_anthro,
                )
            )

        # Create group interaction elements
        interaction = self._random_group_interaction()
        arrangement = self._random_group_arrangement()

        # Create multi-subject component
        group = MultiSubjectComponent(
            characters=characters,
            interaction=interaction,
            arrangement=arrangement,
        )
        self.add_component(group)

        # Add setting component with NLP if enabled
        setting = SettingComponent(
            location=location,
            time_of_day=self._random_time_of_day(),
            weather=self._random_weather() if random.random() < 0.3 else None,
            mood=self._random_mood() if random.random() < 0.3 else None,
            season=self._random_season() if random.random() < 0.3 else None,
            use_nlp_description=self.use_nlp,
        )
        self.add_component(setting)

    def _random_interaction(self) -> str:
        """Get a random interaction for duo scenes

        Returns:
            A random interaction string
        """
        interaction_list = [
            "talking to each other",
            "laughing together",
            "playing together",
            "high-fiving",
            "hugging",
            "facing each other",
            "walking together",
            "sitting together",
            "enjoying each other's company",
            "sharing a moment",
            "playing a game",
            "relaxing together",
        ]
        return random.choice(interaction_list)

    def _random_arrangement(self) -> str:
        """Get a random arrangement for duo scenes

        Returns:
            A random arrangement string
        """
        arrangement_list = [
            "side by side",
            "facing each other",
            "one behind the other",
            "close together",
            "at arm's length",
        ]
        return random.choice(arrangement_list)

    def _random_group_interaction(self) -> str:
        """Get a random interaction for group scenes

        Returns:
            A random interaction string
        """
        interaction_list = [
            "gathered together",
            "engaged in conversation",
            "celebrating",
            "at a party",
            "enjoying a meal together",
            "playing a game",
            "on an adventure",
            "relaxing together",
            "taking a group photo",
            "having fun together",
        ]
        return random.choice(interaction_list)

    def _random_group_arrangement(self) -> str:
        """Get a random arrangement for group scenes

        Returns:
            A random arrangement string
        """
        arrangement_list = [
            "in a circle",
            "in a line",
            "clustered together",
            "scattered around",
            "posing together",
            "sitting around a table",
            "standing in formation",
        ]
        return random.choice(arrangement_list)

    def _choose_location_for_species(self) -> str:
        """Choose a location that complements the species

        Returns:
            A suitable location string
        """
        # Get complementary locations for the species
        suitable_locations = get_complementary_locations(self.species)

        # If no suitable locations or we want to randomize completely
        if not suitable_locations or random.random() < 0.2:
            return self._random_location()

        # Return a random location from the suitable ones
        return random.choice(suitable_locations)

    def _random_species(self) -> str:
        """Get a random furry species

        Returns:
            A random species
        """
        species_list = [
            "wolf",
            "fox",
            "cat",
            "dog",
            "rabbit",
            "tiger",
            "lion",
            "dragon",
            "deer",
            "horse",
            "raccoon",
            "otter",
            "bear",
            "sergal",
            "protogen",
            "skunk",
        ]
        return random.choice(species_list)

    def _random_gender(self) -> str:
        """Get a random gender

        Returns:
            A random gender
        """
        gender_list = ["male", "female"]
        return random.choice(gender_list)

    def _random_features(self) -> List[str]:
        """Get random character features

        Returns:
            A list of random features
        """
        feature_pools = {
            "clothing": [
                "t-shirt",
                "hoodie",
                "jacket",
                "jeans",
                "shorts",
                "dress",
                "sweater",
                "tank top",
                "casual clothes",
                "formal attire",
            ],
            "accessories": [
                "glasses",
                "scarf",
                "hat",
                "necklace",
                "earrings",
                "bracelet",
                "watch",
                "backpack",
                "bag",
                "headphones",
            ],
            "attributes": [
                "cute",
                "fluffy",
                "friendly",
                "smiling",
                "cheerful",
                "athletic",
                "strong",
                "slender",
                "tall",
                "short",
                "muscular",
            ],
        }

        # Select 1-2 features from each category
        selected_features = []
        for category, features in feature_pools.items():
            count = random.randint(1, 2)
            selected = random.sample(features, min(count, len(features)))
            selected_features.extend(selected)

        return selected_features

    def _random_pose(self) -> str:
        """Get a random pose

        Returns:
            A random pose
        """
        pose_list = [
            "standing",
            "sitting",
            "walking",
            "running",
            "jumping",
            "leaning",
            "crouching",
            "kneeling",
            "lying down",
            "floating",
        ]
        return random.choice(pose_list)

    def _random_expression(self) -> str:
        """Get a random expression

        Returns:
            A random expression
        """
        expression_list = [
            "smiling",
            "laughing",
            "grinning",
            "cheerful",
            "happy",
            "calm",
            "serious",
            "thoughtful",
            "curious",
            "excited",
        ]
        return random.choice(expression_list)

    def _random_location(self) -> str:
        """Get a random location

        Returns:
            A random location
        """
        location_list = [
            "forest",
            "beach",
            "mountains",
            "city",
            "park",
            "meadow",
            "river",
            "lake",
            "desert",
            "jungle",
            "coffee shop",
            "library",
            "school",
        ]
        return random.choice(location_list)

    def _random_time_of_day(self) -> str:
        """Get a random time of day

        Returns:
            A random time of day
        """
        time_list = ["morning", "afternoon", "evening", "night", "twilight", "noon"]
        return random.choice(time_list)

    def _random_weather(self) -> str:
        """Get a random weather condition

        Returns:
            A random weather condition
        """
        weather_list = [
            "sunny",
            "cloudy",
            "rainy",
            "foggy",
            "snowy",
            "stormy",
            "misty",
        ]
        return random.choice(weather_list)

    def _random_season(self) -> str:
        """Get a random season

        Returns:
            A random season
        """
        season_list = ["spring", "summer", "autumn", "winter"]
        return random.choice(season_list)

    def _random_mood(self) -> str:
        """Get a random mood/atmosphere

        Returns:
            A random mood
        """
        mood_list = [
            "peaceful",
            "happy",
            "mysterious",
            "romantic",
            "melancholic",
            "dramatic",
            "magical",
            "cozy",
        ]
        return random.choice(mood_list)

    def _random_art_style(self) -> str:
        """Get a random art style

        Returns:
            A random art style
        """
        # Art styles
        style_list = [
            # Basic styles
            "anime",
            "cartoon",
            "realistic",
            "semi-realistic",
            "digital art",
            "digital painting",
            "illustration",
            "concept art",
            # Advanced styles
            "painterly",
            "stylized",
            "comic",
            "manga",
            "chibi",
            "cel shaded",
            "flat color",
            "minimalist",
            "detailed",
            "hyper-detailed",
            "sketch",
            "line art",
            "abstract",
            "surreal",
            "fantasy",
            "cyberpunk",
            "retro",
            "graffiti",
            "gritty",
            "impressionist",
            "expressionist",
            "3D render",
            "watercolor",
            "acrylic",
            "oil painting",
            "pastel",
            "pop art",
            "vector art",
            "pixel art",
            "psychedelic",
            "photorealistic",
            "low poly",
            "storybook illustration",
            "airbrushed",
            "vibrant colors",
            "dark theme",
            "gothic",
            "cinematic",
            "portrait",
            "landscape",
            "isometric",
        ]

        return random.choice(style_list)


class NsfwComponent(PromptComponent):
    """Component for NSFW content"""

    def __init__(self, explicit_level: int = 1, species: str = None, gender: str = None):
        """Initialize an NSFW component

        Args:
            explicit_level: Level of explicitness (1-3)
            species: Species of the character
            gender: Gender of the character
        """
        self.explicit_level = max(1, min(3, explicit_level))
        self.species = species
        self.gender = gender

    def get_tags(self) -> List[str]:
        """Get NSFW-related tags

        Returns:
            A list of NSFW tags
        """
        # Base NSFW tags
        tags = []

        # Add more explicit tags based on level
        if self.explicit_level >= 2:
            tags.extend(["questionable"])

        if self.explicit_level >= 3:
            tags.extend(["nsfw"])

        return tags


class NsfwFurryPromptGenerator(FurryPromptGenerator):
    """Generator for NSFW furry content"""

    def __init__(
        self,
        species: str = None,
        gender: str = None,
        explicit_level: int = 1,
        use_anatomical_terms: bool = True,
        use_nlp: bool = True,
        use_art_style: bool = True,
        is_anthro: bool = True,
        is_feral: bool = False,
        use_human_genitalia: bool = False,
    ):
        """Initialize an NSFW furry prompt generator

        Args:
            species: The species of the character
            gender: The gender of the character
            explicit_level: Level of explicitness (1-3)
            use_anatomical_terms: Whether to use species-specific anatomical terms
            use_nlp: Whether to use NLP features for enhanced descriptions
            use_art_style: Whether to include random art style in the prompt
            is_anthro: Whether the character is anthro (True) or feral (False)
            is_feral: Whether to use feral-specific features (overrides is_anthro)
            use_human_genitalia: Whether to use human genitalia instead of animal genitalia
        """
        self.explicit_level = max(1, min(3, explicit_level))
        self.use_anatomical_terms = use_anatomical_terms
        self.use_human_genitalia = use_human_genitalia

        # Initialize the parent class
        super().__init__(
            species=species,
            gender=gender,
            use_nlp=use_nlp,
            use_art_style=use_art_style,
            is_anthro=is_anthro,
            is_feral=is_feral,
        )

    def _add_default_components(self) -> None:
        """Add default components for NSFW furry content"""
        # Use the parent class method but save and remove the style component from parent
        # We'll use this approach to remove the parent's style component
        # and replace it with our own that includes checkpoint_name
        super()._add_default_components()

        # Find and remove the StyleComponent added by parent class
        style_component_index = None
        for i, component in enumerate(self.components):
            if isinstance(component, StyleComponent):
                style_component_index = i
                break

        if style_component_index is not None:
            self.components.pop(style_component_index)

        # Add our own style component with checkpoint name if art styles are enabled
        if self.use_art_style:
            style = StyleComponent(
                art_style=self._random_art_style(), checkpoint_name=self.checkpoint_name
            )
            self.add_component(style)

        # Add NSFW component
        nsfw = NsfwComponent(
            explicit_level=self.explicit_level, species=self.species, gender=self.gender
        )
        self.add_component(nsfw)

        # Set negative component for NSFW
        negative = NegativePromptComponent(is_nsfw=True)
        self.set_negative_component(negative)

    def generate(self) -> str:
        """Generate a prompt

        Returns:
            A complete prompt string
        """
        # Generate the base prompt
        prompt = super().generate()

        # Enhance with anatomical terms if enabled
        if self.use_anatomical_terms and self.species and self.gender:
            # If using human genitalia, override the anatomical terms
            if self.use_human_genitalia:
                if self.gender.lower() == "male":
                    terms = ["penis"]
                    if self.explicit_level >= 2:
                        terms.append("testicles")
                    if self.explicit_level >= 3:
                        terms.append("erect penis")
                    conjunction = random.choice(["with", "showing", "displaying", "presenting"])
                    prompt = f"{prompt}, {conjunction} {', '.join(terms)}"
                elif self.gender.lower() == "female":
                    terms = ["vagina"]
                    if self.explicit_level >= 2:
                        terms.append("pussy")
                    if self.explicit_level >= 3:
                        terms.append("exposed vagina")
                    conjunction = random.choice(["with", "showing", "displaying", "presenting"])
                    prompt = f"{prompt}, {conjunction} {', '.join(terms)}"
            else:
                # Use species-specific anatomical terms
                prompt = enhance_prompt_with_anatomy(
                    prompt, self.species, self.gender, self.explicit_level
                )

        return prompt

    def _random_pose(self) -> str:
        """Get a random pose suitable for NSFW content

        Returns:
            A random pose
        """
        # Regular poses
        regular_poses = [
            "standing",
            "sitting",
            "leaning",
            "kneeling",
            "lying down",
        ]

        # NSFW-specific poses
        nsfw_poses = [
            "on all fours",
            "bent over",
            "legs spread",
            "on back",
            "submissive pose",
            "provocative pose",
            "seductive pose",
            "teasing pose",
        ]

        # Mix of regular and NSFW poses based on explicit level
        if self.explicit_level == 1:
            # Mostly regular poses, few NSFW
            poses = regular_poses + random.sample(nsfw_poses, 2)
        elif self.explicit_level == 2:
            # Even mix
            poses = regular_poses + random.sample(nsfw_poses, 4)
        else:
            # Mostly NSFW poses
            poses = random.sample(regular_poses, 2) + nsfw_poses

        return random.choice(poses)

    def _random_expression(self) -> str:
        """Get a random expression suitable for NSFW content

        Returns:
            A random expression
        """
        # Regular expressions
        regular_expressions = [
            "smiling",
            "laughing",
            "cheerful",
            "happy",
            "calm",
            "serious",
        ]

        # NSFW-specific expressions
        nsfw_expressions = [
            "flirty",
            "seductive",
            "lustful",
            "passionate",
            "excited",
            "aroused",
            "blushing",
            "embarrassed",
        ]

        # Mix based on explicit level
        if self.explicit_level == 1:
            # Mostly regular expressions, few NSFW
            expressions = regular_expressions + random.sample(nsfw_expressions, 2)
        elif self.explicit_level == 2:
            # Even mix
            expressions = regular_expressions + random.sample(nsfw_expressions, 4)
        else:
            # Mostly NSFW expressions
            expressions = random.sample(regular_expressions, 2) + nsfw_expressions

        return random.choice(expressions)

    def _random_location(self) -> str:
        """Get a random location suitable for NSFW content

        Returns:
            A random location
        """
        # Regular locations
        regular_locations = [
            "forest",
            "beach",
            "mountains",
            "meadow",
            "river",
            "lake",
        ]

        # NSFW-friendly locations
        nsfw_locations = [
            "bedroom",
            "hotel room",
            "private room",
            "secluded area",
            "hot spring",
            "shower",
            "bath",
        ]

        # Mix based on explicit level
        if self.explicit_level == 1:
            # Mostly regular locations, few NSFW
            locations = regular_locations + random.sample(nsfw_locations, 2)
        elif self.explicit_level == 2:
            # Even mix
            locations = regular_locations + random.sample(
                nsfw_locations, min(4, len(nsfw_locations))
            )
        else:
            # Mostly NSFW locations
            locations = random.sample(regular_locations, 2) + nsfw_locations

        return random.choice(locations)

    def _random_features(self) -> List[str]:
        """Get random character features appropriate for NSFW content

        Returns:
            A list of random features
        """
        feature_pools = {
            "clothing": [
                "t-shirt",
                "hoodie",
                "jacket",
                "jeans",
                "shorts",
                "dress",
                "sweater",
                "tank top",
                "casual clothes",
                "formal attire",
                "lingerie",
                "underwear",
            ],
            "accessories": [
                "glasses",
                "scarf",
                "hat",
                "necklace",
                "earrings",
                "bracelet",
                "watch",
                "backpack",
                "bag",
                "headphones",
                "chains",
                "collar",
            ],
            "attributes": [
                "cute",
                "fluffy",
                "athletic",
                "strong",
                "slender",
                "tall",
                "short",
                "muscular",
                "curvy",
                "thick",
                "slim",
                "toned",
                "feminine",
                "masculine",
                "androgynous",
                "naked",
            ],
        }

        # Select features from each category
        selected_features = []

        # For explicit level 3, skip clothing
        if self.explicit_level == 3:
            # Only add "naked" from attributes and some accessories
            selected_features.append("naked")
            count = random.randint(1, 2)
            selected_accessories = random.sample(
                feature_pools["accessories"], min(count, len(feature_pools["accessories"]))
            )
            selected_features.extend(selected_accessories)
        else:
            # Add clothing for explicit levels 1-2
            for category, features in feature_pools.items():
                count = random.randint(1, 2)
                selected = random.sample(features, min(count, len(features)))
                selected_features.extend(selected)

        return selected_features

    def _random_art_style(self) -> str:
        """Get a random art style

        Returns:
            A random art style
        """
        # Art styles
        style_list = [
            # Basic styles
            "anime",
            "cartoon",
            "realistic",
            "semi-realistic",
            "digital art",
            "digital painting",
            "illustration",
            "concept art",
            # Advanced styles
            "painterly",
            "stylized",
            "comic",
            "manga",
            "chibi",
            "cel shaded",
            "flat color",
            "minimalist",
            "detailed",
            "hyper-detailed",
            "sketch",
            "line art",
            "abstract",
            "surreal",
            "fantasy",
            "cyberpunk",
            "retro",
            "graffiti",
            "gritty",
            "impressionist",
            "expressionist",
            "3D render",
            "watercolor",
            "acrylic",
            "oil painting",
            "pastel",
            "pop art",
            "vector art",
            "pixel art",
            "psychedelic",
            "photorealistic",
            "low poly",
            "storybook illustration",
            "airbrushed",
            "vibrant colors",
            "dark theme",
            "gothic",
            "cinematic",
            "portrait",
            "landscape",
            "isometric",
        ]

        return random.choice(style_list)
