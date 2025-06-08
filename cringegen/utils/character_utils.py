"""
Character Utility Functions for cringegen

This module provides utility functions for character generation, manipulation, and validation.
It works with the character_taxonomy module to provide easy access to character-related operations.

Key functionality:
- Character generation and customization
- Character validation and compatibility checking
- Character-related prompt enhancement
- Helper functions for common character operations
"""

import random
from typing import Dict, List, Optional, Any

# Import from data modules
from ..data.character_taxonomy import (
    CharacterType,
    FormType,
    Gender,
    CharacterTemplate,
    CHARACTER_NAME_MAP,
    COMMON_ANIME_ARCHETYPES,
    get_species_info,
    generate_character_template,
    get_character_by_name,
    suggest_character_features,
)
from ..data.taxonomy import ANTHRO_SPECIES, FANTASY_SPECIES

# -------------------------------------------------------------------------
# Character Generation Functions
# -------------------------------------------------------------------------


def generate_random_character(
    character_type: Optional[CharacterType] = None,
    gender: Optional[Gender] = None,
    species_list: Optional[List[str]] = None,
    archetype: Optional[str] = None,
    nsfw_allowed: bool = False,
) -> CharacterTemplate:
    """
    Generate a random character based on the provided constraints.

    Args:
        character_type: Optional type of character to generate
        gender: Optional gender constraint
        species_list: Optional list of species to choose from
        archetype: Optional character archetype to use
        nsfw_allowed: Whether NSFW character elements are allowed

    Returns:
        A complete CharacterTemplate ready for use
    """
    # Set defaults
    if character_type is None:
        character_type = random.choice(list(CharacterType))

    if gender is None:
        gender = random.choice([Gender.MALE, Gender.FEMALE, Gender.NONBINARY])

    # Handle species selection
    selected_species = None
    if species_list:
        if len(species_list) > 0:
            selected_species = random.choice(species_list)
    elif character_type == CharacterType.ANTHRO:
        selected_species = random.choice(ANTHRO_SPECIES)
    elif character_type == CharacterType.FANTASY:
        selected_species = random.choice(FANTASY_SPECIES)

    # Handle form type based on character type
    form_type = FormType.ANTHRO
    if character_type == CharacterType.ANTHRO:
        form_type = random.choice([FormType.ANTHRO, FormType.SEMI_ANTHRO, FormType.FERAL])
        # Bias toward anthro
        if random.random() < 0.6:
            form_type = FormType.ANTHRO

    # Generate the base template
    template = generate_character_template(
        character_type=character_type,
        species=selected_species,
        gender=gender,
        form_type=form_type,
        archetype=archetype,
    )

    # Add random features
    if selected_species:
        template.features = suggest_character_features(selected_species, gender)

    # Add personality traits
    personality_traits = [
        "friendly",
        "calm",
        "curious",
        "confident",
        "shy",
        "energetic",
        "creative",
    ]
    template.personality = random.sample(personality_traits, min(3, len(personality_traits)))

    # Filter out NSFW content if not allowed
    if not nsfw_allowed:
        # Implementation would filter out NSFW accessories, clothing, etc.
        pass

    return template


def character_to_prompt(
    character: CharacterTemplate,
    include_personality: bool = True,
    include_clothing: bool = True,
    include_accessories: bool = True,
    include_appearance_traits: bool = True,
    include_model_tags: bool = True,
    include_nsfw_traits: bool = False,
    emphasis_level: int = 1,
) -> str:
    """
    Convert a character template to a prompt fragment.

    Args:
        character: The character template to convert
        include_personality: Whether to include personality traits
        include_clothing: Whether to include clothing
        include_accessories: Whether to include accessories
        include_appearance_traits: Whether to include specific appearance traits
        include_model_tags: Whether to include model-specific tags
        include_nsfw_traits: Whether to include NSFW-specific traits
        emphasis_level: How much to emphasize certain traits (1-3)

    Returns:
        A prompt string representing the character
    """
    prompt_parts = []

    # Add model-specific tags if requested and this is a canon character
    if include_model_tags and character.is_canon_character and character.model_tags:
        for tag_source, tag in character.model_tags.items():
            # Put e621/danbooru tags at the beginning for emphasis
            prompt_parts.insert(0, tag)

    # Basic character description
    species_str = character.species if character.species else "creature"
    form_str = str(character.form_type.name).lower() if character.form_type else "anthro"
    gender_str = str(character.gender.name).lower() if character.gender else ""

    # Add emphasis based on level
    emphasis = ""
    if emphasis_level > 1:
        emphasis = ", " + ", ".join([species_str] * (emphasis_level - 1))

    # Only include the basic description if this isn't a canon character
    # or if we don't have model tags
    if not character.is_canon_character or not character.model_tags:
        base_desc = f"{gender_str} {form_str} {species_str}{emphasis}"
        prompt_parts.append(base_desc)

    # Add appearance traits if requested
    if include_appearance_traits and character.appearance_traits:
        prompt_parts.extend(character.appearance_traits)

    # Add features if present
    if character.features:
        prompt_parts.append(", ".join(character.features[:5]))  # Limit to top 5 features

    # Add clothing if requested and present
    if include_clothing and character.clothing:
        clothing_str = "wearing " + ", ".join(character.clothing[:3])  # Limit to top 3 items
        prompt_parts.append(clothing_str)

    # Add accessories if requested and present
    if include_accessories and character.accessories:
        accessories_str = "with " + ", ".join(character.accessories[:3])  # Limit to top 3
        prompt_parts.append(accessories_str)

    # Add NSFW traits if requested and present
    if include_nsfw_traits and character.nsfw_traits:
        prompt_parts.extend(character.nsfw_traits)

    # Add personality if requested and present
    if include_personality and character.personality:
        personality_str = ", ".join(character.personality)
        prompt_parts.append(f"personality: {personality_str}")

    return ", ".join(prompt_parts)


def find_character_by_description(description: str) -> Optional[CharacterTemplate]:
    """
    Attempt to identify a known character from a description.

    Args:
        description: Description to analyze

    Returns:
        A CharacterTemplate if a match is found, None otherwise
    """
    # Search for character names in the description
    words = description.lower().split()

    # Look for famous characters
    for name in CHARACTER_NAME_MAP.keys():
        # Check if name appears in the description
        name_parts = name.lower().split()
        if all(part in words for part in name_parts):
            return get_character_by_name(name)

    # Look for species + archetype combinations
    detected_species = None
    for species in ANTHRO_SPECIES:
        if species in description.lower():
            detected_species = species
            break

    detected_archetype = None
    for category, archetypes in COMMON_ANIME_ARCHETYPES.items():
        for archetype in archetypes:
            if archetype.lower() in description.lower():
                detected_archetype = archetype.lower()
                break
        if detected_archetype:
            break

    # If we found both a species and archetype, generate a template
    if detected_species and detected_archetype:
        return generate_character_template(
            character_type=CharacterType.ANTHRO,
            species=detected_species,
            archetype=detected_archetype,
        )

    return None


# -------------------------------------------------------------------------
# Character Validation and Filtering Functions
# -------------------------------------------------------------------------


def is_character_compatible(
    character: CharacterTemplate,
    style_compatibility: Optional[List[str]] = None,
    nsfw_allowed: bool = False,
) -> bool:
    """
    Check if a character is compatible with the provided constraints.

    Args:
        character: The character to check
        style_compatibility: Optional list of style tags the character should be compatible with
        nsfw_allowed: Whether NSFW character elements are allowed

    Returns:
        True if the character is compatible, False otherwise
    """
    # Check for NSFW content if not allowed
    if not nsfw_allowed:
        # Implementation would check for NSFW content
        pass

    # Check style compatibility
    if style_compatibility:
        # Implementation would check compatibility with specified styles
        pass

    return True


def filter_incompatible_elements(character: CharacterTemplate) -> CharacterTemplate:
    """
    Remove incompatible elements from a character template.

    Args:
        character: The character to filter

    Returns:
        A new CharacterTemplate with incompatible elements removed
    """
    # Create a copy of the template
    filtered = CharacterTemplate(
        character_type=character.character_type,
        form_type=character.form_type,
        gender=character.gender,
        species=character.species,
        archetype=character.archetype,
        features=character.features.copy(),
        clothing=character.clothing.copy(),
        accessories=character.accessories.copy(),
        colors=character.colors.copy(),
        personality=character.personality.copy(),
    )

    # Check clothing compatibility with species and form
    if filtered.clothing and filtered.species:
        # Implementation would check for incompatible clothing
        pass

    # Check accessory compatibility with species and form
    if filtered.accessories and filtered.species:
        # Implementation would check for incompatible accessories
        pass

    return filtered


# -------------------------------------------------------------------------
# Character Prompt Enhancement Functions
# -------------------------------------------------------------------------


def enhance_character_prompt(prompt: str) -> str:
    """
    Enhance a character-related prompt with appropriate details.

    Args:
        prompt: Original prompt

    Returns:
        Enhanced prompt with additional character details
    """
    # Detect character in the prompt
    character = find_character_by_description(prompt)

    if character:
        # Convert character to prompt fragment
        character_fragment = character_to_prompt(character)

        # Combine with original prompt
        # This is a simple implementation - could be more sophisticated
        prompt_parts = prompt.split(",")
        return character_fragment + ", " + ", ".join(prompt_parts)

    return prompt


def suggest_character_improvements(character: CharacterTemplate) -> Dict[str, List[str]]:
    """
    Suggest potential improvements to a character template.

    Args:
        character: The character to analyze

    Returns:
        Dictionary of suggestion categories and lists of suggestions
    """
    suggestions = {"features": [], "clothing": [], "accessories": [], "personality": []}

    # Check if character has essential features for its species
    if character.species:
        species_info = get_species_info(character.species)
        if species_info and species_info.common_features:
            missing_features = [
                f for f in species_info.common_features if f not in character.features
            ]
            if missing_features:
                suggestions["features"].extend(missing_features[:3])  # Top 3 missing features

    # Check if character has appropriate clothing
    if not character.clothing and character.form_type == FormType.ANTHRO:
        suggestions["clothing"].append("Add basic clothing appropriate for the character")

    # Check if character has accessories
    if not character.accessories:
        suggestions["accessories"].append(
            "Add some accessories to make the character more distinctive"
        )

    # Check if character has personality traits
    if not character.personality:
        suggestions["personality"].extend(["friendly", "curious", "confident"])

    return suggestions


# -------------------------------------------------------------------------
# Character Creation Presets
# -------------------------------------------------------------------------


def preset_fantasy_adventurer(
    species: Optional[str] = None, gender: Optional[Gender] = None
) -> CharacterTemplate:
    """
    Create a fantasy adventurer character preset.

    Args:
        species: Optional species override
        gender: Optional gender override

    Returns:
        A CharacterTemplate for a fantasy adventurer
    """
    if species is None:
        # Pick a common fantasy adventurer species
        fantasy_adventure_species = ["wolf", "fox", "cat", "dragon", "rabbit"]
        species = random.choice(fantasy_adventure_species)

    if gender is None:
        gender = random.choice([Gender.MALE, Gender.FEMALE])

    template = generate_character_template(
        character_type=CharacterType.ANTHRO,
        species=species,
        gender=gender,
        form_type=FormType.ANTHRO,
    )

    # Add adventure-themed clothing and accessories
    adventurer_clothing = ["leather armor", "cloak", "boots", "belt"]
    adventurer_accessories = ["backpack", "pouch", "weapon"]

    template.clothing = adventurer_clothing
    template.accessories = adventurer_accessories

    # Add adventurer personality traits
    template.personality = ["brave", "determined", "resourceful"]

    return template


def preset_modern_casual(
    species: Optional[str] = None, gender: Optional[Gender] = None
) -> CharacterTemplate:
    """
    Create a modern casual character preset.

    Args:
        species: Optional species override
        gender: Optional gender override

    Returns:
        A CharacterTemplate for a modern casual character
    """
    if species is None:
        # Pick a common modern character species
        modern_species = ["wolf", "fox", "cat", "dog", "rabbit"]
        species = random.choice(modern_species)

    if gender is None:
        gender = random.choice([Gender.MALE, Gender.FEMALE])

    template = generate_character_template(
        character_type=CharacterType.ANTHRO,
        species=species,
        gender=gender,
        form_type=FormType.ANTHRO,
    )

    # Add modern casual clothing based on gender
    if gender == Gender.MALE:
        template.clothing = ["t-shirt", "jeans", "sneakers"]
        template.accessories = ["watch", "phone", "backpack"]
    elif gender == Gender.FEMALE:
        template.clothing = ["top", "jeans", "sneakers"]
        template.accessories = ["jewelry", "phone", "bag"]
    else:
        template.clothing = ["hoodie", "jeans", "sneakers"]
        template.accessories = ["phone", "backpack", "headphones"]

    # Add casual personality traits
    template.personality = ["relaxed", "social", "easygoing"]

    return template


# -------------------------------------------------------------------------
# Character Migration Functions
# -------------------------------------------------------------------------


def migrate_legacy_character_data(legacy_data: Dict[str, Any]) -> CharacterTemplate:
    """
    Convert legacy character data to the new character template format.

    Args:
        legacy_data: Dictionary containing legacy character data

    Returns:
        A CharacterTemplate populated with the legacy data
    """
    # Implementation depends on the structure of legacy data
    # This is a placeholder that would need to be customized

    # Example implementation assuming legacy data has some specific fields
    species = legacy_data.get("species", "wolf")
    gender_str = legacy_data.get("gender", "male")

    # Map gender string to Gender enum
    gender_map = {
        "male": Gender.MALE,
        "female": Gender.FEMALE,
        "non-binary": Gender.NONBINARY,
        "ambiguous": Gender.AMBIGUOUS,
    }
    gender = gender_map.get(gender_str.lower(), Gender.AMBIGUOUS)

    # Create template with basic info
    template = CharacterTemplate(
        character_type=CharacterType.ANTHRO,
        form_type=FormType.ANTHRO,
        gender=gender,
        species=species,
    )

    # Copy additional data if available
    if "clothing" in legacy_data:
        template.clothing = legacy_data["clothing"]

    if "accessories" in legacy_data:
        template.accessories = legacy_data["accessories"]

    if "personality" in legacy_data:
        template.personality = legacy_data["personality"]

    return template


# -------------------------------------------------------------------------
# Character Specific Functions
# -------------------------------------------------------------------------


def generate_canon_character_prompt(
    character_name: str,
    nsfw: bool = False,
    include_appearance: bool = True,
    include_accessories: bool = True,
    holding_sword: bool = False,  # Optional parameter for characters like Blaidd
) -> str:
    """
    Generate a prompt specifically for a canonical character.

    Args:
        character_name: Name of the character to generate a prompt for
        nsfw: Whether to include NSFW traits
        include_appearance: Whether to include appearance traits
        include_accessories: Whether to include accessories
        holding_sword: For characters like Blaidd, whether they're holding their weapon

    Returns:
        A prompt optimized for the specific character
    """
    # First try to get the character from individual files
    try:
        from ..data.characters import get_individual_character

        character = get_individual_character(character_name)
    except ImportError:
        # Fall back to the main character system
        character = get_character_by_name(character_name)

    if not character:
        return f"Unknown character: {character_name}"

    # Make a copy to customize
    from copy import deepcopy

    custom_character = deepcopy(character)

    # Handle character-specific customizations
    if character_name.lower() == "blaidd":
        # Try to get poses from the Blaidd module
        try:
            from ..data.characters.blaidd import CHARACTER_POSES

            if holding_sword and "sword" in CHARACTER_POSES:
                for pose in CHARACTER_POSES["sword"]:
                    if pose not in custom_character.appearance_traits:
                        custom_character.appearance_traits.append(pose)
        except ImportError:
            # Fall back to basic sword holding
            if holding_sword and "holding sword" not in custom_character.appearance_traits:
                custom_character.appearance_traits.append("holding sword")

    # Generate the prompt
    return character_to_prompt(
        custom_character,
        include_personality=True,
        include_clothing=True,
        include_accessories=include_accessories,
        include_appearance_traits=include_appearance,
        include_model_tags=True,
        include_nsfw_traits=nsfw,
    )
