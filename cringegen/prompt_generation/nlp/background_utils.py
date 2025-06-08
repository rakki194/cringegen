"""
Utility functions for background and setting-related natural language processing.
"""

import random
from typing import List, Optional

# Update imports to use new centralized data module
from ...data import backgrounds, habitats

# Original import for reference:
# from cringegen.data.background_data import (
#     BACKGROUND_SETTINGS,
#     MOOD_DESCRIPTORS,
#     SEASONS,
#     SPECIES_HABITATS,
#     TIME_OF_DAY,
#     WEATHER_CONDITIONS,
# )

# Map old variables to new module structure
BACKGROUND_SETTINGS = backgrounds.BACKGROUND_SETTINGS
MOOD_DESCRIPTORS = backgrounds.MOOD_DESCRIPTORS
SEASONS = backgrounds.SEASONS
SPECIES_HABITATS = habitats.SPECIES_HABITATS
TIME_OF_DAY = backgrounds.TIME_OF_DAY
WEATHER_CONDITIONS = backgrounds.WEATHER_CONDITIONS

from ...utils.prompt_utils import get_indefinite_article


def generate_background_description(
    location: str,
    time_of_day: Optional[str] = None,
    weather: Optional[str] = None,
    season: Optional[str] = None,
    mood: Optional[str] = None,
) -> str:
    """Generate a detailed description of a background setting

    Args:
        location: The main location (forest, city, beach, etc.)
        time_of_day: Time of day (morning, noon, evening, night)
        weather: Weather condition (sunny, rainy, cloudy, etc.)
        season: Season (spring, summer, autumn, winter)
        mood: Mood/atmosphere (peaceful, mysterious, etc.)

    Returns:
        A detailed background description
    """
    # Normalize inputs to lowercase for dictionary lookups
    location = location.lower() if location else "forest"
    time_of_day = time_of_day.lower() if time_of_day else random.choice(list(TIME_OF_DAY.keys()))
    weather = weather.lower() if weather else None
    season = season.lower() if season else None
    mood = mood.lower() if mood else None

    # Get location data
    location_data = BACKGROUND_SETTINGS.get(location, BACKGROUND_SETTINGS["forest"])

    # Select a descriptor for the location
    location_descriptor = random.choice(location_data["descriptors"])

    # Select a feature or two for the location
    location_features = random.sample(
        location_data["features"], min(2, len(location_data["features"]))
    )

    # Create the base description with proper article
    article = get_indefinite_article(location_descriptor)
    description = f"{article} {location_descriptor} {location}"

    # Add location modifier if available
    if "modifiers" in location_data and random.random() < 0.5:
        location_modifier = random.choice(location_data["modifiers"])
        # Use different natural phrasings based on the location type
        if location_modifier in ["tropical", "rainforest", "jungle", "alpine", "coastal", 
                               "highland", "valley", "mountain", "desert"]:
            # Get proper article for the new phrase
            article = get_indefinite_article(f"{location_descriptor} {location_modifier}")
            description = f"{article} {location_descriptor} {location_modifier} {location}"
        else:
            # Different phrase options with trees
            phrases = [
                f"{get_indefinite_article(f'{location_descriptor} {location_modifier}')} {location_descriptor} {location_modifier} {location}",
                f"{get_indefinite_article(location_descriptor)} {location_descriptor} {location} with {location_modifier} trees",
                f"{get_indefinite_article(location_descriptor)} {location_descriptor} {location} dotted with {location_modifier} trees",
                f"{get_indefinite_article(location_descriptor)} {location_descriptor} {location} surrounded by {location_modifier} trees",
            ]
            description = random.choice(phrases)

    # Add time of day information
    time_data = TIME_OF_DAY.get(time_of_day, TIME_OF_DAY["afternoon"])
    time_lighting = random.choice(time_data["lighting"])
    description = f"{description} during {time_of_day}, with {time_lighting}"

    # Add weather if specified
    if weather and weather in WEATHER_CONDITIONS:
        weather_data = WEATHER_CONDITIONS[weather]
        weather_feature = random.choice(weather_data["features"])
        description = f"{description}, {weather} with {weather_feature}"

    # Add season if specified
    if season and season in SEASONS:
        season_data = SEASONS[season]
        season_feature = random.choice(season_data["features"])
        season_atmosphere = random.choice(season_data["atmosphere"])
        description = (
            f"{description}, {season} with {season_feature} and a {season_atmosphere} atmosphere"
        )

    # Add mood/atmosphere if specified
    if mood and mood in MOOD_DESCRIPTORS:
        mood_descriptor = random.choice(MOOD_DESCRIPTORS[mood])
        description = f"{description}, creating a {mood_descriptor} atmosphere"

    # Add location features
    features_text = " and ".join(location_features)
    description = f"{description}, featuring {features_text}"

    return description


def generate_scene_description(
    location: str,
    time_of_day: Optional[str] = None,
    weather: Optional[str] = None,
    characters: Optional[List[str]] = None,
    activity: Optional[str] = None,
) -> str:
    """Generate a scene description with characters and activities

    Args:
        location: The main location (forest, city, beach, etc.)
        time_of_day: Time of day (morning, noon, evening, night)
        weather: Weather condition (sunny, rainy, cloudy, etc.)
        characters: List of character descriptions
        activity: The main activity happening in the scene

    Returns:
        A detailed scene description
    """
    # Generate the basic background
    background = generate_background_description(location, time_of_day, weather)

    # Start with the background
    description = background

    # Add characters if specified
    if characters:
        if len(characters) == 1:
            description = f"{description} with {characters[0]}"
        else:
            char_list = ", ".join(characters[:-1]) + f" and {characters[-1]}"
            description = f"{description} with {char_list}"

    # Add activity if specified
    if activity:
        description = f"{description}, {activity}"

    return description


def get_complementary_locations(species: str) -> List[str]:
    """Get locations that complement a given species

    Args:
        species: The species to find complementary locations for

    Returns:
        A list of suitable locations
    """
    # Normalize species to lowercase
    species = species.lower()

    # Get habitats for the species
    habitats = SPECIES_HABITATS.get(species, [])

    # If no specific habitats found, provide some general options
    if not habitats:
        return ["forest", "meadow", "city", "beach", "mountains"]

    # Convert habitat descriptions to location keys from BACKGROUND_SETTINGS
    # when possible for consistency
    valid_locations = []
    for habitat in habitats:
        # Check if this habitat directly matches a background setting
        if habitat in BACKGROUND_SETTINGS:
            valid_locations.append(habitat)
        # Otherwise, try to find a close match
        else:
            for bg_key in BACKGROUND_SETTINGS.keys():
                if bg_key in habitat or habitat in bg_key:
                    valid_locations.append(bg_key)
                    break
            else:
                # No match found, just add the habitat as-is
                valid_locations.append(habitat)

    return valid_locations


def enhance_prompt_with_background(
    prompt: str, location: str, time_of_day: Optional[str] = None, weather: Optional[str] = None
) -> str:
    """Enhance a prompt with background details

    Args:
        prompt: Original prompt text
        location: The location setting
        time_of_day: Time of day
        weather: Weather condition

    Returns:
        Enhanced prompt with background details
    """
    # Generate a simple background phrase
    background_phrases = []

    # Add location
    location = location.lower()
    if location in BACKGROUND_SETTINGS:
        location_data = BACKGROUND_SETTINGS[location]
        descriptor = random.choice(location_data["descriptors"])
        article = get_indefinite_article(descriptor)
        background_phrases.append(f"in {article} {descriptor} {location}")
    else:
        article = get_indefinite_article(location)
        background_phrases.append(f"in {article} {location}")

    # Add time of day if specified
    if time_of_day and time_of_day.lower() in TIME_OF_DAY:
        time_data = TIME_OF_DAY[time_of_day.lower()]
        lighting = random.choice(time_data["lighting"])
        background_phrases.append(f"during {time_of_day} with {lighting}")

    # Add weather if specified
    if weather and weather.lower() in WEATHER_CONDITIONS:
        weather_data = WEATHER_CONDITIONS[weather.lower()]
        atmosphere = random.choice(weather_data["atmosphere"])
        article = get_indefinite_article(atmosphere)
        background_phrases.append(f"in {article} {atmosphere} {weather} conditions")

    # Combine background phrases
    if len(background_phrases) > 1:
        background = ", ".join(background_phrases)
    else:
        background = background_phrases[0]

    # Add the background to the prompt
    enhanced_prompt = f"{prompt}, {background}"

    return enhanced_prompt
