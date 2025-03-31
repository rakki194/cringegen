"""
Style Utility Functions

This module provides utility functions for working with the style taxonomy system.
It ensures backward compatibility with existing code while enabling new
functionality from the hierarchical style taxonomy.
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Union
import random
from collections import defaultdict

from cringegen.data.style_taxonomy import (
    STYLE_MEDIUM,
    STYLE_MOVEMENTS,
    STYLE_GENRES,
    STYLE_ANIMATION,
    STYLE_AESTHETICS,
    STYLE_RELATIONSHIPS,
    STYLE_ATTRIBUTES,
    get_style_by_medium,
    get_style_by_movement,
    get_related_styles,
    get_style_attributes,
    get_styles_by_attribute,
)

from cringegen.data.styles import (
    ART_STYLES,
    FILM_TV_STYLES,
    GAME_ART_STYLES,
    VISUAL_AESTHETICS,
)

# =============================================================================
# Compatibility Functions
# =============================================================================

def get_all_style_names() -> List[str]:
    """
    Get a flat list of all style names from the hierarchical taxonomy.
    
    Returns:
        List[str]: A comprehensive list of all style names
    """
    all_styles = set()
    
    # Add style names from STYLE_MEDIUM
    for medium, medium_data in STYLE_MEDIUM.items():
        all_styles.add(medium_data["name"])
        if "techniques" in medium_data:
            for technique, technique_data in medium_data["techniques"].items():
                all_styles.add(technique_data["name"])
                if "subtypes" in technique_data:
                    all_styles.update(technique_data["subtypes"])
    
    # Add style names from STYLE_MOVEMENTS
    for movement, movement_data in STYLE_MOVEMENTS.items():
        all_styles.add(movement_data["name"])
        if "subtypes" in movement_data:
            all_styles.update(movement_data["subtypes"])
    
    # Add style names from STYLE_GENRES
    for genre, genre_data in STYLE_GENRES.items():
        all_styles.add(genre_data["name"])
        if "subtypes" in genre_data:
            all_styles.update(genre_data["subtypes"])
    
    # Add style names from STYLE_ANIMATION
    for anim_style, anim_data in STYLE_ANIMATION.items():
        all_styles.add(anim_data["name"])
        if "subtypes" in anim_data:
            all_styles.update(anim_data["subtypes"])
    
    # Add style names from STYLE_AESTHETICS
    for aesthetic, aesthetic_data in STYLE_AESTHETICS.items():
        all_styles.add(aesthetic_data["name"])
        if "subtypes" in aesthetic_data:
            all_styles.update(aesthetic_data["subtypes"])
    
    # Convert to list and sort
    return sorted(list(all_styles))

def is_style_in_taxonomy(style_name: str) -> bool:
    """
    Check if a style name exists in the hierarchical style taxonomy.
    
    Args:
        style_name (str): The style name to check
        
    Returns:
        bool: True if the style exists in the taxonomy, False otherwise
    """
    # Check in each taxonomy structure
    for medium, medium_data in STYLE_MEDIUM.items():
        if medium_data["name"] == style_name:
            return True
        if "techniques" in medium_data:
            for technique, technique_data in medium_data["techniques"].items():
                if technique_data["name"] == style_name:
                    return True
                if "subtypes" in technique_data and style_name in technique_data["subtypes"]:
                    return True
    
    for movement, movement_data in STYLE_MOVEMENTS.items():
        if movement_data["name"] == style_name:
            return True
        if "subtypes" in movement_data and style_name in movement_data["subtypes"]:
            return True
    
    for genre, genre_data in STYLE_GENRES.items():
        if genre_data["name"] == style_name:
            return True
        if "subtypes" in genre_data and style_name in genre_data["subtypes"]:
            return True
    
    for anim_style, anim_data in STYLE_ANIMATION.items():
        if anim_data["name"] == style_name:
            return True
        if "subtypes" in anim_data and style_name in anim_data["subtypes"]:
            return True
    
    for aesthetic, aesthetic_data in STYLE_AESTHETICS.items():
        if aesthetic_data["name"] == style_name:
            return True
        if "subtypes" in aesthetic_data and style_name in aesthetic_data["subtypes"]:
            return True
    
    return False

def map_legacy_style_to_taxonomy(legacy_style: str) -> str:
    """
    Map a style name from the legacy flat structures to the hierarchical taxonomy.
    
    Args:
        legacy_style (str): A style name from the legacy style data
        
    Returns:
        str: The corresponding name in the hierarchical taxonomy, or the original if not found
    """
    # Map of legacy style names to taxonomy style names
    style_mapping = {
        # Example mappings - would need to be expanded based on actual data
        "impressionist": "Impressionism",
        "cubist": "Cubism",
        "surreal": "Surrealism",
        "abstract": "Abstract Art",
        "pop art": "Pop Art",
        "digital painting": "Digital Painting",
        "pixel art": "Pixel Art",
        "watercolor": "Watercolor",
        "oil painting": "Oil Painting",
    }
    
    return style_mapping.get(legacy_style, legacy_style)

# =============================================================================
# Enhanced Style Selection Functions
# =============================================================================

def get_complementary_styles(style_name: str, count: int = 3) -> List[str]:
    """
    Get styles that complement the given style.
    
    Args:
        style_name (str): The base style to find complements for
        count (int): The number of complementary styles to return
        
    Returns:
        List[str]: A list of complementary style names
    """
    # Get related styles directly from relationship data if available
    complementary = []
    
    # Check if the style exists in complementary relationship data
    if style_name in STYLE_RELATIONSHIPS["complementary"]:
        complementary.extend(STYLE_RELATIONSHIPS["complementary"][style_name])
    
    # Also check if any style lists the given style as complementary
    for style, complements in STYLE_RELATIONSHIPS["complementary"].items():
        if style_name in complements:
            complementary.append(style)
    
    # If we don't have enough, add some from the same category
    for medium, medium_data in STYLE_MEDIUM.items():
        if medium_data["name"] == style_name and "techniques" in medium_data:
            # Add techniques from the same medium
            for technique, technique_data in medium_data["techniques"].items():
                complementary.append(technique_data["name"])
    
    # If we still don't have enough, add related by attributes
    if len(complementary) < count:
        attributes = get_style_attributes(style_name)
        for attr in attributes:
            styles = get_styles_by_attribute(attr)
            complementary.extend([s for s in styles if s != style_name])
    
    # Remove duplicates, randomize, and return requested count
    complementary = list(set(complementary))
    random.shuffle(complementary)
    return complementary[:count]

def get_style_by_mood(mood: str, count: int = 3) -> List[str]:
    """
    Get styles that evoke a particular mood or emotional tone.
    
    Args:
        mood (str): The mood to find styles for (e.g., "serene", "dramatic", "whimsical")
        count (int): The number of styles to return
        
    Returns:
        List[str]: A list of style names that evoke the requested mood
    """
    matching_styles = []
    
    # Check in emotional tones attributes
    if mood in STYLE_ATTRIBUTES["emotional_tones"]:
        matching_styles.extend(STYLE_ATTRIBUTES["emotional_tones"][mood]["examples"])
    
    # If we don't have enough matches, look for related moods
    related_moods = {
        "serene": ["calm", "peaceful", "tranquil", "gentle"],
        "dramatic": ["intense", "bold", "powerful", "dynamic"],
        "whimsical": ["playful", "quirky", "fanciful", "lighthearted"],
        "melancholic": ["sad", "somber", "gloomy", "reflective"],
        "energetic": ["vibrant", "lively", "dynamic", "active"]
    }
    
    if len(matching_styles) < count and mood in related_moods:
        for related_mood in related_moods[mood]:
            if related_mood in STYLE_ATTRIBUTES["emotional_tones"]:
                matching_styles.extend(STYLE_ATTRIBUTES["emotional_tones"][related_mood]["examples"])
    
    # Remove duplicates, randomize, and return requested count
    matching_styles = list(set(matching_styles))
    random.shuffle(matching_styles)
    return matching_styles[:count]

def get_random_style_combination(count: int = 3) -> List[str]:
    """
    Generate a random but coherent combination of complementary styles.
    
    Args:
        count (int): The number of styles to combine
        
    Returns:
        List[str]: A list of style names that work well together
    """
    style_categories = [
        list(STYLE_MOVEMENTS.values()),
        list(STYLE_GENRES.values()),
        list(STYLE_MEDIUM.values()),
        list(STYLE_AESTHETICS.values())
    ]
    
    # Randomly select a primary style category and style
    category = random.choice(style_categories)
    primary_style_data = random.choice(category)
    primary_style = primary_style_data["name"]
    
    # Get complementary styles to the primary style
    style_combination = [primary_style]
    complementary_styles = get_complementary_styles(primary_style, count=count-1)
    style_combination.extend(complementary_styles)
    
    return style_combination

def get_style_characteristics(style_name: str) -> Dict[str, List[str]]:
    """
    Get the defining characteristics and attributes of a style.
    
    Args:
        style_name (str): The style to analyze
        
    Returns:
        Dict[str, List[str]]: A dictionary of characteristic categories and their values
    """
    characteristics = defaultdict(list)
    
    # Try to find the style in each taxonomy structure
    for medium, medium_data in STYLE_MEDIUM.items():
        if medium_data["name"] == style_name:
            if "description" in medium_data:
                characteristics["description"].append(medium_data["description"])
        
        if "techniques" in medium_data:
            for technique, technique_data in medium_data["techniques"].items():
                if technique_data["name"] == style_name:
                    if "description" in technique_data:
                        characteristics["description"].append(technique_data["description"])
                    if "materials" in technique_data:
                        characteristics["materials"] = technique_data["materials"]
                    if "tools" in technique_data:
                        characteristics["tools"] = technique_data["tools"]
                    if "surfaces" in technique_data:
                        characteristics["surfaces"] = technique_data["surfaces"]
    
    for movement, movement_data in STYLE_MOVEMENTS.items():
        if movement_data["name"] == style_name:
            if "period" in movement_data:
                characteristics["period"].append(movement_data["period"])
            if "characteristics" in movement_data:
                characteristics["visual_traits"] = movement_data["characteristics"]
            if "notable_artists" in movement_data:
                characteristics["notable_artists"] = movement_data["notable_artists"]
    
    # Find attributes associated with the style
    for category, attr_dict in STYLE_ATTRIBUTES.items():
        for attr_name, attr_info in attr_dict.items():
            if style_name in attr_info["examples"]:
                characteristics[category].append(attr_name)
    
    # Convert defaultdict to regular dict
    return dict(characteristics)

# =============================================================================
# Prompt Generation Helper Functions  
# =============================================================================

def generate_style_prompt_segment(styles: List[str], include_artists: bool = True) -> str:
    """
    Generate a prompt segment describing art styles for image generation.
    
    Args:
        styles (List[str]): List of style names to include
        include_artists (bool): Whether to include notable artists
        
    Returns:
        str: A formatted prompt segment for style description
    """
    prompt_parts = []
    
    # Add each style with additional context
    for style in styles:
        # Get characteristics to enrich the prompt
        characteristics = get_style_characteristics(style)
        
        style_part = style
        
        # Add visual characteristics if available
        if "visual_traits" in characteristics and characteristics["visual_traits"]:
            traits = random.sample(characteristics["visual_traits"], 
                                  min(2, len(characteristics["visual_traits"])))
            style_part += f", {', '.join(traits)}"
        
        # Add artist reference if requested and available
        if include_artists and "notable_artists" in characteristics and characteristics["notable_artists"]:
            artist = random.choice(characteristics["notable_artists"])
            style_part += f", in the style of {artist}"
            
        prompt_parts.append(style_part)
    
    # Join all style parts
    if len(prompt_parts) > 1:
        prompt = f"{', '.join(prompt_parts[:-1])} and {prompt_parts[-1]} style"
    else:
        prompt = f"{prompt_parts[0]} style"
    
    return prompt

def generate_medium_specification(medium: str) -> str:
    """
    Generate a specific medium description for use in prompts.
    
    Args:
        medium (str): The medium to describe
        
    Returns:
        str: A formatted medium specification
    """
    medium_info = {}
    
    # Find the medium in the taxonomy
    for medium_key, medium_data in STYLE_MEDIUM.items():
        if medium_data["name"].lower() == medium.lower():
            medium_info = medium_data
            break
        
        # Check in techniques
        if "techniques" in medium_data:
            for technique, technique_data in medium_data["techniques"].items():
                if technique_data["name"].lower() == medium.lower():
                    medium_info = technique_data
                    break
    
    if not medium_info:
        return medium
    
    # Generate a more detailed description
    details = []
    
    # Add random material if available
    if "materials" in medium_info and medium_info["materials"]:
        details.append(random.choice(medium_info["materials"]))
    
    # Add surface if available
    if "surfaces" in medium_info and medium_info["surfaces"]:
        details.append(f"on {random.choice(medium_info['surfaces'])}")
    
    # Format the result
    if details:
        return f"{medium_info['name']}, {' '.join(details)}"
    else:
        return medium_info['name']

# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

def get_compatible_art_styles() -> Dict[str, List[str]]:
    """
    Return a combined dictionary of art styles that maintains compatibility
    with the legacy flat structure while incorporating the new taxonomy.
    
    Returns:
        Dict[str, List[str]]: A dictionary of style categories and their values
    """
    # Start with the original ART_STYLES structure
    compatible_styles = {category: styles[:] for category, styles in ART_STYLES.items()}
    
    # Add new categories from STYLE_MOVEMENTS if not already present
    movement_styles = {}
    for movement, movement_data in STYLE_MOVEMENTS.items():
        category = movement_data.get("period", "Other Movements")
        if category not in movement_styles:
            movement_styles[category] = []
        movement_styles[category].append(movement_data["name"])
    
    # Merge with the existing structure
    for category, styles in movement_styles.items():
        if category in compatible_styles:
            # Add only styles not already in the list
            for style in styles:
                if style not in compatible_styles[category]:
                    compatible_styles[category].append(style)
        else:
            compatible_styles[category] = styles
    
    return compatible_styles

def get_style_suggestions(query: str, count: int = 5) -> List[str]:
    """
    Get style suggestions based on a text query.
    
    Args:
        query (str): The query to find matching styles for (e.g., "futuristic", "medieval", "colorful")
        count (int): The number of style suggestions to return
        
    Returns:
        List[str]: A list of suggested style names that match the query
    """
    query = query.lower()
    matching_styles = []
    
    # First, check if query directly matches a style name
    all_styles = get_all_style_names()
    direct_matches = [style for style in all_styles if query in style.lower()]
    matching_styles.extend(direct_matches)
    
    # Check if query matches a visual aesthetic category
    for category, styles in VISUAL_AESTHETICS.items():
        if query in category.lower():
            matching_styles.extend(styles)
    
    # Check if query matches a mood or attribute
    for category, attr_dict in STYLE_ATTRIBUTES.items():
        for attr_name, attr_info in attr_dict.items():
            if query in attr_name.lower():
                matching_styles.extend(attr_info.get("examples", []))
    
    # Check for related terms in movements
    for movement, movement_data in STYLE_MOVEMENTS.items():
        # Check in description
        description = movement_data.get("description", "").lower()
        if query in description:
            matching_styles.append(movement_data["name"])
        
        # Check in characteristics
        characteristics = movement_data.get("characteristics", [])
        if any(query in trait.lower() for trait in characteristics):
            matching_styles.append(movement_data["name"])
    
    # Check for related terms in mediums
    for medium, medium_data in STYLE_MEDIUM.items():
        # Check in description
        description = medium_data.get("description", "").lower()
        if query in description:
            matching_styles.append(medium_data["name"])
    
    # If we don't have enough matches, add some random recommendations
    if len(matching_styles) < count:
        # Add some random popular styles
        popular_styles = [
            "Digital Art", "Oil Painting", "Watercolor", "Impressionism",
            "Anime", "Cyberpunk", "Photorealistic", "Pop Art", "Abstract",
            "Fantasy", "Surrealism", "Minimalist", "Comic Book", "Concept Art"
        ]
        remaining_needed = count - len(matching_styles)
        random.shuffle(popular_styles)
        matching_styles.extend(popular_styles[:remaining_needed])
    
    # Remove duplicates, shuffle, and return requested count
    matching_styles = list(set(matching_styles))
    random.shuffle(matching_styles)
    return matching_styles[:count]

def generate_style_prompt(styles: List[str], include_medium: bool = True, include_artists: bool = True) -> str:
    """
    Generate a complete style prompt for image generation based on a list of styles.
    
    Args:
        styles (List[str]): List of style names to include
        include_medium (bool): Whether to include medium specification
        include_artists (bool): Whether to include notable artists
        
    Returns:
        str: A formatted prompt segment for style description
    """
    if not styles:
        return ""
    
    # Generate the style prompt segment
    style_prompt = generate_style_prompt_segment(styles, include_artists)
    
    # Add medium specification if requested
    if include_medium and styles:
        # Try to extract a medium from the first style
        primary_style = styles[0]
        for medium_key, medium_data in STYLE_MEDIUM.items():
            if medium_data["name"] == primary_style:
                medium_spec = generate_medium_specification(primary_style)
                style_prompt = f"{style_prompt}, {medium_spec}"
                break
    
    return style_prompt

# Alias for get_style_by_mood to ensure backward compatibility
find_style_by_mood = get_style_by_mood 