"""
Backgrounds command for cringegen CLI.

This module provides a command to generate background settings for furry scenes.
"""

import argparse
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Dictionary of backgrounds by category and time of day
BACKGROUNDS = {
    "natural": {
        "day": [
            "forest clearing",
            "lush meadow",
            "mountain vista",
            "flowing river",
            "sandy beach",
            "grassy plain",
            "flower field",
            "waterfall",
            "rocky cliffs",
            "desert oasis",
            "jungle canopy",
            "bamboo grove",
            "savanna grassland",
            "hilltop view",
            "pine forest",
        ],
        "night": [
            "starry sky forest",
            "moonlit meadow",
            "night mountain vista",
            "glowing fireflies in woods",
            "beach under moonlight",
            "night sky over plains",
            "forest with aurora",
            "moonlit waterfall",
            "starlit cliffs",
            "desert under starlight",
            "jungle at twilight",
            "bamboo grove with lanterns",
            "savanna under moonlight",
        ],
    },
    "urban": {
        "day": [
            "busy city street",
            "skyline view",
            "cafe terrace",
            "town square",
            "public park",
            "shopping district",
            "university campus",
            "subway station",
            "rooftop garden",
            "office building",
            "city fountain",
            "bus stop",
            "outdoor market",
            "botanical garden",
            "city bridge",
        ],
        "night": [
            "neon-lit street",
            "city skyline at night",
            "night cafe with lights",
            "town square with lanterns",
            "park with light trails",
            "nightlife district",
            "campus at night",
            "subway entrance at night",
            "rooftop with city lights",
            "office building at night",
            "illuminated fountain",
            "night market",
            "bridge with city lights",
        ],
    },
    "fantasy": {
        "day": [
            "enchanted forest",
            "elven city",
            "wizard tower",
            "ancient ruins",
            "druid grove",
            "floating islands",
            "crystal cave",
            "magic academy",
            "mythical mountain",
            "fairy hideaway",
            "golden temple",
            "magical oasis",
            "ancient tree village",
            "portal gateway",
            "phoenix nesting grounds",
        ],
        "night": [
            "enchanted forest with glowing plants",
            "elven city with magical lights",
            "wizard tower under twin moons",
            "ruins with spectral glow",
            "druid grove with luminous fungi",
            "floating islands under starlight",
            "crystal cave with bioluminescence",
            "magic academy with arcane lights",
            "mythical mountain with dragon fire",
            "fairy hideaway with glowing dust",
            "temple with moonlight ritual",
            "magical portal with energy streams",
        ],
    },
    "futuristic": {
        "day": [
            "futuristic cityscape",
            "space station interior",
            "high-tech laboratory",
            "cybernetic garden",
            "flying vehicles highway",
            "artificial habitat",
            "hologram plaza",
            "android assembly plant",
            "terraformed landscape",
            "solar power farm",
            "vertical city",
            "mag-lev train station",
            "space elevator",
            "underwater dome city",
        ],
        "night": [
            "neon cityscape",
            "space station with Earth view",
            "laboratory with glowing tech",
            "cybernetic garden with LED plants",
            "night sky with flying vehicles",
            "artificial habitat with night cycle",
            "hologram night club",
            "android repair shop",
            "terraformed landscape with aurora",
            "illuminated power grid",
            "vertical city with light trails",
            "space port with landing ships",
        ],
    },
    "indoor": {
        "day": [
            "cozy bedroom",
            "rustic cabin interior",
            "modern living room",
            "artist studio",
            "library with tall shelves",
            "gym with equipment",
            "kitchen with sunlight",
            "greenhouse conservatory",
            "workshop with tools",
            "classroom setting",
            "museum gallery",
            "indoor garden",
            "aquarium with fish tanks",
            "recording studio",
        ],
        "night": [
            "bedroom with soft lighting",
            "cabin with fireplace",
            "living room with city view",
            "studio with night lamp",
            "library with reading lights",
            "empty gym at night",
            "kitchen with mood lighting",
            "greenhouse under moonlight",
            "workshop with single lamp",
            "classroom at night",
            "museum after hours",
            "indoor garden with string lights",
            "aquarium with blue lighting",
        ],
    },
}

# Weather conditions
WEATHER = {
    "clear": [
        "clear sky",
        "sunny",
        "blue sky",
        "cloudless",
        "bright sunshine",
        "perfect weather",
    ],
    "cloudy": [
        "partly cloudy",
        "scattered clouds",
        "overcast",
        "cloudy sky",
        "grey clouds",
        "fluffy clouds",
    ],
    "rainy": [
        "light rain",
        "drizzle",
        "pouring rain",
        "rainstorm",
        "rain shower",
        "scattered showers",
    ],
    "stormy": [
        "thunderstorm",
        "lightning storm",
        "heavy wind",
        "gathering storm",
        "dark storm clouds",
        "thunder and lightning",
    ],
    "snowy": [
        "light snowfall",
        "snow flurries",
        "heavy snow",
        "blizzard",
        "snowdrifts",
        "gentle snowfall",
    ],
    "foggy": [
        "misty",
        "foggy",
        "hazy",
        "dense fog",
        "morning mist",
        "rolling fog",
    ],
    "special": [
        "rainbow",
        "northern lights",
        "meteor shower",
        "sunset",
        "sunrise",
        "double rainbow",
        "golden hour",
        "blue hour",
    ],
}

# Seasonal modifiers
SEASONS = {
    "spring": [
        "cherry blossoms",
        "blooming flowers",
        "new leaves",
        "spring breeze",
        "nesting birds",
    ],
    "summer": [
        "lush greenery",
        "summer heat",
        "vibrant colors",
        "long daylight",
        "buzzing insects",
    ],
    "autumn": [
        "fallen leaves",
        "autumn colors",
        "harvest time",
        "gold and red foliage",
        "misty mornings",
    ],
    "winter": [
        "snow-covered",
        "frozen",
        "winter chill",
        "bare trees",
        "frost patterns",
        "icicles",
    ],
}


def add_backgrounds_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the backgrounds command to the CLI."""
    parser = subparsers.add_parser(
        "backgrounds",
        parents=[parent_parser],
        help="Generate background settings for furry scenes",
        description="Generate background settings appropriate for furry artwork",
    )

    # Background category argument
    parser.add_argument(
        "--category",
        type=str,
        choices=["natural", "urban", "fantasy", "futuristic", "indoor", "random"],
        default="random",
        help="Category of background to generate (default: random)",
    )

    # Time of day argument
    parser.add_argument(
        "--time",
        type=str,
        choices=["day", "night", "random"],
        default="random",
        help="Time of day for the background (default: random)",
    )

    # Weather argument
    parser.add_argument(
        "--weather",
        type=str,
        choices=["clear", "cloudy", "rainy", "stormy", "snowy", "foggy", "special", "random", "none"],
        default="random",
        help="Weather condition for the background (default: random)",
    )

    # Season argument
    parser.add_argument(
        "--season",
        type=str,
        choices=["spring", "summer", "autumn", "winter", "random", "none"],
        default="none",
        help="Season for the background (default: none)",
    )

    # Count argument
    parser.add_argument(
        "--count", 
        type=int, 
        default=1, 
        help="Number of backgrounds to generate (default: 1)"
    )

    # Detailed flag
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate more detailed background descriptions",
    )

    # Format options
    parser.add_argument(
        "--format",
        type=str,
        choices=["simple", "detailed", "prompt", "csv"],
        default="simple",
        help="Output format (simple, detailed, prompt-ready, or csv)",
    )

    # Set the function to call when this command is invoked
    parser.set_defaults(func=handle_backgrounds_command)


def handle_backgrounds_command(args: Any) -> None:
    """Handle the backgrounds command."""
    # Generate backgrounds
    backgrounds = []
    for _ in range(args.count):
        # Determine category
        category = args.category
        if category == "random":
            category = random.choice(list(BACKGROUNDS.keys()))
        
        # Determine time of day
        time_of_day = args.time
        if time_of_day == "random":
            time_of_day = random.choice(["day", "night"])
        
        # Determine weather
        weather = args.weather
        if weather == "random":
            weather = random.choice(list(WEATHER.keys()))
        
        # Determine season
        season = args.season
        if season == "random":
            season = random.choice(list(SEASONS.keys()))
        
        # Generate the background
        if args.detailed:
            background = generate_detailed_background(
                category=category,
                time_of_day=time_of_day,
                weather=weather if weather != "none" else None,
                season=season if season != "none" else None,
            )
        else:
            background = generate_simple_background(
                category=category,
                time_of_day=time_of_day,
                weather=weather if weather != "none" else None,
                season=season if season != "none" else None,
            )
        
        backgrounds.append(background)
    
    # Output the backgrounds
    output_backgrounds(backgrounds, args.format, args.count)


def generate_simple_background(
    category: str,
    time_of_day: str,
    weather: Optional[str] = None,
    season: Optional[str] = None,
) -> Dict[str, str]:
    """Generate a simple background setting.

    Args:
        category: Category of background (natural, urban, etc.)
        time_of_day: Time of day (day, night)
        weather: Optional weather condition
        season: Optional season

    Returns:
        A dictionary with background details
    """
    # Get base backgrounds for the category and time
    if category in BACKGROUNDS and time_of_day in BACKGROUNDS[category]:
        base_backgrounds = BACKGROUNDS[category][time_of_day]
    else:
        # Fallback to natural day if category/time not found
        logger.warning(f"Background category '{category}' or time '{time_of_day}' not found. Using defaults.")
        base_backgrounds = BACKGROUNDS["natural"]["day"]
    
    # Select random base background
    base = random.choice(base_backgrounds)
    
    # Format the background data
    background = {
        "base": base,
        "category": category,
        "time": time_of_day,
        "weather": None,
        "season": None,
    }
    
    # Add weather if specified
    if weather and weather in WEATHER:
        background["weather"] = random.choice(WEATHER[weather])
    
    # Add season if specified
    if season and season in SEASONS:
        background["season"] = random.choice(SEASONS[season])
    
    return background


def generate_detailed_background(
    category: str,
    time_of_day: str,
    weather: Optional[str] = None,
    season: Optional[str] = None,
) -> Dict[str, str]:
    """Generate a detailed background setting with added elements.

    Args:
        category: Category of background (natural, urban, etc.)
        time_of_day: Time of day (day, night)
        weather: Optional weather condition
        season: Optional season

    Returns:
        A dictionary with detailed background details
    """
    # Start with a simple background
    background = generate_simple_background(category, time_of_day, weather, season)
    
    # Add additional environmental details based on category
    details = []
    
    if category == "natural":
        natural_details = [
            "with wildflowers dotting the landscape",
            "with a gentle breeze rustling the leaves",
            "with birds singing in the trees",
            "with the sound of running water nearby",
            "with rock formations in the distance",
            "with butterflies fluttering about",
            "with towering trees providing shade",
            "with mossy stones along a path",
            "with the scent of pine in the air",
            "with animal tracks in the soft ground",
        ]
        details.append(random.choice(natural_details))
        
    elif category == "urban":
        urban_details = [
            "with people going about their day",
            "with the sound of distant traffic",
            "with street vendors selling food",
            "with music playing from a nearby shop",
            "with colorful storefronts lining the street",
            "with outdoor seating areas",
            "with bicycles passing by",
            "with signs and advertisements",
            "with the smell of coffee in the air",
            "with a mix of old and new architecture",
        ]
        details.append(random.choice(urban_details))
        
    elif category == "fantasy":
        fantasy_details = [
            "with magical creatures in the background",
            "with floating crystals casting light",
            "with strange plants of every color",
            "with ancient runes carved into stone",
            "with wisps of magical energy flowing through the air",
            "with impossibly tall structures",
            "with portals shimmering in the distance",
            "with magical artifacts on display",
            "with mystical symbols glowing faintly",
            "with exotic flora unlike any on Earth",
        ]
        details.append(random.choice(fantasy_details))
        
    elif category == "futuristic":
        futuristic_details = [
            "with holographic displays projecting information",
            "with robots going about their tasks",
            "with clean lines and minimalist architecture",
            "with transparent screens showing data",
            "with energy fields pulsing faintly",
            "with antigravity platforms floating in place",
            "with automated systems at work",
            "with augmented reality overlays visible",
            "with energy conduits running through structures",
            "with sleek, reflective surfaces everywhere",
        ]
        details.append(random.choice(futuristic_details))
        
    elif category == "indoor":
        indoor_details = [
            "with comfortable furniture arranged thoughtfully",
            "with personal belongings adding character",
            "with art hanging on the walls",
            "with plants in decorative pots",
            "with the scent of food or incense",
            "with warm lighting creating ambiance",
            "with books and papers neatly organized",
            "with rugs covering parts of the floor",
            "with windows showing a view outside",
            "with small details that make it feel lived-in",
        ]
        details.append(random.choice(indoor_details))
    
    # Add time-specific details
    if time_of_day == "day":
        time_details = [
            "with sunlight filtering through",
            "with clear visibility in all directions",
            "with shadows cast by the sun",
            "with a bright, vibrant atmosphere",
            "bathed in natural light",
        ]
        details.append(random.choice(time_details))
    else:  # night
        time_details = [
            "with deep shadows and areas of darkness",
            "with moonlight creating a silvery glow",
            "with the twinkling of stars visible",
            "with artificial lights providing illumination",
            "with a mysterious, atmospheric quality",
        ]
        details.append(random.choice(time_details))
    
    # Add the details to the background
    background["details"] = details
    
    return background


def output_backgrounds(backgrounds: List[Dict[str, str]], format_type: str, count: int) -> None:
    """Format and output the generated backgrounds.

    Args:
        backgrounds: List of background dictionaries to output
        format_type: The format to output (simple, detailed, prompt, csv)
        count: Number of backgrounds requested
    """
    if not backgrounds:
        print("No backgrounds generated.")
        return
    
    for i, bg in enumerate(backgrounds):
        if count > 1 and format_type != "csv":
            print(f"\nBackground {i+1}:")
        
        if format_type == "simple":
            # Simple output includes just the base, weather, and season
            desc = bg["base"]
            
            if bg["weather"]:
                desc += f", {bg['weather']}"
            
            if bg["season"]:
                desc += f", {bg['season']}"
            
            print(desc)
            
        elif format_type == "detailed":
            # Detailed output includes all components
            print(f"Base setting: {bg['base']}")
            print(f"Category: {bg['category']}")
            print(f"Time of day: {bg['time']}")
            
            if bg["weather"]:
                print(f"Weather: {bg['weather']}")
            
            if bg["season"]:
                print(f"Season: {bg['season']}")
            
            if "details" in bg:
                print("Details:")
                for detail in bg["details"]:
                    print(f"  - {detail}")
        
        elif format_type == "csv":
            # Create a list of all background elements
            bg_parts = [bg["base"]]
            
            if bg["weather"]:
                bg_parts.append(bg["weather"])
            
            if bg["season"]:
                bg_parts.append(bg["season"])
            
            if "details" in bg:
                # Strip the "with " prefix from details if it exists
                for detail in bg["details"]:
                    if detail.startswith("with "):
                        bg_parts.append(detail[5:])
                    else:
                        bg_parts.append(detail)
            
            # Output as comma-separated list with no other text
            print(",".join(bg_parts))
            
        elif format_type == "prompt":
            # Prompt-ready format for direct inclusion in prompts
            prompt_parts = []
            
            # Add base, weather and season
            desc = bg["base"]
            
            if bg["weather"]:
                desc += f", {bg['weather']}"
            
            if bg["season"]:
                desc += f", {bg['season']}"
            
            prompt_parts.append(desc)
            
            # Add details if present
            if "details" in bg:
                prompt_parts.extend(bg["details"])
            
            # Combine all parts
            prompt = ", ".join(prompt_parts)
            print(f"background of {prompt}") 