"""
Unified Environment System for CringeGen

This module provides a centralized environment system that consolidates all environment,
background, time, weather, and season data for the CringeGen toolkit. It uses consistent
data structures and formats for better maintainability and extensibility.

Key components:
- Time of day with lighting and atmosphere
- Weather conditions with features and atmosphere 
- Seasons with characteristics
- Background settings with descriptors and features
- Mood descriptors for environments
"""

from typing import Dict, List, Set, Union, Any

# =============================================================================
# Time of Day
# =============================================================================

TIME_OF_DAY: Dict[str, Dict[str, Any]] = {
    "dawn": {
        "name": ["dawn", "early morning", "sunrise", "first light", "daybreak", "crack of dawn"],
        "lighting": [
            "soft golden light",
            "gentle rays of sunlight",
            "warm orange glow",
            "first light breaking over the horizon",
        ],
        "atmosphere": ["peaceful", "quiet", "hopeful", "new beginning"],
        "descriptors": ["early", "fresh", "crisp", "awakening", "nascent"]
    },
    "morning": {
        "name": ["morning", "mid-morning", "bright morning", "clear morning", "fresh morning", "dewy morning"],
        "lighting": [
            "bright morning light",
            "clear sunlight",
            "warm golden rays",
            "soft morning glow",
        ],
        "atmosphere": ["fresh", "energetic", "lively", "promising"],
        "descriptors": ["vibrant", "busy", "alert", "active", "productive"]
    },
    "noon": {
        "name": ["noon", "midday", "high noon", "lunchtime", "zenith", "sun at its peak"],
        "lighting": [
            "harsh overhead sunlight",
            "bright midday light",
            "strong direct sunlight",
            "blinding brightness",
        ],
        "atmosphere": ["busy", "active", "vibrant", "bustling"],
        "descriptors": ["peak", "intense", "direct", "overhead", "strong"]
    },
    "afternoon": {
        "name": ["afternoon", "late afternoon", "mid-afternoon", "golden afternoon", "warm afternoon"],
        "lighting": [
            "warm afternoon sunlight",
            "golden hour glow",
            "soft yellow light",
            "long shadows",
        ],
        "atmosphere": ["relaxed", "lazy", "pleasant", "warm"],
        "descriptors": ["comfortable", "mellow", "laid-back", "easygoing"]
    },
    "sunset": {
        "name": ["sunset", "dusk", "twilight", "sundown", "evening glow", "golden hour"],
        "lighting": [
            "vibrant red and orange hues",
            "rich golden light",
            "dramatic sky colors",
            "long purple shadows",
        ],
        "atmosphere": ["romantic", "reflective", "calming", "serene"],
        "descriptors": ["magical", "transitional", "striking", "colorful", "dramatic"]
    },
    "evening": {
        "name": ["evening", "early night", "nightfall", "gloaming"],
        "lighting": [
            "soft evening light",
            "gentle artificial lighting",
            "warm indoor lights",
            "early stars appearing",
        ],
        "atmosphere": ["cozy", "relaxed", "intimate", "social"],
        "descriptors": ["winding down", "calming", "comfortable", "quieting"]
    },
    "night": {
        "name": ["night", "starry night", "moonlit night", "dark night"],
        "lighting": [
            "silver moonlight",
            "dark shadows",
            "starlight",
            "deep blue darkness",
        ],
        "atmosphere": ["mysterious", "quiet", "secretive", "peaceful"],
        "descriptors": ["shadowy", "nocturnal", "hushed", "still", "dark"]
    },
    "midnight": {
        "name": ["midnight", "dead of night", "small hours", "witching hour"],
        "lighting": [
            "deep darkness",
            "eerie moonlight",
            "black shadows",
            "faint starlight",
        ],
        "atmosphere": ["secretive", "forbidden", "mysterious", "supernatural"],
        "descriptors": ["deepest", "darkest", "silent", "otherworldly"]
    },
}

# =============================================================================
# Weather Conditions
# =============================================================================

WEATHER_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "clear": {
        "name": ["clear", "clear skies", "cloudless", "pristine skies", "perfect weather", "fair weather"],
        "features": [
            "blue sky",
            "bright sunshine",
            "not a cloud in sight",
            "perfect visibility",
        ],
        "atmosphere": ["cheerful", "warm", "energetic", "vibrant"],
        "descriptors": ["perfect", "ideal", "unblemished", "crystal clear"]
    },
    "sunny": {
        "name": ["sunny", "bright", "sun-drenched", "sunlit", "sun-filled"],
        "features": [
            "bright sunshine",
            "warm rays of light",
            "golden sunbeams",
            "sun-dappled surfaces",
        ],
        "atmosphere": ["cheerful", "warm", "energetic", "vibrant"],
        "descriptors": ["bright", "warm", "glowing", "radiant"]
    },
    "partly_cloudy": {
        "name": ["partly cloudy", "scattered clouds", "partly sunny", "broken clouds", "passing clouds"],
        "features": [
            "scattered clouds",
            "patches of blue sky",
            "shifting sunlight",
            "dynamic sky",
        ],
        "atmosphere": ["pleasant", "mild", "changing", "variable"],
        "descriptors": ["variable", "changeable", "mixed", "intermediate"]
    },
    "cloudy": {
        "name": ["cloudy", "overcast", "gray skies", "gloomy", "dark clouds"],
        "features": [
            "overcast skies",
            "diffused light",
            "soft gray clouds",
            "muted sunlight",
        ],
        "atmosphere": ["subdued", "calm", "thoughtful", "introspective"],
        "descriptors": ["muted", "dim", "dull", "blanketed"]
    },
    "rainy": {
        "name": ["rainy", "drizzling", "showery", "precipitation", "wet"],
        "features": [
            "falling raindrops",
            "puddles forming",
            "wet surfaces",
            "the sound of rainfall",
        ],
        "atmosphere": ["melancholic", "refreshing", "cleansing", "introspective"],
        "descriptors": ["damp", "soaked", "torrential", "dripping", "pouring"]
    },
    "stormy": {
        "name": ["stormy", "thunderstorm", "tempest", "electrical storm", "squall"],
        "features": [
            "dark thunderclouds",
            "flashes of lightning",
            "rumbling thunder",
            "heavy rainfall",
        ],
        "atmosphere": ["dramatic", "intense", "powerful", "ominous"],
        "descriptors": ["turbulent", "violent", "chaotic", "fierce", "wild"]
    },
    "foggy": {
        "name": ["foggy", "misty", "hazy", "fog-shrouded", "foggy landscape"],
        "features": [
            "thick mist",
            "limited visibility",
            "diffused light",
            "moisture in the air",
        ],
        "atmosphere": ["mysterious", "eerie", "secretive", "isolating"],
        "descriptors": ["veiled", "obscured", "shrouded", "opaque", "blurred"]
    },
    "snowy": {
        "name": ["snowy", "snowfall", "blizzard", "winter wonderland", "snow-covered"],
        "features": [
            "falling snowflakes",
            "white blanket covering",
            "crisp air",
            "pristine landscape",
        ],
        "atmosphere": ["serene", "magical", "quiet", "pure"],
        "descriptors": ["powdery", "frosty", "blanketed", "crystalline", "glistening"]
    },
    "windy": {
        "name": ["windy", "breezy", "gusty", "windswept", "blustery"],
        "features": [
            "swaying trees",
            "rustling leaves",
            "blowing debris",
            "dynamic movement",
        ],
        "atmosphere": ["energetic", "restless", "wild", "changing"],
        "descriptors": ["gusty", "howling", "whistling", "forceful", "vigorous"]
    },
    "hot": {
        "name": ["hot", "scorching", "sweltering", "blistering", "heat wave"],
        "features": [
            "heat waves visible",
            "beating sun",
            "dry conditions",
            "seeking shade",
        ],
        "atmosphere": ["oppressive", "languorous", "sluggish", "intense"],
        "descriptors": ["sweltering", "blazing", "burning", "searing", "tropical"]
    },
    "cold": {
        "name": ["cold", "frigid", "freezing", "icy", "bitter cold"],
        "features": [
            "visible breath",
            "frozen surfaces",
            "bundled figures",
            "crystalline air",
        ],
        "atmosphere": ["crisp", "sharp", "bracing", "invigorating"],
        "descriptors": ["biting", "chilly", "nippy", "wintry", "brisk"]
    },
}

# =============================================================================
# Seasons
# =============================================================================

SEASONS: Dict[str, Dict[str, Any]] = {
    "spring": {
        "name": ["spring", "springtime", "vernal season", "spring months"],
        "features": [
            "blooming flowers",
            "budding trees",
            "new growth",
            "nesting birds",
            "gentle rain",
        ],
        "atmosphere": ["hopeful", "renewing", "vibrant", "fresh"],
        "descriptors": ["blooming", "budding", "growing", "verdant", "rejuvenating"]
    },
    "summer": {
        "name": ["summer", "summertime", "estival season", "summer months"],
        "features": [
            "lush foliage",
            "strong sunlight",
            "warm temperatures",
            "blooming gardens",
            "long days",
        ],
        "atmosphere": ["vibrant", "energetic", "lively", "carefree"],
        "descriptors": ["hot", "sunny", "vibrant", "lush", "flourishing", "tropical"]
    },
    "autumn": {
        "name": ["autumn", "fall", "autumnal season", "fall months"],
        "features": [
            "colorful foliage",
            "falling leaves",
            "harvest time",
            "cooling temperatures",
            "shortening days",
        ],
        "atmosphere": ["contemplative", "transitional", "mellow", "nostalgic"],
        "descriptors": ["golden", "crisp", "colorful", "harvest", "amber", "rustic"]
    },
    "winter": {
        "name": ["winter", "wintertime", "hibernal season", "winter months"],
        "features": [
            "bare trees",
            "snow-covered landscapes",
            "frozen surfaces",
            "short days",
            "holiday decorations",
        ],
        "atmosphere": ["serene", "stark", "contemplative", "introspective"],
        "descriptors": ["snowy", "icy", "frosty", "frozen", "barren", "cozy"]
    },
}

# =============================================================================
# Mood Descriptors
# =============================================================================

MOOD_DESCRIPTORS: Dict[str, List[str]] = {
    "peaceful": [
        "tranquil",
        "serene",
        "calm",
        "quiet",
        "relaxing",
        "soothing",
        "undisturbed",
        "harmonious",
    ],
    "mysterious": [
        "enigmatic",
        "cryptic",
        "obscure",
        "puzzling",
        "strange",
        "peculiar",
        "unexplained",
        "secretive",
    ],
    "romantic": [
        "intimate",
        "loving",
        "passionate",
        "tender",
        "heartfelt",
        "affectionate",
        "dreamy",
        "idyllic",
    ],
    "melancholic": [
        "sad",
        "somber",
        "wistful",
        "nostalgic",
        "bittersweet",
        "pensive",
        "reflective",
        "longing",
    ],
    "cheerful": [
        "happy",
        "joyful",
        "lively",
        "bright",
        "upbeat",
        "playful",
        "gleeful",
        "jubilant",
    ],
    "dramatic": [
        "intense",
        "powerful",
        "striking",
        "emotional",
        "moving",
        "impactful",
        "bold",
        "theatrical",
    ],
    "ethereal": [
        "otherworldly",
        "magical",
        "dreamlike",
        "surreal",
        "enchanted",
        "mystical",
        "celestial",
        "unearthly",
    ],
    "ominous": [
        "threatening",
        "foreboding",
        "menacing",
        "sinister",
        "dark",
        "eerie",
        "portentous",
        "inauspicious",
    ],
}

# =============================================================================
# Background Settings
# =============================================================================

BACKGROUND_SETTINGS: Dict[str, Dict[str, Any]] = {
    # Natural environments
    "forest": {
        "category": "natural",
        "descriptors": [
            "dense",
            "lush",
            "ancient",
            "enchanted",
            "misty",
            "dark",
            "magical",
            "tropical",
            "pine",
        ],
        "features": [
            "trees",
            "foliage",
            "undergrowth",
            "forest floor",
            "branches",
            "leaves",
            "canopy",
            "moss",
        ],
        "lighting": [
            "dappled sunlight",
            "filtered light",
            "forest shade",
            "sunbeams",
            "misty glow",
        ],
        "modifiers": ["deep in the forest", "forest clearing", "forest path", "forest edge"],
        "atmosphere": ["peaceful", "mysterious", "enchanting", "secluded"]
    },
    "mountains": {
        "category": "natural",
        "descriptors": [
            "towering",
            "snow-capped",
            "jagged",
            "majestic",
            "rocky",
            "rugged",
            "steep",
            "foggy",
        ],
        "features": [
            "peaks",
            "cliffs",
            "valleys",
            "slopes",
            "rocks",
            "boulders",
            "caves",
            "waterfalls",
        ],
        "lighting": ["alpine glow", "mountain sunset", "clear mountain light", "foggy peaks"],
        "modifiers": ["mountain pass", "mountain peak", "mountain trail", "mountain valley"],
        "atmosphere": ["majestic", "imposing", "vast", "breathtaking"]
    },
    "beach": {
        "category": "natural",
        "descriptors": [
            "sandy",
            "tropical",
            "rocky",
            "pristine",
            "secluded",
            "white sand",
            "black sand",
        ],
        "features": [
            "sand",
            "waves",
            "shoreline",
            "palm trees",
            "seashells",
            "driftwood",
            "cliffs",
        ],
        "lighting": ["sunset glow", "golden hour", "bright sunshine", "moonlight reflecting"],
        "modifiers": ["seashore", "coastline", "seaside", "oceanfront", "tidal pools"],
        "atmosphere": ["relaxing", "peaceful", "open", "breezy"]
    },
    "river": {
        "category": "natural",
        "descriptors": [
            "flowing",
            "rushing",
            "calm",
            "clear",
            "winding",
            "wide",
            "shallow",
            "deep",
        ],
        "features": ["water", "current", "riverbank", "stones", "rapids", "pools", "wildlife"],
        "lighting": ["reflections on water", "sparkling water", "misty river morning"],
        "modifiers": ["riverside", "river crossing", "river bend", "riverbank"],
        "atmosphere": ["flowing", "changing", "refreshing", "life-giving"]
    },
    "lake": {
        "category": "natural",
        "descriptors": [
            "crystal clear",
            "peaceful",
            "still",
            "vast",
            "alpine",
            "glacial",
            "volcanic",
        ],
        "features": ["water", "shore", "reflections", "ripples", "wildlife", "reeds", "lily pads"],
        "lighting": [
            "mirror reflections",
            "mist rising",
            "sunrise over water",
            "moonlight on water",
        ],
        "modifiers": ["lakeside", "lake shore", "lake view", "across the lake"],
        "atmosphere": ["calm", "reflective", "serene", "still"]
    },
    "desert": {
        "category": "natural",
        "descriptors": ["vast", "endless", "dry", "barren", "sandy", "hot", "arid", "expansive"],
        "features": ["sand dunes", "cacti", "rock formations", "cracked earth", "oasis", "mirages"],
        "lighting": ["harsh sunlight", "desert sunset", "heat waves", "golden sands"],
        "modifiers": ["sand dunes", "desert oasis", "desert plains", "rocky desert"],
        "atmosphere": ["harsh", "desolate", "extreme", "minimalist"]
    },
    "jungle": {
        "category": "natural",
        "descriptors": [
            "dense",
            "lush",
            "exotic",
            "tropical",
            "humid",
            "wild",
            "vibrant",
            "overgrown",
        ],
        "features": ["vines", "exotic flowers", "giant leaves", "trees", "ferns", "wildlife"],
        "lighting": ["filtered sunlight", "jungle shade", "humid glow", "dappled light"],
        "modifiers": ["deep jungle", "jungle clearing", "jungle canopy", "jungle floor"],
        "atmosphere": ["wild", "lush", "primal", "teeming", "vibrant"]
    },
    "meadow": {
        "category": "natural",
        "descriptors": [
            "grassy",
            "flowery",
            "open",
            "sunny",
            "idyllic",
            "peaceful",
            "rolling",
            "wild",
        ],
        "features": ["grass", "wildflowers", "butterflies", "bees", "gentle hills", "wildlife"],
        "lighting": ["bright sunshine", "golden hour", "morning dew", "summer light"],
        "modifiers": ["open meadow", "wildflower meadow", "grassy field", "hillside meadow"],
        "atmosphere": ["peaceful", "idyllic", "open", "cheerful"]
    },
    
    # Urban environments
    "city": {
        "category": "urban",
        "descriptors": [
            "bustling",
            "modern",
            "cosmopolitan",
            "crowded",
            "busy",
            "towering",
            "vibrant",
        ],
        "features": ["buildings", "skyscrapers", "streets", "traffic", "people", "storefronts"],
        "lighting": [
            "city lights",
            "neon glow",
            "street lamps",
            "car headlights",
            "building reflections",
        ],
        "modifiers": ["downtown", "city center", "urban landscape", "city streets", "metropolis"],
        "atmosphere": ["energetic", "fast-paced", "vibrant", "diverse", "urban"]
    },
    "park": {
        "category": "urban",
        "descriptors": ["green", "peaceful", "public", "spacious", "landscaped", "well-maintained"],
        "features": ["trees", "grass", "paths", "benches", "fountain", "playground", "ponds"],
        "lighting": ["afternoon sunshine", "park lights", "dappled shade", "sunset in the park"],
        "modifiers": ["city park", "public garden", "park grounds", "garden path"],
        "atmosphere": ["relaxing", "refreshing", "social", "recreational"]
    },
    "coffee_shop": {
        "category": "urban",
        "descriptors": ["cozy", "busy", "warm", "inviting", "trendy", "quaint", "bustling"],
        "features": ["tables", "chairs", "counter", "coffee machines", "pastries", "people"],
        "lighting": ["warm lights", "window light", "cozy atmosphere", "cafe ambiance"],
        "modifiers": ["corner cafe", "busy coffee shop", "quiet cafe", "specialty coffee bar"],
        "atmosphere": ["cozy", "social", "aromatic", "comfortable"]
    },
    "street": {
        "category": "urban",
        "descriptors": ["busy", "narrow", "wide", "cobblestone", "modern", "old", "picturesque"],
        "features": ["sidewalks", "buildings", "cars", "shops", "pedestrians", "street signs"],
        "lighting": ["street lamps", "car lights", "shop windows", "twilight glow"],
        "modifiers": ["main street", "side street", "quiet street", "busy thoroughfare"],
        "atmosphere": ["lively", "bustling", "dynamic", "changing"]
    },
    "alley": {
        "category": "urban",
        "descriptors": ["narrow", "hidden", "dark", "secluded", "quiet", "mysterious", "urban"],
        "features": ["brick walls", "fire escapes", "dumpsters", "backdoors", "graffiti"],
        "lighting": ["dim lights", "shadows", "single light source", "moonlight shaft"],
        "modifiers": ["back alley", "service alley", "hidden passage", "shortcut"],
        "atmosphere": ["secretive", "hidden", "private", "shadowy"]
    },
    
    # Fantasy environments
    "castle": {
        "category": "fantasy",
        "descriptors": ["ancient", "stone", "massive", "fortified", "royal", "medieval", "grand"],
        "features": ["towers", "battlements", "halls", "tapestries", "thrones", "courtyards", "moat"],
        "lighting": ["torch light", "candlelight", "fireplaces", "stained glass windows"],
        "modifiers": ["castle keep", "throne room", "great hall", "royal chambers", "dungeon"],
        "atmosphere": ["majestic", "imposing", "historical", "powerful"]
    },
    "ruins": {
        "category": "fantasy",
        "descriptors": ["ancient", "crumbling", "overgrown", "forgotten", "mysterious", "stone"],
        "features": ["broken columns", "fallen walls", "moss", "vines", "collapsed roofs"],
        "lighting": ["filtered sunlight", "shadows", "moonlight through gaps", "mist"],
        "modifiers": ["temple ruins", "forgotten city", "ancient structure", "archaeological site"],
        "atmosphere": ["mysterious", "forgotten", "historical", "nostalgic"]
    },
    "cave": {
        "category": "fantasy",
        "descriptors": ["dark", "damp", "echoing", "vast", "narrow", "hidden", "mysterious"],
        "features": ["stalactites", "stalagmites", "rock formations", "underground pools"],
        "lighting": ["dim light", "darkness", "filtered light from entrance", "glowing crystals"],
        "modifiers": ["cave entrance", "deep cavern", "underground chamber", "tunnel system"],
        "atmosphere": ["mysterious", "enclosed", "hidden", "primordial"]
    },
    "magical_forest": {
        "category": "fantasy",
        "descriptors": ["enchanted", "glowing", "mystical", "ancient", "magical", "otherworldly"],
        "features": ["glowing plants", "magical creatures", "strange flowers", "ancient trees"],
        "lighting": ["magical glow", "bioluminescence", "shimmering light", "ethereal illumination"],
        "modifiers": ["fairy glen", "enchanted grove", "magical clearing", "elven forest"],
        "atmosphere": ["magical", "enchanting", "wondrous", "otherworldly"]
    },
    
    # Sci-fi environments
    "space_station": {
        "category": "sci-fi",
        "descriptors": ["futuristic", "high-tech", "sleek", "metallic", "sterile", "advanced"],
        "features": ["control panels", "viewports", "airlocks", "living quarters", "machinery"],
        "lighting": ["artificial lights", "blinking indicators", "starlight from windows", "holographic displays"],
        "modifiers": ["command center", "living quarters", "engineering section", "observation deck"],
        "atmosphere": ["technological", "advanced", "engineered", "efficient"]
    },
    "cyberpunk_city": {
        "category": "sci-fi",
        "descriptors": ["neon-lit", "dystopian", "futuristic", "gritty", "overcrowded", "polluted"],
        "features": ["skyscrapers", "neon signs", "holographic ads", "flying vehicles", "dark alleys"],
        "lighting": ["neon lights", "holographic glow", "artificial lighting", "rain-slick reflections"],
        "modifiers": ["neon district", "corporate zone", "undercity", "slums", "high-tech quarter"],
        "atmosphere": ["gritty", "high-tech", "oppressive", "energetic", "chaotic"]
    },
    "laboratory": {
        "category": "sci-fi",
        "descriptors": ["high-tech", "sterile", "modern", "advanced", "scientific", "clean"],
        "features": ["equipment", "computers", "test tubes", "experimental apparatus", "monitors"],
        "lighting": ["bright lights", "LED indicators", "computer screens", "laser beams"],
        "modifiers": ["research lab", "testing facility", "experimental chamber", "control room"],
        "atmosphere": ["clinical", "scientific", "precise", "controlled"]
    }
}

# =============================================================================
# Species-Specific Habitat Mappings
# =============================================================================

SPECIES_HABITATS: Dict[str, List[str]] = {
    # Canines
    "wolf": ["forest", "mountains", "wilderness", "tundra", "taiga", "snowy landscape"],
    "fox": ["forest", "woodland", "meadow", "countryside", "rural", "fields"],
    "dog": ["urban", "home", "backyard", "city park", "suburban", "countryside"],
    
    # Felines
    "cat": ["urban", "home", "garden", "city streets", "alleyway", "countryside"],
    "tiger": ["jungle", "rainforest", "river", "tall grass", "bamboo forest", "tropical forest"],
    "lion": ["savannah", "grasslands", "plains", "rocky outcrop", "pride rock", "desert edge"],
    "leopard": ["forest", "jungle", "trees", "savannah", "rocky areas", "woodland"],
    
    # Reptiles
    "dragon": ["mountains", "caves", "castle", "volcano", "ruins", "magical forest", "cliffs"],
    "lizard": ["desert", "rocks", "tropical", "jungle", "warm environments", "sunny areas"],
    "snake": ["jungle", "desert", "forest", "grassland", "rocky areas", "tropical environments"],
    
    # Lagomorphs
    "rabbit": ["meadow", "forest", "grassland", "countryside", "garden", "burrow", "fields"],
    "bunny": ["meadow", "garden", "home", "pet store", "farm", "countryside", "fields"],
    
    # Equines
    "horse": ["meadow", "farm", "ranch", "countryside", "field", "grassland", "stable"],
    "pony": ["meadow", "farm", "ranch", "countryside", "stable", "field", "paddock"],
    
    # Aquatic
    "shark": ["ocean", "reef", "deep sea", "marine environment", "underwater", "coastal waters"],
    "dolphin": ["ocean", "sea", "underwater", "coastal waters", "bay", "marine environment"],
    "otter": ["river", "lake", "coast", "wetland", "stream", "waterfall", "pond"],
    
    # Avian
    "bird": ["sky", "trees", "forest", "garden", "mountain", "coast", "city park"],
    "eagle": ["mountains", "cliffs", "forest", "coastline", "peaks", "high places"],
    "owl": ["forest", "trees", "woodland", "barn", "night setting", "old buildings"],
    
    # Fantasy species
    "sergal": ["mountains", "space station", "futuristic city", "snowy terrain", "tundra"],
    "protogen": ["space station", "cyberpunk city", "laboratory", "high-tech environment", "futuristic setting"],
    "primagen": ["space station", "cyberpunk city", "laboratory", "high-tech environment", "futuristic setting"],
    "kobold": ["caves", "mines", "underground", "mountains", "volcanic regions", "fantasy tavern"],
    
    # Other mammals
    "deer": ["forest", "meadow", "woodland", "countryside", "mountains", "glen"],
    "raccoon": ["forest", "urban", "city park", "suburbs", "night setting", "trash area"],
    "bear": ["forest", "mountains", "river", "wilderness", "caves", "woods"],
    "squirrel": ["forest", "park", "trees", "urban park", "woodland", "garden"],
    
    # Default for unknown species
    "default": ["forest", "meadow", "city", "urban", "park", "suburban", "rural", "indoor", "outdoor"]
} 