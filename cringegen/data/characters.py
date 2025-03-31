"""
Unified Character System for CringeGen

This module provides a centralized character system that consolidates character data
from multiple sources including furry characters, anime characters, and game characters.
It provides comprehensive references for character types, archetypes, and specific
named characters across different media.

Key components:
- Famous anthro/furry characters by media source
- Character archetypes and types
- Anime character features
- Character to full name mappings
"""

from typing import Dict, List, Set, Tuple, Any, Optional

# Famous anthro/furry characters organized by source
FAMOUS_FURRY_CHARACTERS: Dict[str, List[str]] = {
    "games": {
        "elden_ring": ["Blaidd the Half-Wolf", "Blaidd", "Maliketh the Black Blade", "Maliketh"],
        "sonic": [
            "Sonic the Hedgehog",
            "Sonic",
            "Tails",
            "Miles Prower",
            "Knuckles the Echidna",
            "Knuckles",
            "Amy Rose",
            "Shadow the Hedgehog",
            "Shadow",
            "Rouge the Bat",
            "Rouge",
            "Silver the Hedgehog",
            "Silver",
            "Blaze the Cat",
            "Blaze",
            "Vector the Crocodile",
            "Vector",
            "Espio the Chameleon",
            "Espio",
            "Charmy Bee",
            "Charmy",
            "Big the Cat",
            "Big",
        ],
        "star_fox": [
            "Fox McCloud",
            "Fox",
            "Falco Lombardi",
            "Falco",
            "Slippy Toad",
            "Slippy",
            "Peppy Hare",
            "Peppy",
            "Krystal",
            "Wolf O'Donnell",
            "Wolf",
        ],
        "crash_bandicoot": [
            "Crash Bandicoot",
            "Crash",
            "Coco Bandicoot",
            "Coco",
            "Crunch Bandicoot",
            "Crunch",
            "Tiny Tiger",
            "Tiny",
            "Dingodile",
        ],
        "sly_cooper": ["Sly Cooper", "Sly", "Bentley", "Murray", "Carmelita Fox", "Carmelita"],
        "spyro": [
            "Spyro the Dragon",
            "Spyro",
            "Hunter the Cheetah",
            "Hunter",
            "Bianca",
            "Sheila the Kangaroo",
            "Sheila",
            "Elora the Faun",
            "Elora",
        ],
        "ratchet_and_clank": ["Ratchet", "Lombax"],
        "animal_crossing": ["Isabelle", "Tom Nook", "K.K. Slider", "Blathers"],
        "pokemon": [
            "Lucario",
            "Zoroark",
            "Zeraora",
            "Incineroar",
            "Cinderace",
            "Lopunny",
            "Renamon",
        ],
        "undertale": ["Toriel", "Asgore", "Asriel", "Undyne", "Alphys", "Sans", "Papyrus"],
        "dust": ["Dust", "Fidget", "Ahrah"],
        "night_in_the_woods": ["Mae Borowski", "Mae", "Gregg", "Angus", "Bea"],
        "freedom_planet": ["Lilac", "Carol", "Milla", "Neera"],
        "beastars": ["Legoshi", "Haru", "Louis", "Juno", "Gouhin", "Jack"],
        "bloody_roar": ["Yugo", "Alice", "Long", "Uriko", "Bakuryu"],
        "dark_souls": ["Sif the Great Grey Wolf", "Sif", "Alvina", "Great Grey Wolf"],
        "divinity": ["Ifan ben-Mezd", "Ifan", "Sebille", "Beast", "Fane", "Lohse", "Red Prince"],
        "league_of_legends": [
            "Ahri",
            "Rengar",
            "Volibear",
            "Warwick",
            "Nasus",
            "Renekton",
            "Wukong",
            "Yuumi",
            "Kindred",
        ],
        "armello": ["Thane", "Amber", "Mercurio", "Zosha", "Barnaby", "Sana"],
        "final_fantasy": [
            "Kimahri Ronso",
            "Kimahri",
            "Freya Crescent",
            "Freya",
            "Red XIII",
            "Nanaki",
        ],
    },
    "cartoons": {
        "disney": [
            "Mickey Mouse",
            "Mickey",
            "Minnie Mouse",
            "Minnie",
            "Donald Duck",
            "Donald",
            "Goofy",
            "Daisy Duck",
            "Daisy",
            "Pluto",
            "Chip and Dale",
            "Chip",
            "Dale",
            "Nick Wilde",
            "Judy Hopps",
            "Robin Hood",
            "Maid Marian",
            "Little John",
            "Simba",
            "Nala",
            "Mufasa",
            "Scar",
            "Timon",
            "Pumbaa",
            "Rafiki",
            "Zazu",
            "Shere Khan",
            "Bagheera",
            "Baloo",
            "King Louie",
            "Kaa",
        ],
        "warner_bros": [
            "Bugs Bunny",
            "Bugs",
            "Daffy Duck",
            "Daffy",
            "Porky Pig",
            "Porky",
            "Sylvester",
            "Tweety",
            "Wile E. Coyote",
            "Road Runner",
            "Taz",
            "Tasmanian Devil",
            "Lola Bunny",
            "Lola",
            "Pepe Le Pew",
            "Pepe",
            "Marvin the Martian",
            "Marvin",
        ],
        "don_bluth": [
            "Charlie Barkin",
            "Charlie",
            "Itchy Itchiford",
            "Itchy",
            "Mrs. Brisby",
            "Justin",
            "Jenner",
            "Jeremy",
            "Fievel Mousekewitz",
            "Fievel",
            "Tiger",
        ],
        "dreamworks": [
            "Po",
            "Tigress",
            "Tai Lung",
            "Shifu",
            "Alex the Lion",
            "Alex",
            "Gloria",
            "Marty",
            "Melman",
            "King Julien",
            "Puss in Boots",
            "Puss",
        ],
        "misc_cartoons": [
            "Tom and Jerry",
            "Tom",
            "Jerry",
            "Brian Griffin",
            "Brian",
            "Scooby-Doo",
            "Scooby",
            "Tony the Tiger",
            "Tony",
            "Chester Cheetah",
            "Chester",
        ],
    },
    "comics": {
        "webcomics": [
            "Jack Farrell",
            "Sabrina Mink",
            "Thomas Omega",
            "Thomas",
            "Flora Winters",
            "Flora",
            "Trace Legacy",
            "Trace",
            "Keith Keiser",
            "Keith",
            "Natani",
            "Red",
            "Raine Silverlock",
            "Raine",
            "Mink",
            "Jay Naylor",
            "Jay",
        ],
        "manga": ["Beast Complex characters", "Legoshi", "Haru", "Louis", "Juno"],
    },
}

# Mapping of short names to full character names
CHARACTER_TO_FULL_NAME: Dict[str, str] = {
    # Video game characters
    "Sonic": "Sonic the Hedgehog",
    "Tails": "Miles 'Tails' Prower",
    "Knuckles": "Knuckles the Echidna",
    "Shadow": "Shadow the Hedgehog",
    "Rouge": "Rouge the Bat",
    "Silver": "Silver the Hedgehog",
    "Blaze": "Blaze the Cat",
    "Vector": "Vector the Crocodile",
    "Espio": "Espio the Chameleon",
    "Charmy": "Charmy Bee",
    "Big": "Big the Cat",
    "Fox": "Fox McCloud",
    "Falco": "Falco Lombardi",
    "Slippy": "Slippy Toad",
    "Peppy": "Peppy Hare",
    "Wolf": "Wolf O'Donnell",
    "Crash": "Crash Bandicoot",
    "Coco": "Coco Bandicoot",
    "Sly": "Sly Cooper",
    "Carmelita": "Carmelita Fox",
    "Spyro": "Spyro the Dragon",
    "Hunter": "Hunter the Cheetah",
    "Sheila": "Sheila the Kangaroo",
    "Elora": "Elora the Faun",
    "Mae": "Mae Borowski",
    "Sif": "Sif the Great Grey Wolf",
    "Ifan": "Ifan ben-Mezd",
    "Kimahri": "Kimahri Ronso",
    "Freya": "Freya Crescent",
    "Nanaki": "Red XIII",
    # Cartoon characters
    "Mickey": "Mickey Mouse",
    "Minnie": "Minnie Mouse",
    "Donald": "Donald Duck",
    "Daisy": "Daisy Duck",
    "Bugs": "Bugs Bunny",
    "Daffy": "Daffy Duck",
    "Porky": "Porky Pig",
    "Lola": "Lola Bunny",
    "Pepe": "Pepe Le Pew",
    "Marvin": "Marvin the Martian",
    "Charlie": "Charlie Barkin",
    "Itchy": "Itchy Itchiford",
    "Fievel": "Fievel Mousekewitz",
    "Alex": "Alex the Lion",
    "Puss": "Puss in Boots",
    "Tom": "Tom Cat",
    "Jerry": "Jerry Mouse",
    "Brian": "Brian Griffin",
    "Scooby": "Scooby-Doo",
    "Tony": "Tony the Tiger",
    "Chester": "Chester Cheetah",
    # Comic characters
    "Thomas": "Thomas Omega",
    "Flora": "Flora Winters",
    "Trace": "Trace Legacy",
    "Keith": "Keith Keiser",
    "Raine": "Raine Silverlock",
    "Jay": "Jay Naylor",
}

# Combinatorial collection of all anthro characters from all sources
ALL_ANTHRO_CHARACTERS: List[str] = [
    # This is a flattened list of all character names and their variants
    # from FAMOUS_FURRY_CHARACTERS, compiled for easy lookup
    "Blaidd the Half-Wolf", "Blaidd", "Maliketh the Black Blade", "Maliketh",
    "Sonic the Hedgehog", "Sonic", "Tails", "Miles Prower", "Knuckles the Echidna", "Knuckles",
    "Amy Rose", "Shadow the Hedgehog", "Shadow", "Rouge the Bat", "Rouge", 
    "Silver the Hedgehog", "Silver", "Blaze the Cat", "Blaze", "Vector the Crocodile", "Vector",
    "Espio the Chameleon", "Espio", "Charmy Bee", "Charmy", "Big the Cat", "Big",
    "Fox McCloud", "Fox", "Falco Lombardi", "Falco", "Slippy Toad", "Slippy",
    "Peppy Hare", "Peppy", "Krystal", "Wolf O'Donnell", "Wolf",
    "Crash Bandicoot", "Crash", "Coco Bandicoot", "Coco", "Crunch Bandicoot", "Crunch",
    "Tiny Tiger", "Tiny", "Dingodile",
    "Sly Cooper", "Sly", "Bentley", "Murray", "Carmelita Fox", "Carmelita",
    "Spyro the Dragon", "Spyro", "Hunter the Cheetah", "Hunter", "Bianca",
    "Sheila the Kangaroo", "Sheila", "Elora the Faun", "Elora",
    "Ratchet", "Lombax", "Isabelle", "Tom Nook", "K.K. Slider", "Blathers",
    "Lucario", "Zoroark", "Zeraora", "Incineroar", "Cinderace", "Lopunny", "Renamon",
    "Toriel", "Asgore", "Asriel", "Undyne", "Alphys", "Sans", "Papyrus",
    "Dust", "Fidget", "Ahrah", "Mae Borowski", "Mae", "Gregg", "Angus", "Bea",
    "Lilac", "Carol", "Milla", "Neera", "Legoshi", "Haru", "Louis", "Juno", "Gouhin", "Jack",
    "Yugo", "Alice", "Long", "Uriko", "Bakuryu",
    "Sif the Great Grey Wolf", "Sif", "Alvina", "Great Grey Wolf",
    "Ifan ben-Mezd", "Ifan", "Sebille", "Beast", "Fane", "Lohse", "Red Prince",
    "Ahri", "Rengar", "Volibear", "Warwick", "Nasus", "Renekton", "Wukong", "Yuumi", "Kindred",
    "Thane", "Amber", "Mercurio", "Zosha", "Barnaby", "Sana",
    "Kimahri Ronso", "Kimahri", "Freya Crescent", "Freya", "Red XIII", "Nanaki",
    "Mickey Mouse", "Mickey", "Minnie Mouse", "Minnie", "Donald Duck", "Donald", "Goofy",
    "Daisy Duck", "Daisy", "Pluto", "Chip and Dale", "Chip", "Dale", "Nick Wilde", "Judy Hopps",
    "Robin Hood", "Maid Marian", "Little John", "Simba", "Nala", "Mufasa", "Scar", "Timon", "Pumbaa",
    "Rafiki", "Zazu", "Shere Khan", "Bagheera", "Baloo", "King Louie", "Kaa",
    "Bugs Bunny", "Bugs", "Daffy Duck", "Daffy", "Porky Pig", "Porky", "Sylvester", "Tweety",
    "Wile E. Coyote", "Road Runner", "Taz", "Tasmanian Devil", "Lola Bunny", "Lola",
    "Pepe Le Pew", "Pepe", "Marvin the Martian", "Marvin",
    "Charlie Barkin", "Charlie", "Itchy Itchiford", "Itchy", "Mrs. Brisby", "Justin", "Jenner", "Jeremy",
    "Fievel Mousekewitz", "Fievel", "Tiger",
    "Po", "Tigress", "Tai Lung", "Shifu", "Alex the Lion", "Alex", "Gloria", "Marty", "Melman", 
    "King Julien", "Puss in Boots", "Puss",
    "Tom and Jerry", "Tom", "Jerry", "Brian Griffin", "Brian", "Scooby-Doo", "Scooby",
    "Tony the Tiger", "Tony", "Chester Cheetah", "Chester",
    "Jack Farrell", "Sabrina Mink", "Thomas Omega", "Thomas", "Flora Winters", "Flora",
    "Trace Legacy", "Trace", "Keith Keiser", "Keith", "Natani", "Red", "Raine Silverlock", "Raine",
    "Mink", "Jay Naylor", "Jay"
]

# Anime character archetypes
ANIME_CHARACTER_TYPES: Dict[str, List[str]] = {
    # Kemonomimi (Animal ears/tails on human forms)
    "kemonomimi": [
        "nekomimi",
        "cat girl",
        "cat boy",
        "neko",
        "inumimi",
        "dog girl",
        "dog boy",
        "inu",
        "kitsunemimi",
        "fox girl",
        "fox boy",
        "kitsune",
        "usagimimi",
        "bunny girl",
        "bunny boy",
        "usagi",
        "okami",
        "wolf girl",
        "wolf boy",
        "ryumimi",
        "dragon girl",
        "dragon boy",
        "ryu",
        "bat girl",
        "bat boy",
        "bat ears",
        "tanuki girl",
        "tanuki boy",
        "raccoon ears",
    ],
    # Full transformation characters
    "monster_people": [
        "lamia",
        "snake girl",
        "snake boy",
        "harpy",
        "bird girl",
        "bird boy",
        "centaur",
        "horse girl",
        "horse boy",
        "mermaid",
        "merman",
        "fish tail",
        "slime girl",
        "slime boy",
        "slime person",
        "arachne",
        "spider girl",
        "spider boy",
    ],
    # Fantasy races
    "fantasy_races": [
        "elf",
        "high elf",
        "dark elf",
        "night elf",
        "wood elf",
        "dwarf",
        "halfling",
        "gnome",
        "orc",
        "goblin",
        "ogre",
        "troll",
        "demon",
        "demon girl",
        "demon boy",
        "succubus",
        "incubus",
        "angel",
        "angel girl",
        "angel boy",
        "vampire",
        "vampire girl",
        "vampire boy",
        "werewolf",
        "werewolf girl",
        "werewolf boy",
        "fairy",
        "pixie",
        "sprite",
    ],
    # Common anime archetypes
    "character_archetypes": [
        "tsundere",
        "yandere",
        "kuudere",
        "dandere",
        "deredere",
        "loli",
        "shota",
        "bishonen",
        "bishoujo",
        "trap",
        "reverse trap",
        "ojou-sama",
        "onee-san",
        "onii-chan",
        "imouto",
        "otouto",
        "magical girl",
        "magical boy",
        "mahou shoujo",
        "maid",
        "butler",
        "shrine maiden",
        "miko",
        "ninja",
        "samurai",
        "ronin",
        "swordsman",
        "hero",
        "villain",
        "anti-hero",
        "rival",
        "protagonist",
        "antagonist",
        "student",
        "teacher",
        "senpai",
        "kouhai",
        "idol",
        "singer",
        "band member",
    ],
}

# Anime character features
ANIME_CHARACTER_FEATURES: Dict[str, List[str]] = {
    "hair_colors": [
        "black",
        "brown",
        "blonde",
        "red",
        "pink",
        "blue",
        "purple",
        "green",
        "white",
        "silver",
        "orange",
        "cyan",
        "rainbow",
        "multicolored",
        "dual-colored",
        "gradient",
        "ombre",
    ],
    "hair_styles": [
        "long",
        "short",
        "medium",
        "pixie",
        "bob",
        "bun",
        "ponytail",
        "twin tails",
        "twintails",
        "pigtails",
        "braids",
        "ahoge",
        "antenna",
        "drill hair",
        "curly",
        "wavy",
        "straight",
        "spiky",
        "messy",
        "side swept",
        "hime cut",
        "bowl cut",
        "mohawk",
        "undercut",
        "hair bun",
        "ponytail",
        "twin ponytails",
        "side ponytail",
        "double bun",
        "odango",
        "meatballs",
        "chignon",
    ],
    "eye_colors": [
        "blue",
        "green",
        "brown",
        "red",
        "purple",
        "yellow",
        "pink",
        "black",
        "gray",
        "orange",
        "cyan",
        "turquoise",
        "amber",
        "multicolored",
        "heterochromatic",
        "gradient",
    ],
    "eye_types": [
        "large",
        "round",
        "almond-shaped",
        "cat-like",
        "slitted pupils",
        "glowing",
        "spiral",
        "heart-shaped",
        "star-shaped",
        "X-shaped",
        "closed",
        "half-lidded",
        "tareme",
        "tsurime",
        "droopy",
        "sharp",
        "rectangular",
    ],
    "expressions": [
        "smiling",
        "frowning",
        "pouting",
        "crying",
        "laughing",
        "angry",
        "shocked",
        "blushing",
        "embarrassed",
        "smug",
        "determined",
        "sad",
        "happy",
        "emotionless",
        "ecstatic",
        "excited",
        "nervous",
        "serious",
        "playful",
        "teasing",
        "gentle",
        "intimidating",
        "confident",
        "shy",
        "sleepy",
        "dazed",
        "confused",
    ],
    "accessories": [
        "glasses",
        "sunglasses",
        "eyepatch",
        "monocle",
        "hair ribbon",
        "hair clips",
        "hair band",
        "headband",
        "flower",
        "flower crown",
        "tiara",
        "crown",
        "hat",
        "cap",
        "beret",
        "bandana",
        "hijab",
        "veil",
        "mask",
        "earrings",
        "necklace",
        "choker",
        "collar",
        "scarf",
        "tie",
        "bow tie",
        "bracelet",
        "wristband",
        "watch",
        "gloves",
        "arm warmers",
        "leg warmers",
        "stockings",
        "socks",
        "thigh highs",
        "garter belt",
        "belt",
        "suspenders",
        "backpack",
        "bag",
        "purse",
        "weapon",
    ],
}

# Video game character archetypes
GAME_CHARACTER_TYPES: Dict[str, List[str]] = {
    "protagonists": [
        "silent protagonist",
        "voiced protagonist",
        "player avatar",
        "chosen one",
        "reluctant hero",
        "antihero",
        "military hero",
        "adventurer",
        "explorer",
        "detective",
        "mercenary",
        "bounty hunter",
        "soldier",
        "pilot",
        "space marine",
        "knight",
        "warrior",
        "ranger",
        "treasure hunter",
        "outlaw",
        "criminal",
        "underdog",
        "everyman",
        "amnesiac hero",
        "farmboy hero",
        "ordinary person",
        "unlikely hero",
    ],
    "player_classes": [
        "warrior",
        "soldier",
        "fighter",
        "barbarian",
        "berserker",
        "knight",
        "paladin",
        "dark knight",
        "samurai",
        "monk",
        "rogue",
        "thief",
        "assassin",
        "archer",
        "ranger",
        "hunter",
        "mage",
        "wizard",
        "witch",
        "sorcerer",
        "warlock",
        "necromancer",
        "druid",
        "shaman",
        "healer",
        "cleric",
        "priest",
        "bard",
        "summoner",
        "gunslinger",
        "engineer",
        "technician",
        "hacker",
        "psion",
        "psychic",
    ],
    "npcs": [
        "quest giver",
        "merchant",
        "shopkeeper",
        "blacksmith",
        "innkeeper",
        "guard",
        "noble",
        "peasant",
        "scholar",
        "sage",
        "mentor",
        "ally",
        "companion",
        "sidekick",
        "love interest",
        "rival",
        "nemesis",
        "villain",
        "boss",
        "miniboss",
        "minion",
        "enemy",
        "monster",
        "creature",
        "familiar",
        "pet",
        "mount",
    ],
    "anthro_archetypes": [
        "animal warrior",
        "beast master",
        "shapeshifter",
        "were-creature",
        "animal companion",
        "feral",
        "wild",
        "beast",
        "monster",
        "creature",
        "familiar",
        "pet",
        "mount",
        "spirit animal",
        "animal god",
        "animal spirit",
        "animal deity",
        "animal guardian",
        "animal protector",
        "animal guide",
    ],
}

# Default character generation weights/parameters
DEFAULT_CHARACTER_WEIGHTS: Dict[str, float] = {
    "anthro": 0.85,           # vs feral
    "western": 0.6,           # vs anime
    "popular_species": 0.7,   # vs rare species
    "fantasy_species": 0.35,  # vs real species
    "male": 0.5,              # vs female
    "adult": 0.8,             # vs young
    "casual_clothing": 0.7,   # vs formal 
    "custom_outfit": 0.6,     # vs generic
    "multi_colored": 0.5,     # vs mono color
    "common_pattern": 0.6,    # vs unusual patterns
    "accessories": 0.65,      # likelihood of accessories
    "named_character": 0.2,   # vs original character
}

# Character accessories
ACCESSORIES: Dict[str, List[str]] = {
    "jewelry": [
        "necklace", "pendant", "choker", "collar", "earrings", "studs", "hoops",
        "bracelet", "wristband", "anklet", "rings", "piercings", "chains", "amulet"
    ],
    "headwear": [
        "hat", "cap", "beanie", "beret", "headband", "bandana", "ribbon", "bow",
        "flower crown", "crown", "tiara", "circlet", "glasses", "sunglasses", "goggles", 
        "eyepatch", "visor", "helmet", "hood", "veil", "face mask", "earmuffs", "ear covers"
    ],
    "neckwear": [
        "scarf", "ascot", "tie", "bowtie", "neckerchief", "bandana", "cravat"
    ],
    "technology": [
        "headphones", "earbuds", "smartwatch", "phone", "tablet", "VR headset", 
        "cybernetic implants", "augmented reality lens", "technomancer gear"
    ],
    "miscellaneous": [
        "backpack", "messenger bag", "purse", "satchel", "fanny pack", "holster",
        "belt pouches", "badge", "pins", "buttons", "patches", "bandages", "tattoos", 
        "body paint", "markings", "scars", "cybernetic arm", "cybernetic leg", "prosthetic"
    ],
    "spiritual": [
        "crystal", "totem", "amulet", "runes", "talismans", "prayer beads", "sacred texts"
    ],
    "weapons": [
        "sword", "dagger", "knife", "bow", "arrows", "quiver", "staff", "wand", "gun",
        "pistol", "rifle", "lance", "spear", "shield", "gauntlets", "throwing stars"
    ],
}

# Clothing items by category
CLOTHING_ITEMS: Dict[str, List[str]] = {
    "tops": [
        "t-shirt", "shirt", "blouse", "tank top", "crop top", "tube top", "halter top",
        "sweater", "hoodie", "cardigan", "blazer", "suit jacket", "vest", "tunic", 
        "turtleneck", "button-up", "polo", "flannel", "jersey", "bodysuit", "robe",
        "kimono", "yukata", "hanfu"
    ],
    "bottoms": [
        "jeans", "pants", "slacks", "chinos", "shorts", "cargo pants", "cargo shorts",
        "skirt", "miniskirt", "maxi skirt", "leggings", "joggers", "sweatpants", 
        "trousers", "khakis", "capris", "culottes", "sarong"
    ],
    "dresses": [
        "dress", "sundress", "mini dress", "maxi dress", "evening gown", "cocktail dress",
        "bodycon dress", "slip dress", "sheath dress", "wrap dress", "a-line dress",
        "ballgown", "qipao", "cheongsam", "hanbok", "robe"
    ],
    "outerwear": [
        "jacket", "coat", "raincoat", "trench coat", "parka", "windbreaker", "shawl",
        "poncho", "cape", "cloak", "bomber jacket", "denim jacket", "leather jacket",
        "puffer jacket", "cardigan", "fleece"
    ],
    "footwear": [
        "shoes", "sneakers", "boots", "high boots", "thigh-high boots", "sandals", 
        "loafers", "moccasins", "oxfords", "flip-flops", "slippers", "heels", "platforms",
        "combat boots", "hiking boots", "clogs", "stilettos", "flats", "mary janes"
    ],
    "formal": [
        "suit", "tuxedo", "dress shirt", "slacks", "evening gown", "cocktail dress",
        "formal jacket", "formal trousers", "tie", "bowtie", "cummerbund", "formal coat",
        "formal gloves", "formal shoes", "formal hat", "formal vest", "formal cape"
    ],
    "sports": [
        "gym shorts", "track pants", "sports bra", "jersey", "tracksuit", "sweatshirt",
        "sweatpants", "running shoes", "baseball cap", "wristbands", "headband",
        "athletic socks", "leotard", "bike shorts", "swimming trunks", "swimsuit"
    ],
    "sleepwear": [
        "pajamas", "robe", "nightgown", "nightshirt", "nightdress", "sleep mask", 
        "slippers", "loungewear", "negligee", "bathrobe"
    ],
    "underwear": [
        "underwear", "boxers", "briefs", "boxer briefs", "panties", "thong", "g-string",
        "bra", "sports bra", "bralette", "corset", "bustier", "teddy", "camisole",
        "bodysuit", "lingerie", "stockings", "garters", "garter belt"
    ],
    "swimwear": [
        "swimsuit", "bikini", "one-piece", "swim trunks", "swim shorts", "tankini",
        "monokini", "rash guard", "wetsuit", "board shorts", "swimming briefs"
    ],
    "accessories": [
        "gloves", "mittens", "scarf", "tie", "bowtie", "neckerchief", "belt", "suspenders",
        "socks", "stockings", "tights", "leg warmers", "arm warmers", "wristbands",
        "sweatbands", "headband", "hairband", "hair clips", "hat", "cap", "beanie", "beret"
    ],
    "culturally_specific": [
        "kimono", "yukata", "hakama", "qipao", "hanfu", "hanbok", "sari", "dhoti",
        "kurta", "salwar kameez", "kilt", "dirndl", "lederhosen", "poncho", "serape",
        "keffiyeh", "turban", "shalwar", "kaftan", "dashiki", "boubou"
    ]
}

# Clothing styles with characteristic elements
CLOTHING_STYLES: Dict[str, List[str]] = {
    "casual": [
        "jeans and t-shirt", "hoodie and sweatpants", "shorts and tank top",
        "casual dress", "sneakers", "casual jacket", "everyday wear", "relaxed fit"
    ],
    "formal": [
        "suit", "tuxedo", "dress shirt", "formal dress", "formal gown", "formal wear",
        "business formal", "evening wear", "black tie", "white tie", "tailcoat"
    ],
    "business": [
        "business suit", "blazer", "dress shirt", "blouse", "slacks", "pencil skirt",
        "loafers", "business casual", "office wear", "professional attire"
    ],
    "streetwear": [
        "hoodie", "graphic tee", "baggy pants", "snapback", "beanie", "sneakers",
        "joggers", "urban style", "streetwear", "skater style", "hip hop fashion"
    ],
    "punk": [
        "leather jacket", "ripped jeans", "band t-shirt", "studs", "chains", "combat boots",
        "fingerless gloves", "mohawk", "punk aesthetic", "safety pins", "torn clothing"
    ],
    "goth": [
        "black clothing", "corset", "lace", "platform boots", "fishnet", "leather",
        "choker", "dark makeup", "victorian inspired", "gothic fashion", "dark romantic"
    ],
    "bohemian": [
        "flowing dress", "loose clothing", "natural fabrics", "earthy tones",
        "fringe", "layered clothing", "patterned", "boho chic", "hippie inspired"
    ],
    "vintage": [
        "retro", "classic", "50s style", "60s mod", "70s disco", "80s", "90s grunge",
        "pin-up", "rockabilly", "swing era", "flapper style", "vintage inspired"
    ],
    "athleisure": [
        "leggings", "sports bra", "track pants", "hoodie", "athletic shoes",
        "sweatshirt", "joggers", "activewear", "performance wear", "gym clothes"
    ],
    "cyberpunk": [
        "tech-wear", "neon accents", "futuristic", "urban tech", "goggles",
        "cybernetic implants", "utility-focused", "dystopian fashion", "glowing elements"
    ],
    "steampunk": [
        "victorian inspired", "brass accents", "goggles", "pocket watch", "waistcoat",
        "corset", "top hat", "gears", "mechanical elements", "airship captain"
    ],
    "fantasy": [
        "medieval inspired", "renaissance", "elven", "dwarven", "magical robes",
        "adventurer's garb", "leather armor", "fantasy elements", "enchanted clothing"
    ],
    "sci-fi": [
        "futuristic", "space suit", "minimalist", "jumpsuit", "high tech",
        "utilitarian", "sleek design", "advanced materials", "otherworldly"
    ],
    "military": [
        "uniform", "camo pattern", "tactical gear", "combat boots", "utility belt",
        "dog tags", "military jacket", "service dress", "battle ready", "armored"
    ],
    "academic": [
        "blazer with elbow patches", "oxford shoes", "tweed", "turtleneck", "glasses",
        "scholarly", "preppy", "boarding school uniform", "professor style"
    ],
    "sporty": [
        "jersey", "athletic shorts", "track suit", "sneakers", "baseball cap",
        "team colors", "sports equipment", "athletic", "varsity jacket"
    ]
}

# Facial expressions for character descriptions
FACIAL_EXPRESSIONS: Dict[str, List[str]] = {
    "happy": [
        "smiling", "grinning", "beaming", "cheerful", "joyful", "radiant smile",
        "happy expression", "bright-eyed", "laughing", "playful smile", "warm smile"
    ],
    "sad": [
        "frowning", "downcast", "teary-eyed", "sullen", "dejected", "mournful",
        "melancholy", "sorrowful", "forlorn", "downturned mouth", "sad eyes"
    ],
    "angry": [
        "scowling", "glaring", "furious", "enraged", "indignant", "irate",
        "grumpy", "fuming", "seething", "fierce expression", "furrowed brow"
    ],
    "surprised": [
        "wide-eyed", "jaw dropped", "startled", "astonished", "shocked",
        "amazed", "mouth agape", "bewildered", "stunned expression"
    ],
    "afraid": [
        "fearful", "terrified", "scared", "panicked", "anxious", "worried",
        "frightened", "timid", "cowering", "wide-eyed with fear"
    ],
    "thoughtful": [
        "pensive", "contemplative", "reflective", "meditative", "philosophical",
        "introspective", "deep in thought", "thoughtful gaze", "considering"
    ],
    "confident": [
        "self-assured", "proud", "assertive", "bold", "dauntless", "poised",
        "composed", "smug", "cocky", "determined expression", "resolute"
    ],
    "shy": [
        "bashful", "timid", "coy", "blushing", "demure", "modest",
        "reserved", "diffident", "sheepish", "eyes averted", "nervous smile"
    ],
    "flirty": [
        "coquettish", "playful smile", "winking", "suggestive", "teasing",
        "seductive", "beguiling", "charming", "sultry", "alluring"
    ],
    "serious": [
        "stern", "grave", "solemn", "intense", "earnest", "severe",
        "grim", "no-nonsense", "deadpan", "focused", "determined"
    ],
    "neutral": [
        "expressionless", "stoic", "poker-faced", "impassive", "blank",
        "composed", "unreadable", "calm", "placid", "tranquil"
    ],
    "mischievous": [
        "impish", "playful", "naughty", "roguish", "devilish", "prankish",
        "scheming", "sly smile", "glint in the eye", "mischievous grin"
    ]
}

# Personality traits for character generation
PERSONALITY_TRAITS: Dict[str, List[str]] = {
    "positive": [
        "kind", "brave", "compassionate", "intelligent", "loyal", "honest", "optimistic",
        "creative", "determined", "reliable", "patient", "gentle", "playful", "charismatic",
        "generous", "curious", "humble", "wise", "adaptable", "empathetic", "enthusiastic",
        "friendly", "thoughtful", "respectful", "cooperative", "supportive", "sincere",
        "witty", "diligent", "disciplined", "adventurous", "energetic", "resourceful"
    ],
    "neutral": [
        "analytical", "serious", "reserved", "practical", "logical", "methodical", "composed",
        "independent", "introspective", "observant", "reflective", "cautious", "pragmatic",
        "rational", "stoic", "deliberate", "precise", "focused", "aloof", "quiet", "formal",
        "straightforward", "meticulous", "thorough", "traditional", "conventional", "private"
    ],
    "negative": [
        "stubborn", "impatient", "arrogant", "jealous", "selfish", "temperamental", "lazy",
        "manipulative", "cynical", "vengeful", "impulsive", "careless", "deceitful", "greedy",
        "insecure", "paranoid", "aggressive", "cold", "callous", "judgmental", "pessimistic",
        "vain", "hostile", "anxious", "compulsive", "obsessive", "controlling", "demanding"
    ],
    "complex": [
        "tragic", "conflicted", "mysterious", "unpredictable", "enigmatic", "contradictory",
        "misunderstood", "morally ambiguous", "tormented", "traumatized", "haunted", "intense",
        "passionate", "volatile", "sensitive", "perfectionist", "idealistic", "nostalgic",
        "jaded", "rebellious", "melancholic", "visionary", "calculating", "contemplative"
    ]
}

# Occupation categories for character backgrounds
OCCUPATION_CATEGORIES: Dict[str, List[str]] = {
    "academic": ["teacher", "professor", "researcher", "scientist", "scholar", "librarian", "historian", "archaeologist", "linguist", "astronomer"],
    "artistic": ["artist", "musician", "writer", "poet", "dancer", "actor", "singer", "designer", "illustrator", "photographer", "sculptor", "composer"],
    "business": ["businessperson", "entrepreneur", "CEO", "manager", "consultant", "accountant", "marketer", "salesperson", "financial advisor", "investor"],
    "magical": ["wizard", "mage", "sorcerer", "witch", "warlock", "alchemist", "enchanter", "spellcaster", "summoner", "diviner", "necromancer"],
    "medical": ["doctor", "nurse", "surgeon", "therapist", "veterinarian", "pharmacist", "paramedic", "psychiatrist", "dentist", "healer"],
    "military": ["soldier", "captain", "admiral", "general", "lieutenant", "commander", "pilot", "marine", "guard", "strategist", "knight"],
    "service": ["chef", "bartender", "waiter", "host", "receptionist", "flight attendant", "tour guide", "hotel staff", "personal assistant", "butler"],
    "technical": ["engineer", "programmer", "technician", "mechanic", "electrician", "architect", "designer", "craftsperson", "inventor", "hacker"],
    "adventuring": ["adventurer", "explorer", "treasure hunter", "archaeologist", "monster hunter", "bounty hunter", "mercenary", "ranger", "scout", "survivalist"],
    "criminal": ["thief", "assassin", "smuggler", "con artist", "fence", "forger", "hacker", "pickpocket", "burglar", "mob boss", "gang member"],
    "entertainment": ["performer", "musician", "actor", "dancer", "comedian", "magician", "celebrity", "model", "influencer", "streamer", "athlete"],
    "legal": ["lawyer", "judge", "detective", "investigator", "police officer", "sheriff", "bailiff", "prosecutor", "defender", "bounty hunter"],
    "spiritual": ["priest", "monk", "nun", "cleric", "shaman", "oracle", "prophet", "druid", "paladin", "mystic", "spiritual guide"],
    "political": ["politician", "diplomat", "ambassador", "senator", "mayor", "governor", "councillor", "monarch", "advisor", "revolutionary"],
    "outdoor": ["farmer", "fisher", "hunter", "forester", "ranger", "gardener", "park ranger", "zoologist", "botanist", "naturalist", "guide"],
    "transportation": ["pilot", "driver", "captain", "sailor", "astronaut", "train conductor", "navigator", "explorer", "courier", "delivery person"]
}

# Specific occupations for characters
OCCUPATIONS: List[str] = [
    # Academic/Scientific
    "professor", "researcher", "scientist", "historian", "archaeologist", "astronomer", "mathematician", "physicist", "biologist", "chemist", "linguist", "anthropologist", "teacher", "librarian", "archivist", "curator",
    
    # Art/Creative
    "artist", "painter", "sculptor", "musician", "composer", "writer", "poet", "novelist", "journalist", "photographer", "filmmaker", "dancer", "actor", "singer", "designer", "architect", "illustrator", "animator", "game developer",
    
    # Business/Professional
    "CEO", "business owner", "entrepreneur", "manager", "consultant", "accountant", "financial analyst", "marketer", "salesperson", "real estate agent", "insurance agent", "broker", "investor", "banker", "economist",
    
    # Medical
    "doctor", "surgeon", "nurse", "therapist", "psychiatrist", "psychologist", "veterinarian", "dentist", "pharmacist", "paramedic", "physical therapist", "nutritionist", "midwife", "healer",
    
    # Technical
    "engineer", "programmer", "software developer", "technician", "mechanic", "electrician", "plumber", "carpenter", "welder", "blacksmith", "jeweler", "tailor", "seamstress", "inventor", "hacker",
    
    # Service industry
    "chef", "cook", "baker", "bartender", "waiter", "waitress", "server", "barista", "host", "receptionist", "hotel manager", "flight attendant", "tour guide", "travel agent", "personal assistant", "maid", "butler", "cleaner",
    
    # Retail/Sales
    "shopkeeper", "store clerk", "cashier", "sales associate", "retail manager", "merchandiser", "vendor", "auctioneer", "antique dealer",
    
    # Law enforcement/Military
    "police officer", "detective", "investigator", "security guard", "soldier", "military officer", "general", "admiral", "pilot", "sailor", "marine", "captain", "lieutenant", "sergeant", "spy", "intelligence agent",
    
    # Legal
    "lawyer", "attorney", "judge", "prosecutor", "defender", "paralegal", "notary", "bailiff", "mediator", "legal consultant",
    
    # Political
    "politician", "diplomat", "ambassador", "senator", "representative", "mayor", "governor", "advisor", "chief of staff", "spokesperson", "activist", "revolutionary",
    
    # Fantasy/Magical
    "wizard", "witch", "sorcerer", "mage", "enchanter", "alchemist", "potion maker", "cleric", "paladin", "druid", "bard", "necromancer", "summoner", "battlemage", "spellsword", "court magician",
    
    # Religious/Spiritual
    "priest", "priestess", "monk", "nun", "bishop", "cardinal", "pope", "rabbi", "imam", "pastor", "minister", "chaplain", "shaman", "oracle", "prophet", "fortune teller", "medium", "mystic",
    
    # Adventuring/Exploration
    "adventurer", "explorer", "treasure hunter", "archaeologist", "monster hunter", "bounty hunter", "mercenary", "ranger", "scout", "survivalist", "guide", "cartographer", "sailor", "navigator",
    
    # Criminal/Underworld
    "thief", "burglar", "pickpocket", "assassin", "hitman", "smuggler", "pirate", "bandit", "outlaw", "con artist", "forger", "fence", "gang member", "mob boss", "crime lord",
    
    # Entertainment
    "entertainer", "performer", "celebrity", "model", "athlete", "gladiator", "racer", "gamer", "esports player", "streamer", "influencer", "comedian", "clown", "magician", "acrobat", "ringmaster",
    
    # Transportation
    "driver", "chauffeur", "taxi driver", "pilot", "captain", "sailor", "train conductor", "astronaut", "courier", "delivery person", "navigator", "helmsman",
    
    # Outdoor/Nature
    "farmer", "rancher", "fisher", "hunter", "forester", "ranger", "gardener", "landscaper", "botanist", "zoologist", "wildlife photographer", "park ranger", "naturalist", "environmentalist"
] 