"""
Poses command for cringegen CLI.

This module provides a command to generate poses for furry characters.
"""

import argparse
import logging
import random
from typing import Any, Dict, List, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Dictionary of poses by species taxonomy and form type
POSES = {
    "default": {
        "anthro": {
            "neutral": [
                "standing",
                "sitting",
                "walking",
                "running",
                "jumping",
                "leaning against wall",
                "crouching",
                "stretching",
                "looking back over shoulder",
                "arms crossed",
                "hands on hips",
            ],
            "intimate": [
                "laying down",
                "hugging",
                "nuzzling",
                "cuddling",
                "holding hands",
                "leaning on shoulder",
                "back to back",
            ],
            "action": [
                "fighting stance",
                "mid-punch",
                "casting spell",
                "drawing weapon",
                "aiming",
                "dodging",
                "blocking",
                "action pose",
                "battle ready",
            ],
        },
        "feral": {
            "neutral": [
                "standing",
                "sitting",
                "laying down",
                "walking",
                "running",
                "jumping",
                "pouncing",
                "stalking",
                "alert ears",
                "looking back",
                "stretching",
            ],
            "action": [
                "hunting stance",
                "mid-leap",
                "pouncing",
                "running at full speed",
                "sprinting",
                "bounding",
            ],
        },
    },
    "canine": {
        "anthro": {
            "specific": [
                "tail wagging",
                "ears perked",
                "playful stance",
                "nose twitching",
                "head tilted",
            ],
        },
        "feral": {
            "specific": [
                "play bow",
                "tail wagging",
                "ears perked up",
                "sniffing the ground",
                "howling",
                "panting",
            ],
        },
    },
    "feline": {
        "anthro": {
            "specific": [
                "tail swishing",
                "grooming paw",
                "ears flicking",
                "stretching claws",
                "crouched ready to pounce",
                "balancing on ledge",
            ],
        },
        "feral": {
            "specific": [
                "grooming",
                "tail twitching",
                "kneading",
                "arched back",
                "stalking low to ground",
                "pouncing position",
                "stretched out sunbathing",
            ],
        },
    },
    "avian": {
        "anthro": {
            "specific": [
                "wings spread",
                "wings folded",
                "preening feathers",
                "head cocked",
                "wings partially extended",
                "feathers ruffled",
            ],
        },
        "feral": {
            "specific": [
                "wings spread wide",
                "perched",
                "in flight",
                "landing",
                "taking off",
                "gliding",
                "preening",
                "wings folded",
            ],
        },
    },
    "reptile": {
        "anthro": {
            "specific": [
                "tail curled",
                "sunning on rock",
                "scales gleaming",
                "tongue flicking",
                "showing fangs",
                "slithering motion",
            ],
        },
        "feral": {
            "specific": [
                "basking",
                "slithering",
                "coiled",
                "swimming",
                "tail raised",
                "tongue flicking",
                "scales glistening",
            ],
        },
    },
    "equine": {
        "anthro": {
            "specific": [
                "tail swishing",
                "pawing ground",
                "ears forward",
                "mane flowing",
                "tossing head",
            ],
        },
        "feral": {
            "specific": [
                "galloping",
                "trotting",
                "rearing",
                "grazing",
                "pawing the ground",
                "ears flicking",
                "tail swishing",
            ],
        },
    },
    "rodent": {
        "anthro": {
            "specific": [
                "nibbling",
                "whiskers twitching",
                "nose wiggling",
                "ears perked",
                "cheeks full",
                "tail curled",
            ],
        },
        "feral": {
            "specific": [
                "foraging",
                "nibbling",
                "whiskers twitching",
                "standing on hind legs",
                "cheeks stuffed",
                "burrowing",
            ],
        },
    },
    "lagomorph": {  # Rabbits, hares
        "anthro": {
            "specific": [
                "ears perked up",
                "nose twitching",
                "legs ready to hop",
                "alert stance",
                "grooming ears",
                "nibbling",
            ],
        },
        "feral": {
            "specific": [
                "hopping",
                "nose twitching",
                "ears alert",
                "munching greens",
                "binky jump",
                "resting in loaf position",
            ],
        },
    },
    "cervid": {  # Deer, elk, moose
        "anthro": {
            "specific": [
                "antlers held high",
                "graceful stance",
                "alert and watchful",
                "ears rotating",
                "poised elegantly",
                "head tilted inquisitively",
            ],
        },
        "feral": {
            "specific": [
                "grazing",
                "alert with ears forward",
                "antlers displayed proudly",
                "bounding through forest",
                "frozen in alertness",
                "drinking from stream",
            ],
        },
    },
    "bovine": {  # Cows, bulls, buffalo
        "anthro": {
            "specific": [
                "steady stance",
                "tail swishing flies",
                "slow deliberate movement",
                "powerful posture",
                "head lowered",
                "horns displayed",
            ],
        },
        "feral": {
            "specific": [
                "grazing peacefully",
                "standing with tail swishing",
                "head lowered in charge",
                "laying in field",
                "horns held high",
                "muzzle raised",
            ],
        },
    },
    "dragon": {
        "anthro": {
            "specific": [
                "wings partially spread",
                "tail coiled around feet",
                "scales shimmering",
                "smoke from nostrils",
                "claws extended",
                "wings mantled dramatically",
            ],
        },
        "feral": {
            "specific": [
                "wings unfurled majestically",
                "coiled on treasure",
                "fire breath pose",
                "soaring flight",
                "tail lashing",
                "sleeping with one eye open",
            ],
        },
    },
    "mustelid": {  # Otters, ferrets, weasels
        "anthro": {
            "specific": [
                "playful stance",
                "sleek agile pose",
                "mischievous crouch",
                "rolling playfully",
                "curious tilt of head",
                "flexible body twist",
            ],
        },
        "feral": {
            "specific": [
                "sliding on belly",
                "swimming gracefully",
                "carrying object",
                "playing with stone",
                "weaving through obstacles",
                "diving into water",
            ],
        },
    },
    "marsupial": {  # Kangaroos, koalas
        "anthro": {
            "specific": [
                "balanced stance",
                "relaxed leaning",
                "alert upright posture",
                "joey checking pouch",
                "stretching limbs",
                "hopping motion",
            ],
        },
        "feral": {
            "specific": [
                "hopping",
                "standing tall on hind legs",
                "tail as third leg",
                "joey in pouch",
                "resting in tree fork",
                "alert ears forward",
            ],
        },
    },
    "aquatic": {  # Fish, sharks, dolphins
        "anthro": {
            "specific": [
                "fluid movements",
                "gills flaring",
                "fins extended gracefully",
                "scales glistening",
                "tail swishing",
                "swimming motion",
            ],
        },
        "feral": {
            "specific": [
                "swimming smoothly",
                "leaping from water",
                "floating in current",
                "circling pattern",
                "gliding through water",
                "tail propulsion",
            ],
        },
    },
    "ursine": {  # Bears
        "anthro": {
            "specific": [
                "powerful stance",
                "arms spread wide",
                "slow deliberate movement",
                "towering presence",
                "showing strength",
                "gentle gestures",
            ],
        },
        "feral": {
            "specific": [
                "standing on hind legs",
                "fishing in stream",
                "lumbering walk",
                "foraging",
                "claws extended",
                "playful rolling",
            ],
        },
    },
    "hyena": {
        "anthro": {
            "specific": [
                "hunched predatory stance",
                "laughing pose",
                "powerful shoulders forward",
                "mischievous look",
                "clan leader posture",
                "sly watching position",
            ],
        },
        "feral": {
            "specific": [
                "cackling stance",
                "stalking prey",
                "pack formation",
                "hunting crouch",
                "scavenging position",
                "powerful jaw display",
            ],
        },
    },
    "insectoid": {  # Insects, spiders
        "anthro": {
            "specific": [
                "multiple limbs in action",
                "antenna twitching",
                "mandibles clicking",
                "wings buzzing",
                "compound eyes alert",
                "unique joint articulation",
            ],
        },
        "feral": {
            "specific": [
                "wings spread",
                "spinning web",
                "antenna sensing",
                "multiple legs moving",
                "iridescent display",
                "molting position",
            ],
        },
    },
    "protogen": {
        "anthro": {
            "specific": [
                "visor displaying emotions",
                "mechanical joints posed",
                "LED patterns active",
                "suspension hover stance",
                "cyber interface engaged",
                "system booting pose",
            ],
        },
        "feral": {
            "specific": [
                "quadrupedal stance with tech elements",
                "visor scanning environment",
                "systems in standby mode",
                "digital display active",
                "mechanical limbs articulated",
                "drone-like hovering",
            ],
        },
    },
    "sergal": {
        "anthro": {
            "specific": [
                "unique head shape displayed",
                "rain-clan stance",
                "tribal posture",
                "predatory crouch",
                "northern kingdom salute",
                "warrior pose",
            ],
        },
        "feral": {
            "specific": [
                "wedge-head profile",
                "stalking low to ground",
                "fluffy tail alert",
                "tribal hunting stance",
                "pack formation",
                "sergal howl",
            ],
        },
    },
}


def add_poses_command(subparsers: Any, parent_parser: Any) -> None:
    """Add the poses command to the CLI."""
    parser = subparsers.add_parser(
        "poses",
        parents=[parent_parser],
        help="Generate poses for furry characters",
        description="Generate poses appropriate for specific furry species",
    )

    # Species argument
    parser.add_argument(
        "--species",
        type=str,
        help="Species of the furry character (e.g., wolf, fox, dragon)",
    )

    # Form type arguments (mutually exclusive)
    form_group = parser.add_mutually_exclusive_group()
    form_group.add_argument(
        "--anthro", 
        action="store_true", 
        help="Generate poses for anthropomorphic characters"
    )
    form_group.add_argument(
        "--feral", 
        action="store_true", 
        help="Generate poses for feral characters"
    )

    # Pose category argument
    parser.add_argument(
        "--category",
        type=str,
        choices=["neutral", "intimate", "action", "specific", "all"],
        default="all",
        help="Category of poses to generate (default: all)"
    )

    # Count argument
    parser.add_argument(
        "--count", 
        type=int, 
        default=1, 
        help="Number of poses to generate (default: 1)"
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
    parser.set_defaults(func=handle_poses_command)


def handle_poses_command(args: Any) -> None:
    """Handle the poses command."""
    # Check if species is provided
    if not args.species:
        logger.error("Species is required. Use --species to specify a species.")
        return

    # Determine form type (default to anthro if neither is specified)
    is_anthro = not args.feral if args.feral else True
    form_type = "anthro" if is_anthro else "feral"

    # Get the taxonomy group for the species
    from ..data.character_taxonomy import SPECIES_TAXONOMY
    taxonomy = SPECIES_TAXONOMY.get(args.species.lower(), "default")

    # Generate poses
    poses = generate_poses(
        taxonomy=taxonomy,
        form_type=form_type,
        category=args.category,
        count=args.count,
    )

    # Format and output the poses
    output_poses(poses, args.format, args.count)


def generate_poses(
    taxonomy: str,
    form_type: str, 
    category: str = "all", 
    count: int = 1
) -> List[str]:
    """Generate poses for a furry character.

    Args:
        taxonomy: The taxonomy group of the species
        form_type: The form type (anthro/feral)
        category: The category of poses to generate
        count: Number of poses to generate

    Returns:
        A list of poses
    """
    all_poses = []
    
    # Get default poses for the form type
    default_poses = POSES.get("default", {}).get(form_type, {})
    
    # Get species-specific poses
    species_poses = POSES.get(taxonomy, {}).get(form_type, {})
    
    # Combine poses based on category
    if category == "all":
        # Add all poses from default categories
        for cat, pose_list in default_poses.items():
            all_poses.extend(pose_list)
        
        # Add all poses from species-specific categories
        for cat, pose_list in species_poses.items():
            all_poses.extend(pose_list)
    else:
        # Add poses from specific category
        if category in default_poses:
            all_poses.extend(default_poses[category])
        
        # For "specific" category, check the species poses
        if category == "specific" and "specific" in species_poses:
            all_poses.extend(species_poses["specific"])
            
    # If no poses were found, use generic poses
    if not all_poses:
        logger.warning(f"No specific poses found for {taxonomy} ({form_type}). Using generic poses.")
        all_poses = [
            "standing",
            "sitting",
            "looking around",
            "walking",
            "running"
        ]
    
    # Select random poses
    if count >= len(all_poses):
        return all_poses
    else:
        return random.sample(all_poses, count)


def output_poses(poses: List[str], format_type: str, count: int) -> None:
    """Format and output the generated poses.

    Args:
        poses: List of poses to output
        format_type: The format to output (simple, detailed, prompt, csv)
        count: Number of poses requested
    """
    if not poses:
        print("No poses generated.")
        return
        
    if format_type == "simple":
        if count == 1:
            print(poses[0])
        else:
            for pose in poses:
                print(f"- {pose}")
    elif format_type == "detailed":
        if count == 1:
            print(f"Pose: {poses[0]}")
            print(f"Description: Character is {poses[0]}")
        else:
            for i, pose in enumerate(poses, 1):
                print(f"Pose {i}: {pose}")
                print(f"Description: Character is {pose}")
                if i < len(poses):
                    print()  # Add blank line between poses
    elif format_type == "csv":
        # Just output the comma-separated values with no other text
        print(",".join(poses))
    elif format_type == "prompt":
        # Format for direct inclusion in prompts
        if count == 1:
            print(f"in a {poses[0]} pose")
        else:
            pose_phrases = [f"in a {pose} pose" for pose in poses]
            print(", ".join(pose_phrases)) 