#!/usr/bin/env python3
"""
Script to fix invalid tags in cringegen modules

This script identifies tags in the cringegen data modules that don't exist in tags.json
and suggests replacements or removes them. It doesn't modify tags.json,
but instead updates the modules to use valid tags.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any

# Get the root directory of the project
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))

# Load valid tags from tags.json
tags_file = os.path.join(root_dir, "cringegen", "tags.json")
with open(tags_file, "r") as f:
    tags_data = json.load(f)
    valid_tags = set(tags_data.keys())

# Define replacements for invalid tags across different modules
tag_replacements = {
    # Anatomy module replacements
    "anatomy": {
        "alicorn": "anatomy-alicorn",
        "anthro": "anthro_general",
        "paws": "paw",
        "digitigrade": "digitigrade_legs",
        "plantigrade": "plantigrade_legs",
        "unguligrade": "unguligrade_legs",
        "horns": "horn",
        "back": "back_view",
        "front": "front_view",
        # New replacements from validation report
        "buck teeth": "teeth",
        "caprid": "bovid",
        "cervid": "deer",
        "chest fluff": "chest_tuft",
        "chiroptera": "yangochiropteran",
        "claspers": "claws",
        "common": "common_hippopotamus",
        "crest": "head_crest",
        "default": "general",
        "digitigrade legs": "digitigrade",
        "dutch angel dragon": "dragon",
        "ear fluff": "inner_ear_fluff",
        "feather patterns": "feather_tuft",
        "hyaenid": "canine",
        "long muzzle": "long_snout",
        "muzzle": "snout",
        "neck fluff": "neck_tuft",
        "ovid": "bovid",
        "paw pads": "paw",
        "pointed ears": "humanoid_pointy_ears",
        "primagen": "protogen",
        "pronounced incisors": "teeth",
        "retractable claws": "claws",
        "rounded ears": "humanoid_pointy_ears",
        "selachii": "shark",
        "synth": "synthetic",
        "tail tip": "tail",
        "testicles": "balls",
        "triangular ears": "humanoid_pointy_ears",
        "tufted ears": "tuft",
        "ursid": "ursid_humanoid",
        "vagina": "pussy",
    },
    # Accessories module replacements
    "accessories": {
        "blindfold": "blindfolded",
        "chastity": "chastity_cage",
        "glasses": "eyeglasses",
        "cuffs": "handcuffs",
        "leash": "on_leash",
        "sunglasses": "wearing_sunglasses",
        # New replacements from validation report
        "aquatic motifs": "aquatic_dragon",
        "chitin enhancers": "exoskeleton",
        "chitin polish": "exoskeleton",
        "ear accessories": "ear_piercing",
        "ranger tracker": "tracker_drone",
        "swimmer's cap": "visor_cap",
        "vent's goggles": "goggles",
    },
    # Colors module replacements
    "colors": {
        "orange": "orange_color",
        "gold": "gold_color",
        "silver": "silver_color",
        "purple": "purple_color",
        "rainbow": "rainbow_color",
        "chromatic": "chromatic_colors",
        "pink": "pink_color",
        "blue": "blue_color",
        "brown": "brown_color",
        "green": "green_color",
        "red": "red_color",
        "yellow": "yellow_color",
        "black": "black_color",
        "white": "white_color",
        "warm": "warm_colors",
        "cool": "cool_colors",
        "vibrant": "vibrant_colors",
        "pastel": "pastel_colors",
        "neon": "neon_colors",
        "hot_pink": "hot_pink_color",
        "dark_blue": "dark_blue_color",
        # New replacements from validation report
        "agouti": "brown_fur",
        "brindle": "stripes",
        "bronze": "gold_(metal)",
    },
    # Taxonomy module replacements
    "taxonomy": {
        "feline": "feline_species",
        "avian": "avian_species",
        "canine": "canine_species",
        "fox": "fox_species",
        "wolf": "wolf_species",
        "dragon": "dragon_species",
        "shark": "shark_species",
        "lion": "lion_species",
        "tiger": "tiger_species",
        "cat": "cat_species",
        "dog": "dog_species",
        "deer": "deer_species",
        "horse": "horse_species",
        "rabbit": "rabbit_species",
        "hybrid": "hybrid_species",
    },
}


def fix_tags_in_file(module_name, file_path):
    """
    Fix invalid tags in the specified file.

    Args:
        module_name: Name of the module ('anatomy', 'accessories', etc.)
        file_path: Path to the file to process

    Returns:
        Number of replacements made
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Use regex to find potential tags in the file
        # Look for strings in quotes that might be tags
        pattern = r'(?:\'|")([a-zA-Z0-9_\s\'-]+)(?:\'|")'
        matches = re.findall(pattern, content)

        replacements = 0

        # If this module has defined replacements
        if module_name in tag_replacements:
            for match in matches:
                # If the tag is in our replacement dictionary for this module
                # and it's not already a valid tag
                if match in tag_replacements[module_name] and match not in valid_tags:
                    replacement = tag_replacements[module_name][match]

                    # Verify the replacement is valid
                    if replacement in valid_tags:
                        # Replace the tag in the content
                        # Make sure we're only replacing the exact tag (with quotes)
                        content = re.sub(
                            f"(?:'|\"){re.escape(match)}(?:'|\")", f'"{replacement}"', content
                        )
                        replacements += 1
                        print(f"  Replaced '{match}' with '{replacement}'")

        # Direct replacements for specific cases not caught by the dictionary
        # These are based on context from the files

        # Handle 'default' in anatomy.py (when not talking about the dictionary key)
        if module_name == "anatomy":
            # Skip direct replacements for now as some tags might be dictionary keys
            pass

        # Handle accessories-specific replacements
        if module_name == "accessories":
            # Replace with valid tags
            content = re.sub(r'"ranger tracker"', '"tracker_drone"', content)
            content = re.sub(r'"swimmer\'s cap"', '"visor_cap"', content)
            content = re.sub(r'"chitin enhancers"', '"exoskeleton"', content)
            content = re.sub(r'"chitin polish"', '"exoskeleton"', content)

        # If we made any replacements, write the file back
        if replacements > 0 or content != open(file_path, "r").read():
            with open(file_path, "w") as f:
                f.write(content)
            if replacements == 0:
                replacements = 1  # At least one replacement was made via direct regex
                print("  Made direct regex replacements")

        return replacements

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


# Process each module
modules = ["anatomy", "accessories", "colors", "taxonomy"]

for module_name in modules:
    print(f"Processing {module_name} module...")
    # Use the correct path to the data files
    file_path = os.path.join(root_dir, "cringegen", "cringegen", "data", f"{module_name}.py")

    if os.path.exists(file_path):
        replacements = fix_tags_in_file(module_name, file_path)
        if replacements == 0:
            print(f"  No replacements needed in {module_name}")
        else:
            print(f"  Made {replacements} replacements in {module_name}")
    else:
        print(f"  File not found: {file_path}")

print("\nDone! Please run validate_tags.py to verify the changes.")
