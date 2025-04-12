#!/usr/bin/env python3
"""
Test script for the enhance_prompt_with_species_tags function.
This script tests the tag enhancement functionality without contacting any servers.
"""

import sys
import os
from pathlib import Path

# Add the cringegen directory to the path so we can import from it
REPO_ROOT = Path("/home/kade/code/cringe.live")
CRINGEGEN_DIR = REPO_ROOT / "cringegen"
sys.path.append(str(CRINGEGEN_DIR))

# Import the necessary modules
try:
    from cringegen.commands.llm_noobai_nsfw import (
        enhance_prompt_with_species_tags,
        get_explicit_level,
    )
    from cringegen.utils.ollama_api import TAGS_PROCESSOR_AVAILABLE

    print(f"Tags processor available: {TAGS_PROCESSOR_AVAILABLE}")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_tag_enhancement():
    """Test the enhance_prompt_with_species_tags function with various combinations."""
    print("\n=== Testing Tag Enhancement Functionality ===\n")

    # Test base prompt
    base_prompt = "anthro, muscular, standing, dimly lit room, red eyes"

    # Test various species/gender combinations
    test_cases = [
        {"species": "fox", "gender": "male", "intensity": "explicit"},
        {"species": "wolf", "gender": "male", "intensity": "hardcore"},
        {"species": "cat", "gender": "male", "intensity": "explicit"},
        {"species": "dragon", "gender": "male", "intensity": "hardcore"},
        {"species": "horse", "gender": "male", "intensity": "explicit"},
        {"species": "fox", "gender": "female", "intensity": "explicit"},
        {"species": "wolf", "gender": "female", "intensity": "hardcore"},
    ]

    # Test with prompts that already contain anatomical terms
    anatomical_prompts = [
        "anthro, fox, male, penis, standing tall",
        "anthro, wolf, female, pussy, laying on bed",
        "anthro, dragon, male, hemipenes, spread wings",
    ]

    # Run the tests for species/gender combinations
    for idx, case in enumerate(test_cases):
        species = case["species"]
        gender = case["gender"]
        intensity = case["intensity"]
        explicit_level = get_explicit_level(intensity)

        print(
            f"\nTest Case {idx+1}: {gender} {species} ({intensity}, level {explicit_level})"
        )
        print(f"Base prompt: {base_prompt}")

        # Enhance the prompt
        enhanced = enhance_prompt_with_species_tags(
            base_prompt, species, gender, explicit_level=explicit_level
        )

        print(f"Enhanced prompt: {enhanced}")
        print(f"Tags added: {enhanced.replace(base_prompt + ', ', '')}")

    # Test prompts that already have anatomical terms
    print("\n\n=== Testing Prompts With Existing Anatomical Terms ===\n")
    for idx, prompt in enumerate(anatomical_prompts):
        print(f"\nExisting Anatomical Prompt {idx+1}: {prompt}")
        species = "fox" if "fox" in prompt else "wolf" if "wolf" in prompt else "dragon"
        gender = "male" if "male" in prompt else "female"

        # Enhance the prompt
        enhanced = enhance_prompt_with_species_tags(
            prompt, species, gender, explicit_level=3
        )

        print(f"After enhancement: {enhanced}")
        print(f"Changed: {'No' if prompt == enhanced else 'Yes'}")


if __name__ == "__main__":
    test_tag_enhancement()
