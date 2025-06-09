#!/usr/bin/env python3
"""
Test script to demonstrate color and multi-subject features.
"""

import random

from cringegen.prompt_generation.generators.furry_generator import FurryPromptGenerator
from cringegen.prompt_generation.nlp.color_utils import (
    parse_color_input,
    generate_color_description,
    get_body_covering_type,
)


def test_single_character() -> None:
    """Test single character with color specifications."""
    print("\n=== Testing Single Character Generator ===\n")
    
    # Basic character with single color
    generator1 = FurryPromptGenerator(
        species="wolf",
        gender="male", 
        colors="blue",
        use_nlp=True
    )
    prompt1 = generator1.generate()
    
    print("Wolf with blue fur:")
    print(f"Prompt: {prompt1}")
    print()
    
    # Character with multiple colors and pattern
    generator2 = FurryPromptGenerator(
        species="tiger",
        gender="female",
        colors="orange,black,white",
        pattern="striped",
        use_nlp=True
    )
    prompt2 = generator2.generate()
    
    print("Tiger with orange, black, and white striped fur:")
    print(f"Prompt: {prompt2}")
    print()
    
    # Dragon with scales
    generator3 = FurryPromptGenerator(
        species="dragon",
        gender="male",
        colors="red,gold",
        pattern="iridescent",
        use_nlp=True
    )
    prompt3 = generator3.generate()
    
    print("Dragon with red and gold scales:")
    print(f"Prompt: {prompt3}")
    print()


def test_duo_characters() -> None:
    """Test duo characters with color specifications."""
    print("\n=== Testing Duo Character Generator ===\n")
    
    # Wolf and Fox duo
    generator1 = FurryPromptGenerator(
        species="wolf",
        gender="male", 
        colors="gray,silver",
        pattern="solid",
        species2="fox",
        gender2="female",
        colors2="red,white",
        pattern2="bicolor",
        use_duo=True,
        use_nlp=True
    )
    prompt1 = generator1.generate()
    
    print("Wolf and Fox duo:")
    print(f"Prompt: {prompt1}")
    print()
    
    # Cat and Rabbit duo
    generator2 = FurryPromptGenerator(
        species="cat",
        gender="female",
        colors="black,white",
        pattern="tuxedo",
        species2="rabbit",
        gender2="male",
        colors2="brown,cream",
        use_duo=True,
        use_nlp=True
    )
    prompt2 = generator2.generate()
    
    print("Cat and Rabbit duo:")
    print(f"Prompt: {prompt2}")
    print()


def test_group_characters() -> None:
    """Test group characters with color specifications."""
    print("\n=== Testing Group Character Generator ===\n")
    
    # Four-character group
    generator = FurryPromptGenerator(
        species="wolf",
        gender="male", 
        colors="gray,white",
        species2="fox",
        gender2="female",
        colors2="red,orange",
        group_size=4,
        use_nlp=True
    )
    prompt = generator.generate()
    
    print("Group of 4 characters (with wolf and fox specified):")
    print(f"Prompt: {prompt}")
    print()


def test_color_utils() -> None:
    """Test the color utility functions directly."""
    print("\n=== Testing Color Utilities ===\n")
    
    # Test color parsing
    color_input = "red, blue, green, yellow"
    colors = parse_color_input(color_input)
    print(f"Parsed colors from '{color_input}': {colors}")
    
    # Test body covering type detection
    species_list = ["wolf", "dragon", "bird", "fish", "deer"]
    for species in species_list:
        covering = get_body_covering_type(species)
        print(f"Body covering for {species}: {covering}")
    
    # Test color description generation
    for species in species_list:
        # Monotone
        desc1 = generate_color_description(species, colors=["blue"])
        print(f"{species} (blue): {desc1}")
        
        # Bicolor
        desc2 = generate_color_description(species, colors=["red", "black"])
        print(f"{species} (red+black): {desc2}")
        
        # Tricolor
        desc3 = generate_color_description(species, colors=["gold", "silver", "bronze"])
        print(f"{species} (gold+silver+bronze): {desc3}")
        
        print()


def main() -> None:
    """Run all tests."""
    # Set random seed for reproducibility
    random.seed(42)
    
    print("\nTESTING COLOR AND MULTI-SUBJECT FEATURES")
    print("=======================================\n")
    
    # Test single character generation
    test_single_character()
    
    # Test duo character generation
    test_duo_characters()
    
    # Test group character generation
    test_group_characters()
    
    # Test color utilities directly
    test_color_utils()


if __name__ == "__main__":
    main() 