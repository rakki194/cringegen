#!/usr/bin/env python3
"""
Comprehensive tests for tag categorization with furry Stable Diffusion prompts
"""

import sys
import os
import json
from pprint import pprint

# Add cringegen directory to the Python path
sys.path.append(os.path.join(os.getcwd(), "cringegen"))

try:
    from cringegen.prompt_generation.nlp import categorize_tags, tags_to_text, natural_tags_to_text
    print("Successfully imported tag categorization modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def print_categorization(tags, expected=None):
    """
    Print the categorization results for a set of tags
    
    Args:
        tags: List of tags to categorize
        expected: Optional expected categorization for validation
    """
    print(f"\n=== Testing Tags: {tags} ===")
    categories = categorize_tags(tags)
    
    # Print only non-empty categories for readability
    non_empty = {k: v for k, v in categories.items() if v}
    print(json.dumps(non_empty, indent=2))
    
    # Generate text representations
    descriptive = tags_to_text(tags, style="descriptive")
    natural = natural_tags_to_text(tags)
    
    print(f"\nDescriptive: {descriptive}")
    print(f"Natural: {natural}")
    
    # Check against expected results if provided
    if expected:
        for category, expected_tags in expected.items():
            actual_tags = set(categories.get(category, []))
            expected_tags_set = set(expected_tags)
            
            if actual_tags != expected_tags_set:
                print(f"\nWARNING: Mismatch in category '{category}'")
                print(f"Expected: {expected_tags}")
                print(f"Actual:   {categories.get(category, [])}")
    
    print("\n" + "-" * 80)

def test_basic_species():
    """Test basic furry species categorization"""
    print("\n===== TESTING BASIC SPECIES CATEGORIZATION =====\n")
    
    # Test common species
    print_categorization(
        ["fox"], 
        {"subject": ["fox"], "species": ["fox"]}
    )
    
    print_categorization(
        ["wolf"], 
        {"subject": ["wolf"], "species": ["wolf"]}
    )
    
    print_categorization(
        ["dragon"], 
        {"subject": ["dragon"], "species": ["dragon"]}
    )
    
    print_categorization(
        ["sergal"],
        {"subject": ["sergal"], "species": ["sergal"]}
    )
    
    # Test with quality tags
    print_categorization(
        ["masterpiece", "detailed", "fox"]
    )

def test_anthro_species():
    """Test anthro species categorization"""
    print("\n===== TESTING ANTHRO SPECIES CATEGORIZATION =====\n")
    
    # Test with anthro modifier
    print_categorization(
        ["anthro fox"],
    )
    
    print_categorization(
        ["anthropomorphic wolf"],
    )
    
    # Test with multiple species
    print_categorization(
        ["fox", "wolf hybrid"],
    )
    
    # Test with species and gender
    print_categorization(
        ["female anthro dragon"],
    )
    
    # Test with complex species description
    print_categorization(
        ["muscular anthro tiger", "tough", "battle scarred"],
    )

def test_color_and_markings():
    """Test color and marking patterns"""
    print("\n===== TESTING COLOR AND MARKING PATTERNS =====\n")
    
    # Basic color
    print_categorization(
        ["red fox"],
    )
    
    # Multiple colors
    print_categorization(
        ["blue and white wolf"],
    )
    
    # Color with fur specified
    print_categorization(
        ["red fur", "white markings", "fox"],
    )
    
    # Complex color pattern
    print_categorization(
        ["orange and black tiger", "stripes", "green eyes"],
    )
    
    # Color with species-specific pattern
    print_categorization(
        ["spotted leopard", "yellow fur", "black spots"],
    )

def test_clothing_and_accessories():
    """Test clothing and accessories categorization"""
    print("\n===== TESTING CLOTHING AND ACCESSORIES =====\n")
    
    # Basic clothing
    print_categorization(
        ["anthro fox", "wearing shirt"],
    )
    
    # Complex outfit
    print_categorization(
        ["anthro wolf", "leather jacket", "ripped jeans", "sunglasses"],
    )
    
    # Accessories
    print_categorization(
        ["fox", "sword", "backpack", "armor"],
    )
    
    # Clothing with descriptors
    print_categorization(
        ["anthro cat", "elegant dress", "jewelry", "formal attire"],
    )
    
    # Fantasy/sci-fi gear
    print_categorization(
        ["anthro dragon", "battle armor", "glowing runes", "magical staff"],
    )

def test_art_styles():
    """Test art style categorization"""
    print("\n===== TESTING ART STYLE CATEGORIZATION =====\n")
    
    # Digital art
    print_categorization(
        ["anthro fox", "digital art"],
        {"style": ["digital art"]}
    )
    
    # Specific art styles
    print_categorization(
        ["wolf", "anime style"],
    )
    
    print_categorization(
        ["sergal", "oil painting"],
    )
    
    # Multiple styles
    print_categorization(
        ["fox", "watercolor", "sketch", "concept art"],
    )
    
    # Style with descriptors
    print_categorization(
        ["dragon", "vibrant colors", "cel shaded", "comic book style"],
    )

def test_quality_tags():
    """Test quality tags categorization"""
    print("\n===== TESTING QUALITY TAGS CATEGORIZATION =====\n")
    
    # Common quality tags
    print_categorization(
        ["masterpiece", "best quality", "highly detailed", "fox"],
    )
    
    # Resolution descriptors
    print_categorization(
        ["hires", "4k", "ultra detailed", "wolf"],
    )
    
    # Mixed quality descriptors
    print_categorization(
        ["masterpiece", "intricate details", "professional", "anthro lion"],
    )
    
    # Negative quality tags (should still be categorized as quality)
    print_categorization(
        ["wolf", "low quality", "blurry", "amateur"],
    )

def test_settings_and_backgrounds():
    """Test settings and background categorization"""
    print("\n===== TESTING SETTINGS AND BACKGROUNDS =====\n")
    
    # Basic setting
    print_categorization(
        ["fox", "forest"],
    )
    
    # Complex environment
    print_categorization(
        ["wolf", "mountain", "snowy landscape", "night sky"],
    )
    
    # Urban setting
    print_categorization(
        ["anthro raccoon", "city streets", "urban", "neon lights"],
    )
    
    # Fantasy environment
    print_categorization(
        ["dragon", "castle", "medieval", "fantasy landscape"],
    )
    
    # Setting with atmospheric elements
    print_categorization(
        ["anthro wolf", "foggy forest", "moonlight", "misty atmosphere"],
    )

def test_complex_prompts():
    """Test complex multi-tag prompts"""
    print("\n===== TESTING COMPLEX MULTI-TAG PROMPTS =====\n")
    
    # Complex prompt 1 - Portrait
    print_categorization([
        "masterpiece", "high quality", "detailed", 
        "anthro female fox", "red fur", "white underbelly", 
        "blue eyes", "short hair", "tank top", "jeans", 
        "happy expression", "forest background", "sunlight",
        "digital art", "concept art"
    ])
    
    # Complex prompt 2 - Action scene
    print_categorization([
        "masterpiece", "best quality", "highly detailed",
        "anthro male wolf", "muscular", "gray fur", "battle scars",
        "armor", "sword", "fighting stance", 
        "castle ruins", "rainy", "night", "dramatic lighting",
        "oil painting", "fantasy", "epic scene"
    ])
    
    # Complex prompt 3 - Multiple characters
    print_categorization([
        "masterpiece", "detailed", "professional",
        "anthro dragon and fox", "friends", "walking together",
        "blue scales", "orange fur", "casual clothing",
        "city street", "sunset", "warm colors",
        "digital painting", "slice of life"
    ])
    
    # Complex prompt 4 - Sci-fi setting
    print_categorization([
        "high quality", "detailed", "4k",
        "anthro sergal", "cyborg", "white and blue fur", 
        "robotic arm", "glowing eyes", "tech outfit",
        "futuristic city", "holographic displays", "neon lights",
        "cyberpunk", "sci-fi", "digital art"
    ])
    
    # Complex prompt 5 - Fantasy setting
    print_categorization([
        "masterpiece", "best quality", "detailed",
        "anthro rabbit", "mage", "brown fur", "long ears",
        "magical robes", "glowing staff", "spell casting",
        "ancient library", "floating books", "magic particles",
        "fantasy illustration", "mystical atmosphere"
    ])

def main():
    """Main function to run all tests"""
    print("=== COMPREHENSIVE TAG CATEGORIZATION TESTS ===")
    print("Testing furry prompt tag categorization for Stable Diffusion\n")
    
    # Run all test categories
    test_basic_species()
    test_anthro_species()
    test_color_and_markings()
    test_clothing_and_accessories()
    test_art_styles()
    test_quality_tags()
    test_settings_and_backgrounds()
    test_complex_prompts()

if __name__ == "__main__":
    main() 