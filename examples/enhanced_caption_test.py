#!/usr/bin/env python3
"""
Test enhanced natural caption generation for furry/anime content
"""

import sys
import os
from pprint import pprint

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cringegen.prompt_generation.nlp import (
    categorize_tags,
    tags_to_text,
    natural_tags_to_text
)

def test_enhanced_captions():
    """Test enhanced caption generation for furry/anime content"""
    print("\n===== TESTING ENHANCED CAPTION GENERATION FOR FURRY/ANIME CONTENT =====\n")
    
    # Test case 1: Basic anthro character
    test_case_1 = [
        "masterpiece", "best quality", "detailed", "anthro fox", 
        "red fur", "blue eyes", "wearing casual clothes", "smiling", 
        "forest", "sunset", "digital art"
    ]
    
    # Test case 2: Anime character
    test_case_2 = [
        "high quality", "detailed", "cat girl", "long pink hair",
        "green eyes", "school uniform", "cherry blossom tree",
        "spring", "soft lighting", "anime style"
    ]
    
    # Test case 3: Fantasy anthro
    test_case_3 = [
        "masterpiece", "highly detailed", "anthro wolf", "warrior",
        "gray fur", "battle armor", "sword", "medieval castle",
        "night", "moonlight", "dramatic lighting", "fantasy"
    ]
    
    # Test case 4: Feral animal
    test_case_4 = [
        "best quality", "detailed", "realistic", "tiger", 
        "orange fur", "black stripes", "jungle", "waterfall", 
        "morning", "mist", "photorealistic"
    ]
    
    # Test case 5: Mixed character types
    test_case_5 = [
        "high quality", "detailed", "human adventurer", "anthro dragon",
        "cyberpunk city", "neon lights", "night", "rain",
        "sci-fi", "digital painting"
    ]
    
    # Run all test cases
    test_sets = [
        ("Anthro Character", test_case_1),
        ("Anime Character", test_case_2),
        ("Fantasy Anthro", test_case_3),
        ("Feral Animal", test_case_4),
        ("Mixed Character Types", test_case_5)
    ]
    
    for name, tags in test_sets:
        print(f"\n--- Test: {name} ---")
        print(f"Tags: {', '.join(tags)}")
        
        print("\nCategorized tags:")
        categories = categorize_tags(tags)
        for category, items in categories.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")
        
        print("\nStandard tags_to_text output (descriptive style):")
        standard_text = tags_to_text(tags, style="descriptive")
        print(f"  \"{standard_text}\"")
        
        print("\nEnhanced natural caption:")
        natural_text = natural_tags_to_text(tags)
        print(f"  \"{natural_text}\"")
        
        print("\n" + "-"*50)
    
    # Additional test for complex prompts
    print("\n===== TESTING COMPLEX PROMPTS WITH NATURAL CAPTION GENERATION =====\n")
    
    complex_prompts = [
        [
            "masterpiece", "best quality", "ultra detailed", "anthro", "female fox",
            "red fur", "blue eyes", "fluffy tail", "yellow sundress", "flower crown",
            "sitting on grass", "meadow", "blooming flowers", "spring", "sunshine", 
            "soft lighting", "happy expression", "anime style"
        ],
        [
            "high quality", "detailed", "4k", "anime style", "neko boy", "white hair",
            "heterochromia", "blue and red eyes", "school uniform", "headphones",
            "classroom", "afternoon", "golden hour", "slice of life", "soft lighting"
        ],
        [
            "masterpiece", "best quality", "detailed", "anthro wolf", "warrior",
            "gray fur", "scar on face", "battle armor", "sword", "shield",
            "standing on cliff", "overlooking battlefield", "medieval castle",
            "stormy sky", "lightning", "dramatic lighting", "serious expression",
            "fantasy", "digital painting"
        ]
    ]
    
    for i, tags in enumerate(complex_prompts):
        print(f"\n--- Complex Test {i+1} ---")
        print(f"Tags: {', '.join(tags)}")
        
        print("\nStandard styles output:")
        print(f"  Concise: \"{tags_to_text(tags, style='concise')}\"")
        print(f"  Descriptive: \"{tags_to_text(tags, style='descriptive')}\"")
        print(f"  Detailed: \"{tags_to_text(tags, style='detailed')}\"")
        
        print("\nEnhanced natural caption:")
        natural_text = natural_tags_to_text(tags)
        print(f"  \"{natural_text}\"")
        
        print("\n" + "-"*70)

if __name__ == "__main__":
    test_enhanced_captions() 