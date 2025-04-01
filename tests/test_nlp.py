#!/usr/bin/env python3
"""
Test script for cringegen NLP capabilities
"""

import sys
import os

# Add cringegen directory to the Python path
sys.path.append(os.path.join(os.getcwd(), "cringegen"))

try:
    # Import from the cringegen module
    from cringegen.prompt_generation.nlp import (
        # Tag conversion
        tags_to_text, 
        text_to_tags, 
        categorize_tags, 
        natural_tags_to_text,
        
        # Entity extraction
        extract_entities,
        
        # Prompt analysis
        analyze_prompt_structure,
        
        # Text analysis
        extract_keywords,
        
        # Enhancement
        suggest_improvements,
        enhance_prompt_general,
        
        # Color and species utilities
        generate_color_description,
        get_species_colors,
        generate_species_description
    )
    print("Successfully imported all NLP modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_tag_conversion():
    """Test tag to text and text to tag conversion"""
    print("\n=== Testing Tag Conversion ===")
    
    # Test tags to text
    tags = ["masterpiece", "detailed", "fox", "red fur", "forest", "digital art"]
    print(f"Original tags: {tags}")
    
    descriptive = tags_to_text(tags, style="descriptive")
    print(f"Descriptive: {descriptive}")
    
    natural = natural_tags_to_text(tags)
    print(f"Natural text: {natural}")
    
    # Test text to tags
    text = "A high quality digital painting of a fox with blue eyes in a forest"
    extracted_tags = text_to_tags(text)
    print(f"\nOriginal text: {text}")
    print(f"Extracted tags: {extracted_tags}")
    
    # Test tag categorization
    categories = categorize_tags(tags)
    print(f"\nCategorized tags: {categories}")

def test_entity_extraction():
    """Test entity extraction"""
    print("\n=== Testing Entity Extraction ===")
    
    text = "An anthro fox warrior with a sword battling alongside human allies in a fantasy castle setting."
    print(f"Text: {text}")
    
    entities = extract_entities(text)
    print(f"Extracted entities: {entities}")

def test_prompt_analysis():
    """Test prompt analysis"""
    print("\n=== Testing Prompt Analysis ===")
    
    prompt = "masterpiece, detailed, fox with red fur in a forest, digital art"
    print(f"Prompt: {prompt}")
    
    analysis = analyze_prompt_structure(prompt)
    print(f"Word count: {analysis.word_count}")
    print(f"Complexity score: {analysis.complexity_score}/15")
    print(f"Issues detected: {len(analysis.detected_issues)}")
    if analysis.detected_issues:
        print(f"Issues: {analysis.detected_issues}")
    
    # Extract keywords
    keywords = extract_keywords(prompt)
    print(f"\nExtracted keywords: {keywords}")
    
    # Suggest improvements
    improvements = suggest_improvements(prompt)
    print(f"\nSuggested improvements: {improvements}")

def test_prompt_enhancement():
    """Test prompt enhancement"""
    print("\n=== Testing Prompt Enhancement ===")
    
    prompt = "fox in forest"
    print(f"Original prompt: {prompt}")
    
    enhanced = enhance_prompt_general(prompt)
    print(f"Enhanced prompt: {enhanced}")

def test_species_and_color():
    """Test species and color utilities"""
    print("\n=== Testing Species and Color Utilities ===")
    
    # Test color description
    colors = ["red", "white"]
    color_desc = generate_color_description("fox", colors)
    print(f"Color description for a {colors} fox: {color_desc}")
    
    # Test species colors
    species_colors = get_species_colors("wolf")
    print(f"\nCommon colors for wolf: {species_colors[:5] if species_colors else 'None'}")
    
    # Test species description
    species_desc = generate_species_description("dragon", ["blue", "silver"], include_anatomy=True)
    print(f"\nDescription for blue and silver dragon: {species_desc}")

def main():
    """Main function to run all tests"""
    print("Testing cringegen NLP capabilities...\n")
    
    try:
        test_tag_conversion()
        test_entity_extraction()
        test_prompt_analysis()
        test_prompt_enhancement()
        test_species_and_color()
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    main() 