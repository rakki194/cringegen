#!/usr/bin/env python3
"""
Comprehensive test for all NLP features in CringeGen.
This script demonstrates and tests:
1. Tag/Text conversion (including natural caption generation)
2. Entity detection (with furry/anime specialization)
3. Prompt analysis and enhancement
4. NLTK integration
"""

import sys
import os
from pprint import pprint

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cringegen.prompt_generation.nlp import (
    # Tag converter utilities
    tags_to_text, text_to_tags, categorize_tags, natural_tags_to_text,
    # Prompt analyzer utilities
    analyze_prompt_structure, get_prompt_structure, 
    compare_prompts, suggest_improvements, simplify_prompt_structure,
    extract_keywords, detect_prompt_style,
    # NLTK utilities
    analyze_text_pos, extract_phrases, get_synonyms, get_antonyms,
    get_hypernyms, get_hyponyms, analyze_sentiment, extract_keywords_with_pos,
    lemmatize_text, compute_text_similarity, extract_entities, generate_ngrams,
    enhance_prompt_general, enhance_prompt_with_anatomy, enhance_prompt_with_background
)

def separator(title):
    """Print a separator with title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")

def test_tag_text_conversion():
    """Test tag to text conversion including natural captions"""
    separator("TAG-TEXT CONVERSION TEST")
    
    # Test sets with various content types
    test_sets = [
        {
            "name": "Basic Furry",
            "tags": ["masterpiece", "detailed", "anthro fox", "red fur", "blue eyes", 
                    "wearing casual clothes", "forest", "sunset", "digital art"]
        },
        {
            "name": "Anime Character",
            "tags": ["best quality", "detailed", "anime style", "cat girl", "pink hair", 
                    "school uniform", "cherry blossom", "spring"]
        },
        {
            "name": "Fantasy Scene",
            "tags": ["high quality", "detailed", "dragon", "knight", "castle", 
                    "battle", "fire", "dramatic lighting", "fantasy", "epic"]
        },
        {
            "name": "Landscape",
            "tags": ["masterpiece", "photorealistic", "mountain", "lake", "forest", 
                    "sunrise", "mist", "reflections", "dramatic sky"]
        }
    ]
    
    for test_set in test_sets:
        print(f"\n--- {test_set['name']} ---")
        tags = test_set['tags']
        print(f"Tags: {', '.join(tags)}\n")
        
        # Test categorization
        categories = categorize_tags(tags)
        print("Categorized Tags:")
        for category, items in categories.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")
        
        # Test text generation
        print("\nText Conversion:")
        for style in ["concise", "descriptive", "detailed"]:
            result = tags_to_text(tags, style=style)
            print(f"  {style.capitalize()}: \"{result}\"")
        
        # Test natural caption
        print("\nNatural Caption:")
        natural_caption = natural_tags_to_text(tags)
        print(f"  \"{natural_caption}\"")
        
        # Test bidirectional conversion
        print("\nBidirectional Conversion Test:")
        reconverted_tags = text_to_tags(natural_caption)
        print(f"  Original tags: {len(tags)} tags")
        print(f"  Reconverted tags: {len(reconverted_tags)} tags")
        print(f"  Recovered tags: {', '.join(reconverted_tags)}")
        
        # Calculate overlap
        original_set = set([tag.lower() for tag in tags])
        reconverted_set = set([tag.lower() for tag in reconverted_tags])
        overlap = original_set.intersection(reconverted_set)
        
        print(f"  Overlap percentage: {len(overlap) / len(original_set) * 100:.1f}%")

def test_entity_detection():
    """Test entity detection with various content types"""
    separator("ENTITY DETECTION TEST")
    
    test_texts = [
        "An anthro fox warrior with a blue sword battling alongside human allies in a fantasy castle.",
        "A cute anime-style neko girl with pink hair wearing a school uniform under cherry blossoms.",
        "A digital painting of a feral wolf howling at the moon in a snowy forest.",
        "A cyberpunk cityscape with robot citizens, anthro raccoons, and human hackers at night.",
        "An oil painting of a medieval fantasy battle between elves, dwarves, and orcs near a dragon's lair."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nText #{i+1}:")
        print(f"\"{text}\"\n")
        
        entities = extract_entities(text)
        print("Detected Entities:")
        for category, items in entities.items():
            if items:
                print(f"  {category.capitalize()}: {', '.join(items)}")

def test_prompt_analysis():
    """Test prompt analysis and enhancement capabilities"""
    separator("PROMPT ANALYSIS TEST")
    
    prompts = [
        "masterpiece, best quality, anthro fox, red fur, blue eyes, forest, digital art",
        "high quality, anime style, cat girl, school uniform, classroom, slice of life",
        "a photorealistic painting of a mountain landscape with a lake and forest at sunrise with dramatic lighting and clouds",
        "cyberpunk cityscape, night, neon lights, rain, futuristic technology, detailed"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt #{i+1}:")
        print(f"\"{prompt}\"\n")
        
        # Analyze structure
        print("Structure Analysis:")
        analysis = analyze_prompt_structure(prompt)
        print(f"  Word count: {analysis.word_count}")
        print(f"  Complexity score: {analysis.complexity_score}/15")
        print(f"  Section presence: {', '.join([k for k, v in analysis.section_presence.items() if v])}")
        
        # Extract keywords
        print("\nKeywords:")
        keywords = extract_keywords(prompt, top_n=5)
        print(f"  {', '.join(keywords)}")
        
        # Suggest improvements
        print("\nSuggested Improvements:")
        suggestions = suggest_improvements(prompt)
        for suggestion in suggestions[:3]:  # Limit to 3 suggestions
            print(f"  - {suggestion['description']}")
        
        # Simplify prompt
        print("\nSimplified Version:")
        simplified = simplify_prompt_structure(prompt, target_length=6)
        print(f"  \"{simplified}\"")
        
        # Detect style
        style = detect_prompt_style(prompt)
        print(f"\nDetected Style: {style}")

def test_prompt_comparison():
    """Test prompt comparison functionality"""
    separator("PROMPT COMPARISON TEST")
    
    prompt_pairs = [
        {
            "name": "Similar Animal Prompts",
            "prompt1": "masterpiece, detailed, anthro fox, red fur, blue eyes, forest, sunset, digital art",
            "prompt2": "high quality, detailed, anthro wolf, gray fur, yellow eyes, forest, night, oil painting"
        },
        {
            "name": "Different Content Types",
            "prompt1": "anime style cat girl, pink hair, school uniform, classroom, slice of life",
            "prompt2": "cyberpunk cityscape, night, neon lights, rain, futuristic, detailed"
        }
    ]
    
    for pair in prompt_pairs:
        print(f"\n--- {pair['name']} ---")
        print(f"Prompt 1: \"{pair['prompt1']}\"")
        print(f"Prompt 2: \"{pair['prompt2']}\"\n")
        
        comparison = compare_prompts(pair['prompt1'], pair['prompt2'])
        
        print("Comparison Results:")
        print(f"  Unique to prompt 1: {', '.join(comparison['unique_to_prompt1'][:5])}...")
        print(f"  Unique to prompt 2: {', '.join(comparison['unique_to_prompt2'][:5])}...")
        print(f"  Common elements: {', '.join(comparison['common_words'])}")
        
        similarity = compute_text_similarity(pair['prompt1'], pair['prompt2'])
        print(f"\nSimilarity Score: {similarity:.2f} (0-1 scale)")

def test_nltk_features():
    """Test NLTK integration features"""
    separator("NLTK FEATURES TEST")
    
    texts = [
        "A masterfully detailed digital painting of an anthropomorphic red fox with a fluffy tail, wearing a blue hoodie.",
        "An anime style cat girl with long pink hair and green eyes, wearing a school uniform, in a cherry blossom scene."
    ]
    
    for i, text in enumerate(texts):
        print(f"\nText #{i+1}:")
        print(f"\"{text}\"\n")
        
        # Part-of-speech analysis
        pos_analysis = analyze_text_pos(text)
        print("Part-of-Speech Analysis:")
        for pos, words in pos_analysis.items():
            if words:
                print(f"  {pos.capitalize()}: {', '.join(words[:5])}{'...' if len(words) > 5 else ''}")
        
        # Extract phrases
        print("\nNoun Phrases:")
        noun_phrases = extract_phrases(text, phrase_type="NP")
        for phrase in noun_phrases[:5]:
            print(f"  - {phrase}")
        
        # Synonyms and semantic relationships
        print("\nSemantic Relationships:")
        for word in ["detailed", "fox", "anime"]:
            if word in text.lower():
                print(f"  Synonyms for '{word}': {', '.join(get_synonyms(word)[:5])}")
        
        # Sentiment analysis
        sentiment = analyze_sentiment(text)
        print("\nSentiment Analysis:")
        print(f"  Positive: {sentiment['pos']:.2f}")
        print(f"  Negative: {sentiment['neg']:.2f}")
        print(f"  Neutral: {sentiment['neu']:.2f}")
        print(f"  Compound: {sentiment['compound']:.2f}")

def test_text_generation():
    """Test text generation capabilities"""
    separator("TEXT GENERATION TEST")
    
    # Test text generation with different styles and characters
    test_cases = [
        {
            "name": "Furry Character",
            "species": "fox",
            "gender": "male",
            "attributes": ["adventurous", "clever", "friendly"],
            "setting": "forest",
            "clothing": "explorer outfit",
            "style": "digital art"
        },
        {
            "name": "Anthro Character",
            "species": "wolf",
            "gender": "female",
            "attributes": ["fierce", "protective", "elegant"],
            "setting": "mountains",
            "clothing": "warrior armor",
            "style": "fantasy illustration"
        },
        {
            "name": "Anime Character",
            "species": "cat girl",
            "gender": "female",
            "attributes": ["shy", "cute", "innocent"],
            "setting": "school",
            "clothing": "school uniform",
            "style": "anime style"
        },
        {
            "name": "Fantasy Character",
            "species": "dragon",
            "gender": "male",
            "attributes": ["ancient", "wise", "powerful"],
            "setting": "castle ruins",
            "clothing": "None",
            "style": "epic fantasy"
        }
    ]
    
    from cringegen.prompt_generation.generators.furry_generator import FurryPromptGenerator
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        
        # Create a generator for the specific character type
        generator = FurryPromptGenerator(
            species=case["species"],
            gender=case["gender"],
            use_nlp=True,
            use_art_style=True
        )
        
        try:
            # Set up specific components based on test case
            from cringegen.prompt_generation.base import SettingComponent, StyleComponent
            
            # Override the setting
            setting = SettingComponent(
                location=case["setting"],
                use_nlp_description=True
            )
            
            # Override the style
            style = StyleComponent(
                art_style=case["style"]
            )
            
            # Generate the prompt
            prompt = generator.generate()
            
            # Convert to tags for testing
            tags = text_to_tags(prompt)
            print(f"Generated Tags ({len(tags)}):")
            print(", ".join(tags[:10]) + ("..." if len(tags) > 10 else ""))
            
            # Convert tags to different text formats
            print("\nGenerated Descriptions:")
            for style_name in ["concise", "descriptive", "detailed"]:
                text = tags_to_text(tags, style=style_name)
                print(f"  {style_name.capitalize()}: \"{text[:100]}...\"")
            
            # Generate natural language caption
            natural_text = natural_tags_to_text(tags)
            print(f"\nNatural Caption:\n\"{natural_text}\"")
            
            # Original prompt
            print(f"\nOriginal Generated Prompt:\n\"{prompt}\"")
            
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            continue

def test_integrated_workflow():
    """Test an integrated workflow using multiple NLP components together"""
    separator("INTEGRATED NLP WORKFLOW TEST")
    
    print("This test demonstrates how multiple NLP components can be used together in a realistic workflow.")
    print("It simulates a user creating and refining a prompt through several stages.\n")
    
    # Stage 1: Start with a basic prompt
    print("Stage 1: Starting with a basic prompt")
    base_prompt = "anthro wolf, forest, digital art"
    print(f"Initial prompt: \"{base_prompt}\"")
    
    # Stage 2: Analyze the prompt structure and identify areas for improvement
    print("\nStage 2: Analyzing prompt structure")
    analysis = analyze_prompt_structure(base_prompt)
    print(f"Complexity score: {analysis.complexity_score}/15")
    print(f"Word count: {analysis.word_count}")
    print(f"Sections present: {', '.join([k for k, v in analysis.section_presence.items() if v])}")
    print(f"Missing sections: {', '.join([k for k, v in analysis.section_presence.items() if not v])}")
    
    # Stage 3: Get improvement suggestions
    print("\nStage 3: Getting improvement suggestions")
    suggestions = suggest_improvements(base_prompt)
    print("Suggestions:")
    for i, suggestion in enumerate(suggestions[:5]):
        print(f"  {i+1}. {suggestion['description']}")
    
    # Stage 4: Enhance the prompt with suggested improvements
    print("\nStage 4: Enhancing the prompt")
    enhanced_prompt = enhance_prompt_general(base_prompt)
    print(f"Enhanced prompt: \"{enhanced_prompt}\"")
    
    # Stage 5: Extract entities from the enhanced prompt
    print("\nStage 5: Extracting entities from enhanced prompt")
    entities = extract_entities(enhanced_prompt)
    print("Detected entities:")
    for category, items in entities.items():
        if items:
            print(f"  {category.capitalize()}: {', '.join(items)}")
    
    # Stage 6: Convert to natural language caption
    print("\nStage 6: Converting to natural language caption")
    tags = text_to_tags(enhanced_prompt)
    natural_caption = natural_tags_to_text(tags)
    print(f"Natural caption: \"{natural_caption}\"")
    
    # Stage 7: Analyze sentiment and linguistic features
    print("\nStage 7: Analyzing sentiment and linguistic features")
    sentiment = analyze_sentiment(natural_caption)
    print(f"Sentiment: Positive {sentiment['pos']:.2f}, Negative {sentiment['neg']:.2f}, Neutral {sentiment['neu']:.2f}")
    
    pos_analysis = analyze_text_pos(natural_caption)
    print("Part of speech distribution:")
    for pos, words in pos_analysis.items():
        if words:
            print(f"  {pos.capitalize()}: {len(words)} words")
    
    # Stage 8: Create a simplified version for comparison
    print("\nStage 8: Creating a simplified version")
    simplified = simplify_prompt_structure(enhanced_prompt, target_length=5)
    print(f"Simplified prompt: \"{simplified}\"")
    
    # Stage 9: Compare original, enhanced, and simplified
    print("\nStage 9: Comparing versions")
    print("Original  : " + base_prompt)
    print("Enhanced  : " + enhanced_prompt)
    print("Simplified: " + simplified)
    
    # Show similarity scores
    base_to_enhanced = compute_text_similarity(base_prompt, enhanced_prompt)
    base_to_simplified = compute_text_similarity(base_prompt, simplified)
    enhanced_to_simplified = compute_text_similarity(enhanced_prompt, simplified)
    
    print(f"\nSimilarity scores (0-1 scale):")
    print(f"  Original to Enhanced: {base_to_enhanced:.2f}")
    print(f"  Original to Simplified: {base_to_simplified:.2f}")
    print(f"  Enhanced to Simplified: {enhanced_to_simplified:.2f}")
    
    # Stage 10: Generate a final optimized prompt with a realistic use case
    print("\nStage 10: Final optimized prompt")
    
    # Create a more detailed version with specific species details
    # Add anatomical details
    anatomy_enhanced = enhance_prompt_with_anatomy(enhanced_prompt, species="wolf", gender="male")
    
    # Add background details
    final_prompt = enhance_prompt_with_background(anatomy_enhanced, location="forest", time_of_day="evening")
    
    print(f"Final optimized prompt: \"{final_prompt}\"")
    
    # Output a summary of the workflow
    print("\nWorkflow Summary:")
    print("1. Started with basic prompt")
    print("2. Analyzed structure and identified improvements")
    print("3. Applied general enhancements")
    print("4. Extracted and categorized entities")
    print("5. Generated natural language caption")
    print("6. Analyzed linguistic features")
    print("7. Created simplified alternative")
    print("8. Compared different versions")
    print("9. Applied species-specific and background enhancements")
    print("10. Generated final optimized prompt")

def run_all_tests():
    """Run all NLP feature tests"""
    separator("COMPREHENSIVE NLP FEATURES TEST")
    
    print("This test will demonstrate all NLP features in CringeGen, with special focus on:")
    print("1. Tag/Text conversion including natural caption generation")
    print("2. Entity detection with furry/anime specialization")
    print("3. Prompt analysis and enhancement")
    print("4. NLTK integration for advanced language processing")
    print("5. Text generation for different character types")
    print("6. Integrated workflows combining multiple NLP components")
    
    # Run all tests
    test_tag_text_conversion()
    test_entity_detection()
    test_prompt_analysis()
    test_prompt_comparison()
    test_nltk_features()
    test_text_generation()
    test_integrated_workflow()
    
    separator("ALL TESTS COMPLETED")

if __name__ == "__main__":
    run_all_tests() 