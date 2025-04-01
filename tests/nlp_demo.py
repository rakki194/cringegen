#!/usr/bin/env python3
"""
Demonstration of the new NLP features:
1. Bidirectional conversion between natural language and tags
2. Prompt analysis utilities
3. NLTK integration features
"""

import sys
import os

# Add the parent directory to the path so we can import cringegen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cringegen.prompt_generation.nlp import (
    # Tag converter utilities
    tags_to_text,
    text_to_tags,
    categorize_tags,
    # Prompt analyzer utilities
    analyze_prompt_structure,
    get_prompt_structure,
    compare_prompts,
    suggest_improvements,
    # NLTK utilities
    analyze_text_pos,
    extract_phrases,
    get_synonyms,
    extract_entities,
)


def demo_tag_converter():
    """Demonstrate tag-to-text and text-to-tag conversion"""
    print("\n=== Tag Converter Demo ===\n")

    # Sample tags
    tags = [
        "masterpiece",
        "high quality",
        "detailed",
        "fox",
        "anthro",
        "red fur",
        "wearing hoodie",
        "sitting",
        "forest",
        "digital art",
    ]

    print("Original tags:")
    print(", ".join(tags))

    # Categorize tags
    print("\nCategorized tags:")
    categories = categorize_tags(tags)
    for category, items in categories.items():
        if items:
            print(f"{category.capitalize()}: {', '.join(items)}")

    # Convert tags to natural language
    print("\nTags to text (concise):")
    concise_text = tags_to_text(tags, style="concise")
    print(concise_text)

    print("\nTags to text (descriptive):")
    descriptive_text = tags_to_text(tags, style="descriptive")
    print(descriptive_text)

    print("\nTags to text (detailed):")
    detailed_text = tags_to_text(tags, style="detailed")
    print(detailed_text)

    # Convert text back to tags
    print("\nText to tags:")
    extracted_tags = text_to_tags(descriptive_text)
    print(", ".join(extracted_tags))


def demo_prompt_analyzer():
    """Demonstrate prompt analysis utilities"""
    print("\n=== Prompt Analyzer Demo ===\n")

    # Sample prompt
    prompt = "masterpiece, high quality, detailed, red fox with fluffy tail, wearing a blue hoodie, sitting under a tree in a forest, digital art style, warm lighting"

    print("Original prompt:")
    print(prompt)

    # Analyze prompt structure
    print("\nPrompt structure:")
    structure = get_prompt_structure(prompt)
    for component, items in structure.items():
        if items:
            if isinstance(items, list):
                print(f"{component.capitalize()}: {', '.join(items)}")
            else:
                print(f"{component.capitalize()}: {items}")

    # Full prompt analysis
    print("\nPrompt analysis:")
    analysis = analyze_prompt_structure(prompt)
    print(f"Word count: {analysis.word_count}")
    print(f"Complexity score: {analysis.complexity_score:.2f}/15")

    print("\nSection presence:")
    for section, present in analysis.section_presence.items():
        print(f"{section.capitalize()}: {'Yes' if present else 'No'}")

    print("\nKeyword density:")
    for word, density in analysis.keyword_density.items():
        print(f"{word}: {density:.2f}")

    # Suggest improvements
    print("\nSuggested improvements:")
    suggestions = suggest_improvements(prompt)
    if suggestions:
        for suggestion in suggestions:
            print(f"- {suggestion['description']}")
    else:
        print("No suggestions - prompt looks good!")

    # Compare prompts
    print("\nComparing prompts:")
    prompt2 = "high quality digital painting of a fox in the woods, wearing clothes, detailed fur"
    comparison = compare_prompts(prompt, prompt2)

    print("\nUnique to prompt 1:")
    print(", ".join(comparison["unique_to_prompt1"]))

    print("\nUnique to prompt 2:")
    print(", ".join(comparison["unique_to_prompt2"]))

    print("\nCommon words:")
    print(", ".join(comparison["common_words"]))


def demo_nltk_utils():
    """Demonstrate NLTK integration features"""
    print("\n=== NLTK Utilities Demo ===\n")

    # Sample text
    text = "A masterfully detailed digital painting of an anthropomorphic red fox with a fluffy tail, wearing a cozy blue hoodie, sitting peacefully under an ancient oak tree in a lush green forest. The scene is illuminated by warm golden sunlight filtering through the leaves."

    print("Original text:")
    print(text)

    # Analyze part-of-speech
    print("\nPart-of-speech analysis:")
    pos_analysis = analyze_text_pos(text)

    print(f"Nouns: {', '.join(pos_analysis['nouns'][:5])}...")
    print(f"Verbs: {', '.join(pos_analysis['verbs'])}")
    print(f"Adjectives: {', '.join(pos_analysis['adjectives'][:5])}...")

    # Extract phrases
    print("\nNoun phrases:")
    noun_phrases = extract_phrases(text, phrase_type="NP")
    for i, phrase in enumerate(noun_phrases[:5]):
        print(f"{i+1}. {phrase}")

    # Get synonyms
    print("\nSynonyms for 'detailed':")
    synonyms = get_synonyms("detailed")
    print(", ".join(synonyms))

    # Extract entities
    print("\nEntities:")
    entities = extract_entities(text)
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"{entity_type.capitalize()}: {', '.join(entity_list)}")


if __name__ == "__main__":
    # Run all demos
    demo_tag_converter()
    demo_prompt_analyzer()
    demo_nltk_utils()

    print("\nDemo completed successfully!")
