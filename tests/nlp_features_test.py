#!/usr/bin/env python3
"""
Comprehensive testing of the NLP features in cringegen:
1. Tag/Text conversion
2. Prompt analysis utilities
3. NLTK integration
"""

import sys
import os
from pprint import pprint

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
    simplify_prompt_structure,
    extract_keywords,
    detect_prompt_style,
    # NLTK utilities
    analyze_text_pos,
    extract_phrases,
    get_synonyms,
    get_antonyms,
    get_hypernyms,
    get_hyponyms,
    analyze_sentiment,
    extract_keywords_with_pos,
    lemmatize_text,
    compute_text_similarity,
    extract_entities,
    generate_ngrams,
    find_collocations,
)


def test_tag_converter():
    """Test tag conversion features"""
    print("\n===== TESTING TAG CONVERTER =====\n")

    # Test categorize_tags with different tag types
    print("Testing tag categorization:")
    test_tags = [
        "masterpiece",
        "best quality",
        "detailed",
        "fox girl",
        "anthro",
        "red fur",
        "blue eyes",
        "wearing blue dress",
        "smiling",
        "forest background",
        "digital art",
        "dynamic lighting",
    ]
    print("\nInput tags:", ", ".join(test_tags))
    categories = categorize_tags(test_tags)
    print("\nCategorized tags:")
    for category, items in categories.items():
        if items:
            print(f"  {category.capitalize()}: {', '.join(items)}")

    # Test conversion from tags to text in different styles
    print("\nTesting tags to text conversion:")

    # Concise style
    concise = tags_to_text(test_tags, style="concise")
    print("\nConcise style:")
    print(f"  {concise}")

    # Descriptive style (default)
    descriptive = tags_to_text(test_tags, style="descriptive")
    print("\nDescriptive style:")
    print(f"  {descriptive}")

    # Detailed style
    detailed = tags_to_text(test_tags, style="detailed")
    print("\nDetailed style:")
    print(f"  {detailed}")

    # Test conversion from text to tags
    print("\nTesting text to tags conversion:")

    # Convert from descriptive text back to tags
    print("\nInput text:")
    print(f"  {descriptive}")

    tags_from_text = text_to_tags(descriptive)
    print("\nExtracted tags:")
    print(f"  {', '.join(tags_from_text)}")

    # Try with a different format
    custom_text = "A high quality digital painting of a happy fox girl with red fur and blue eyes, standing in a snowy forest"
    print("\nInput custom text:")
    print(f"  {custom_text}")

    custom_tags = text_to_tags(custom_text)
    print("\nExtracted tags from custom text:")
    print(f"  {', '.join(custom_tags)}")

    # Bidirectional conversion test
    print("\nBidirectional conversion test:")
    original_tags = ["masterpiece", "detailed", "fox", "snowy forest", "blue jacket"]
    print(f"  Original tags: {', '.join(original_tags)}")

    text = tags_to_text(original_tags)
    print(f"  Converted to text: {text}")

    back_to_tags = text_to_tags(text)
    print(f"  Back to tags: {', '.join(back_to_tags)}")

    # Check how many original tags we recovered
    recovered = set(tag.lower() for tag in back_to_tags).intersection(
        set(tag.lower() for tag in original_tags)
    )
    print(f"  Recovered {len(recovered)} out of {len(original_tags)} original tags")


def test_prompt_analyzer():
    """Test prompt analysis utilities"""
    print("\n===== TESTING PROMPT ANALYZER =====\n")

    # Test prompts
    prompts = [
        "masterpiece, high quality, detailed, red fox, forest, digital art",
        "a beautiful painting of a wolf howling at the moon, moonlight, forest, artstation, 4k, highly detailed",
        "low quality, blurry image of a fox with big ears",
        "masterpiece, photorealistic, extremely detailed, octane render, 8k, professional photograph, natural lighting, photoshoot, sharp focus, depth of field",
        "A cyberpunk city street at night with neon lights and a female character wearing a futuristic outfit, highly detailed, realistic, cinematic lighting, digital art",
    ]

    # Test get_prompt_structure
    print("Testing prompt structure extraction:")
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        structure = get_prompt_structure(prompt)
        print("\nStructure:")
        for component, items in structure.items():
            if items:
                if isinstance(items, list):
                    print(f"  {component.capitalize()}: {', '.join(items)}")
                else:
                    print(f"  {component.capitalize()}: {items}")

    # Test prompt analysis
    print("\nTesting prompt analysis:")
    for i, prompt in enumerate(prompts):
        print(f"\nAnalyzing prompt {i+1}:")
        analysis = analyze_prompt_structure(prompt)
        print(f"  Word count: {analysis.word_count}")
        print(f"  Complexity score: {analysis.complexity_score:.2f}/15")
        print(f"  Issues detected: {len(analysis.detected_issues)}")

        # Show suggested improvements
        suggestions = analysis.suggestions
        if suggestions:
            print("\n  Suggested improvements:")
            for suggestion in suggestions:
                print(f"    - {suggestion['description']}")

    # Test prompt comparison
    print("\nTesting prompt comparison:")
    prompt1 = prompts[0]  # Simple prompt
    prompt2 = prompts[1]  # More complex prompt

    print(f"\nPrompt 1: {prompt1}")
    print(f"Prompt 2: {prompt2}")

    comparison = compare_prompts(prompt1, prompt2)

    print("\nComparison results:")
    print(f"  Unique to prompt 1: {', '.join(comparison['unique_to_prompt1'])}")
    print(f"  Unique to prompt 2: {', '.join(comparison['unique_to_prompt2'])}")
    print(f"  Common words: {', '.join(comparison['common_words'])}")
    print(f"  Word count difference: {comparison['word_count_diff']}")

    # Test extracting keywords
    print("\nTesting keyword extraction:")
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1} keywords:")
        keywords = extract_keywords(prompt)
        print(f"  {', '.join(keywords)}")

    # Test style detection
    print("\nTesting style detection:")
    for i, prompt in enumerate(prompts):
        style = detect_prompt_style(prompt)
        print(f"  Prompt {i+1} style: {style}")

    # Test prompt simplification
    print("\nTesting prompt simplification:")
    complex_prompt = prompts[4]  # Use the most complex prompt
    print(f"\nOriginal prompt ({len(complex_prompt.split())} words):")
    print(f"  {complex_prompt}")

    simplified = simplify_prompt_structure(complex_prompt, target_length=15)
    print(f"\nSimplified prompt:")
    print(f"  {simplified}")


def test_nltk_integration():
    """Test NLTK integration features"""
    print("\n===== TESTING NLTK INTEGRATION =====\n")

    # Test texts
    texts = [
        "A masterful digital painting of a red fox with a fluffy tail wearing a blue hoodie.",
        "The wolf howled at the moon as the snow fell gently on the ancient forest.",
        "A cyberpunk cityscape with tall skyscrapers, neon signs, and flying cars. The streets are crowded with people wearing futuristic clothing.",
    ]

    # Test POS tagging
    print("Testing part-of-speech analysis:")
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: {text}")
        pos_analysis = analyze_text_pos(text)
        print("\nPart-of-speech distribution:")
        for pos, words in pos_analysis.items():
            print(f"  {pos.capitalize()}: {', '.join(words[:5])}{'...' if len(words) > 5 else ''}")

    # Test phrase extraction
    print("\nTesting phrase extraction:")
    text = texts[2]  # Use the most complex text
    print(f"\nText: {text}")

    # Extract noun phrases
    noun_phrases = extract_phrases(text, phrase_type="NP")
    print("\nNoun phrases:")
    for i, phrase in enumerate(noun_phrases[:10]):
        print(f"  {i+1}. {phrase}")

    # Extract verb phrases
    verb_phrases = extract_phrases(text, phrase_type="VP")
    print("\nVerb phrases:")
    for i, phrase in enumerate(verb_phrases[:5]):
        print(f"  {i+1}. {phrase}")

    # Test WordNet functions
    print("\nTesting WordNet functions:")

    # Test synonyms
    words = ["detailed", "happy", "forest", "animal"]
    print("\nSynonyms:")
    for word in words:
        synonyms = get_synonyms(word)
        print(f"  {word}: {', '.join(synonyms[:5])}{'...' if len(synonyms) > 5 else ''}")

    # Test antonyms
    print("\nAntonyms:")
    for word in words:
        antonyms = get_antonyms(word)
        print(f"  {word}: {', '.join(antonyms[:5])}{'...' if len(antonyms) > 5 else ''}")

    # Test hypernyms (more general terms)
    print("\nHypernyms (more general terms):")
    for word in words:
        hypernyms = get_hypernyms(word)
        print(f"  {word} → {', '.join(hypernyms[:3])}{'...' if len(hypernyms) > 3 else ''}")

    # Test hyponyms (more specific terms)
    print("\nHyponyms (more specific terms):")
    for word in ["animal", "forest", "clothing"]:
        hyponyms = get_hyponyms(word)
        print(f"  {word} → {', '.join(hyponyms[:5])}{'...' if len(hyponyms) > 5 else ''}")

    # Test sentiment analysis
    print("\nTesting sentiment analysis:")
    for i, text in enumerate(texts):
        sentiment = analyze_sentiment(text)
        print(f"\nText {i+1} sentiment:")
        print(f"  Positive: {sentiment['pos']:.2f}")
        print(f"  Negative: {sentiment['neg']:.2f}")
        print(f"  Neutral: {sentiment['neu']:.2f}")
        print(f"  Compound: {sentiment['compound']:.2f}")

    # Test text lemmatization
    print("\nTesting text lemmatization:")
    text = "The foxes were running quickly through the forests and jumping over fallen trees"
    print(f"\nOriginal: {text}")
    lemmatized = lemmatize_text(text)
    print(f"Lemmatized: {lemmatized}")

    # Test text similarity
    print("\nTesting text similarity:")
    text1 = "A red fox in a forest"
    text2 = "A crimson colored fox inside a woodland area"
    text3 = "A wolf howling at the moon"

    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")

    sim1_2 = compute_text_similarity(text1, text2)
    sim1_3 = compute_text_similarity(text1, text3)

    print(f"\nSimilarity 1-2: {sim1_2:.2f}")
    print(f"Similarity 1-3: {sim1_3:.2f}")

    # Test entity extraction
    print("\nTesting entity extraction:")
    entity_text = (
        "John Smith visited New York City and met with representatives from Google and Apple."
    )
    print(f"\nText: {entity_text}")

    entities = extract_entities(entity_text)
    print("\nExtracted entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"  {entity_type.capitalize()}: {', '.join(entity_list)}")

    # Test n-gram generation
    print("\nTesting n-gram generation:")
    ngram_text = "The quick brown fox jumps over the lazy dog."
    print(f"\nText: {ngram_text}")

    bigrams = generate_ngrams(ngram_text, n=2)
    print(f"\nBigrams: {', '.join(bigrams[:5])}...")

    trigrams = generate_ngrams(ngram_text, n=3)
    print(f"Trigrams: {', '.join(trigrams[:5])}...")

    # Test collocation finding
    print("\nTesting collocation finding:")
    collocation_text = "The red fox jumped over the brown fence. Another red fox was seen nearby. The foxes in this forest are usually red."
    print(f"\nText: {collocation_text}")

    collocations = find_collocations(collocation_text, window_size=3, min_freq=2)
    print("\nCollocations (word pairs that appear together):")
    for word1, word2, freq in collocations[:5]:
        print(f"  '{word1}' + '{word2}': {freq} occurrences")


if __name__ == "__main__":
    print("=====================================")
    print("COMPREHENSIVE NLP FEATURES TEST")
    print("=====================================")

    # Test tag converter
    test_tag_converter()

    # Test prompt analyzer
    test_prompt_analyzer()

    # Test NLTK integration
    test_nltk_integration()

    print("\n=====================================")
    print("ALL TESTS COMPLETED")
    print("=====================================")
