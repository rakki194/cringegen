"""
NLTK utilities for advanced NLP capabilities in prompt generation.

This module provides NLTK-powered functions for:
1. Part-of-speech tagging and analysis
2. Named entity recognition
3. Sentiment analysis
4. Text similarity and comparison
5. Chunking and phrase extraction
"""

import string
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import nltk
from nltk.chunk import RegexpParser
from nltk.corpus import stopwords, wordnet
from nltk.metrics.distance import edit_distance
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

# Update imports to use new centralized data module
from cringegen.data.character_taxonomy import ALL_ANTHRO_CHARACTERS, CHARACTER_TO_FULL_NAME

# Original import for reference:
# from cringegen.data.furry_characters import ALL_ANTHRO_CHARACTERS, CHARACTER_TO_FULL_NAME

# Ensure NLTK data is downloaded
# nltk.download('punkt', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)
# nltk.download('maxent_ne_chunker', quiet=True)
# nltk.download('words', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('vader_lexicon', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag: str) -> str:
    """
    Convert NLTK POS tag to WordNet POS tag

    Args:
        tag: NLTK POS tag

    Returns:
        WordNet POS tag
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


def analyze_text_pos(text: str) -> Dict[str, List[str]]:
    """
    Analyze part-of-speech distribution in text

    Args:
        text: Text to analyze

    Returns:
        Dictionary mapping POS categories to lists of words
    """
    # Tokenize and tag
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)

    # Group by POS
    pos_categories = {"nouns": [], "verbs": [], "adjectives": [], "adverbs": [], "other": []}

    for word, tag in tagged:
        if tag.startswith("N"):
            pos_categories["nouns"].append(word)
        elif tag.startswith("V"):
            pos_categories["verbs"].append(word)
        elif tag.startswith("J"):
            pos_categories["adjectives"].append(word)
        elif tag.startswith("R"):
            pos_categories["adverbs"].append(word)
        else:
            pos_categories["other"].append(word)

    return pos_categories


def extract_phrases(text: str, phrase_type: str = "NP") -> List[str]:
    """
    Extract specific phrase types from text using chunking

    Args:
        text: Text to analyze
        phrase_type: Type of phrase to extract ('NP' for noun phrases,
                    'VP' for verb phrases, 'PP' for prepositional phrases)

    Returns:
        List of extracted phrases
    """
    # Tokenize and tag
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Define grammar patterns
    if phrase_type == "NP":
        # Noun phrase pattern
        grammar = r"""
            NP: {<DT|PP\$>?<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>+}  # Noun phrase
        """
    elif phrase_type == "VP":
        # Verb phrase pattern
        grammar = r"""
            VP: {<VB|VBD|VBG|VBN|VBP|VBZ><RB|RBR|RBS>*<NP|PP>?}  # Verb phrase
        """
    elif phrase_type == "PP":
        # Prepositional phrase pattern
        grammar = r"""
            PP: {<IN><NP>}  # Prepositional phrase
            NP: {<DT|PP\$>?<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>+}  # Noun phrase
        """
    else:
        raise ValueError(f"Unsupported phrase type: {phrase_type}")

    # Create parser and parse
    parser = RegexpParser(grammar)
    chunks = parser.parse(tagged)

    # Extract phrases
    phrases = []
    for subtree in chunks.subtrees():
        if subtree.label() == phrase_type:
            phrase = " ".join(word for word, tag in subtree.leaves())
            phrases.append(phrase)

    return phrases


def get_synonyms(word: str, pos: Optional[str] = None) -> List[str]:
    """
    Get synonyms for a word using WordNet

    Args:
        word: Word to find synonyms for
        pos: Part of speech (if known)

    Returns:
        List of synonyms
    """
    synonyms = []

    # If POS is provided, look up synonyms for that specific POS
    if pos:
        wordnet_pos = get_wordnet_pos(pos)
        for synset in wordnet.synsets(word, pos=wordnet_pos):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
    else:
        # Otherwise, look up synonyms for all POS
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)

    return synonyms[:10]  # Limit to top 10 to avoid overwhelming results


def get_antonyms(word: str) -> List[str]:
    """
    Get antonyms for a word using WordNet

    Args:
        word: Word to find antonyms for

    Returns:
        List of antonyms
    """
    antonyms = []

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                for antonym in lemma.antonyms():
                    antonym_word = antonym.name().replace("_", " ")
                    if antonym_word not in antonyms:
                        antonyms.append(antonym_word)

    return antonyms


def get_hypernyms(word: str, depth: int = 1) -> List[str]:
    """
    Get hypernyms (more general terms) for a word

    Args:
        word: Word to find hypernyms for
        depth: How many levels up to go

    Returns:
        List of hypernyms
    """
    hypernyms = []

    # Get synsets
    synsets = wordnet.synsets(word)
    if not synsets:
        return []

    # Use first synset (most common meaning)
    synset = synsets[0]

    # Get hypernyms up to specified depth
    current_depth = 0
    current_synsets = [synset]

    while current_depth < depth and current_synsets:
        next_synsets = []
        for current_synset in current_synsets:
            for hypernym in current_synset.hypernyms():
                for lemma in hypernym.lemmas():
                    hypernym_word = lemma.name().replace("_", " ")
                    if hypernym_word not in hypernyms:
                        hypernyms.append(hypernym_word)
                next_synsets.append(hypernym)

        current_synsets = next_synsets
        current_depth += 1

    return hypernyms


def get_hyponyms(word: str, depth: int = 1) -> List[str]:
    """
    Get hyponyms (more specific terms) for a word

    Args:
        word: Word to find hyponyms for
        depth: How many levels down to go

    Returns:
        List of hyponyms
    """
    hyponyms = []

    # Get synsets
    synsets = wordnet.synsets(word)
    if not synsets:
        return []

    # Use first synset (most common meaning)
    synset = synsets[0]

    # Get hyponyms up to specified depth
    current_depth = 0
    current_synsets = [synset]

    while current_depth < depth and current_synsets:
        next_synsets = []
        for current_synset in current_synsets:
            for hyponym in current_synset.hyponyms():
                for lemma in hyponym.lemmas():
                    hyponym_word = lemma.name().replace("_", " ")
                    if hyponym_word not in hyponyms:
                        hyponyms.append(hyponym_word)
                next_synsets.append(hyponym)

        current_synsets = next_synsets
        current_depth += 1

    return hyponyms


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text using VADER

    Args:
        text: Text to analyze

    Returns:
        Dictionary with sentiment scores
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)

    return sentiment


def extract_keywords_with_pos(text: str, top_n: int = 5) -> List[Tuple[str, str]]:
    """
    Extract keywords with their POS tags

    Args:
        text: Text to analyze
        top_n: Number of top keywords to extract

    Returns:
        List of (word, POS) tuples
    """
    # Tokenize and tag
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)

    # Filter out stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    content_words = [
        (word, tag) for word, tag in tagged if word.isalpha() and word not in stop_words
    ]

    # Count frequency of (word, tag) pairs
    freq = FreqDist(content_words)

    # Return top keywords with POS
    return freq.most_common(top_n)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text to get base forms of words

    Args:
        text: Text to lemmatize

    Returns:
        Lemmatized text
    """
    # Tokenize and tag
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Lemmatize each word with its POS
    lemmatized_words = []
    for word, tag in tagged:
        wordnet_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, wordnet_pos)
        lemmatized_words.append(lemma)

    # Join back into text
    return " ".join(lemmatized_words)


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two texts based on word overlap

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0-1)
    """
    # Tokenize and filter stopwords
    tokens1 = [
        word.lower()
        for word in word_tokenize(text1)
        if word.isalpha() and word.lower() not in stopwords.words("english")
    ]
    tokens2 = [
        word.lower()
        for word in word_tokenize(text2)
        if word.isalpha() and word.lower() not in stopwords.words("english")
    ]

    # Create sets for comparison
    set1 = set(tokens1)
    set2 = set(tokens2)

    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Avoid division by zero
    if union == 0:
        return 0

    return intersection / union


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text with special handling for furry/anthro characters,
    feral animals, humans, and other entities relevant to image generation.

    Args:
        text: Text to analyze

    Returns:
        Dictionary mapping entity types to lists of entities
    """
    # Initialize entity containers
    entities = {
        "humans": [],
        "anthro_characters": [],
        "feral_animals": [],
        "species": [],
        "places": [],
        "organizations": [],
        "companies": [],
        "dates": [],
        "times": [],
        "art_styles": [],
        "other": [],
    }

    # Tokenize, tag, and use NLTK's built-in named entity chunker
    sentences = sent_tokenize(text)

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)

        # Use NLTK's named entity chunker
        chunks = nltk.ne_chunk(tagged)

        # Process named entity chunks
        for chunk in chunks:
            if hasattr(chunk, "label"):
                entity_text = " ".join(c[0] for c in chunk)

                # Check if this entity is a known anthro character first
                if entity_text in ALL_ANTHRO_CHARACTERS:
                    # Use full name if available
                    full_name = CHARACTER_TO_FULL_NAME.get(entity_text, entity_text)
                    entities["anthro_characters"].append(full_name)
                    continue

                # Map NLTK's entity labels to our categories
                if chunk.label() == "PERSON":
                    entities["humans"].append(entity_text)
                elif chunk.label() in ("GPE", "GSP", "LOCATION"):
                    entities["places"].append(entity_text)
                elif chunk.label() == "ORGANIZATION":
                    entities["organizations"].append(entity_text)
                elif chunk.label() == "FACILITY":
                    entities["places"].append(entity_text)
                elif chunk.label() == "DATE":
                    entities["dates"].append(entity_text)
                elif chunk.label() == "TIME":
                    entities["times"].append(entity_text)
                else:
                    entities["other"].append(entity_text)

    # Define patterns for different entity types
    feral_animals = [
        "fox",
        "wolf",
        "dog",
        "cat",
        "tiger",
        "lion",
        "bear",
        "rabbit",
        "deer",
        "mouse",
        "rat",
        "horse",
        "cow",
        "sheep",
        "goat",
        "pig",
        "chicken",
        "duck",
        "goose",
        "elephant",
        "giraffe",
        "zebra",
        "snake",
        "lizard",
        "turtle",
        "frog",
        "alligator",
        "crocodile",
        "eagle",
        "hawk",
        "owl",
        "parrot",
        "raven",
        "crow",
    ]

    anthro_patterns = [
        "anthro",
        "anthropomorphic",
        "furry",
        "kemono",
        "fursona",
        "anthro fox",
        "fox girl",
        "cat girl",
        "wolf girl",
        "bunny girl",
        "dragon girl",
        "neko",
        "kitsune",
        "kemonomimi",
        "fursuit",
    ]

    species_patterns = [
        "human",
        "elf",
        "dwarf",
        "orc",
        "goblin",
        "troll",
        "fairy",
        "vampire",
        "werewolf",
        "dragon",
        "demon",
        "angel",
        "alien",
        "robot",
        "android",
        "cyborg",
        "mermaid",
        "centaur",
        "satyr",
    ]

    art_styles = [
        "anime",
        "manga",
        "digital art",
        "pixel art",
        "oil painting",
        "watercolor",
        "sketch",
        "cartoon",
        "realistic",
        "photorealistic",
        "impressionist",
        "expressionist",
        "surrealist",
        "cubist",
        "minimalist",
        "chibi",
        "cel shaded",
        "line art",
        "comic",
        "illustration",
    ]

    companies = [
        "google",
        "microsoft",
        "apple",
        "amazon",
        "facebook",
        "meta",
        "twitter",
        "netflix",
        "disney",
        "sony",
        "nintendo",
        "sega",
        "adobe",
        "autodesk",
        "blizzard",
        "ubisoft",
        "ea",
        "valve",
        "samsung",
        "lg",
        "intel",
        "amd",
        "nvidia",
    ]

    # Process text for special entity types
    words = word_tokenize(text.lower())
    text_lower = text.lower()

    # Sort anthro characters by length (longest first) to prioritize full names over short ones
    sorted_characters = sorted(ALL_ANTHRO_CHARACTERS, key=len, reverse=True)

    # Check for known anthro characters in the text (longer character names first)
    detected_characters = set()
    for character in sorted_characters:
        char_lower = character.lower()

        # Skip very short names that might cause false positives
        if len(char_lower.split()) == 1 and len(char_lower) < 4:
            continue

        # For multi-word character names (more specific, less chance of false positives)
        if len(char_lower.split()) > 1:
            # Look for the exact phrase
            if char_lower in text_lower:
                full_name = CHARACTER_TO_FULL_NAME.get(character, character)

                # Check if a longer form of this name was already detected
                skip = False
                for detected in detected_characters:
                    if char_lower in detected.lower():
                        skip = True
                        break

                if not skip:
                    entities["anthro_characters"].append(full_name)
                    detected_characters.add(full_name.lower())

        # For single-word character names (higher chance of false positives, be more careful)
        else:
            # Make sure it's a standalone word and not part of another word
            if char_lower in words:
                # If this character is a generic animal name (like "Fox", "Wolf"), be extra careful
                if char_lower in [animal.lower() for animal in feral_animals]:
                    # Check for context to confirm it's a character name
                    # Look for capitalization or specific context
                    capital_match = False
                    for word in word_tokenize(text):
                        if word.lower() == char_lower and word[0].isupper():
                            capital_match = True
                            break

                    # If not clearly a character name, skip it
                    if not capital_match:
                        continue

                full_name = CHARACTER_TO_FULL_NAME.get(character, character)

                # Check if a longer form of this name was already detected
                skip = False
                for detected in detected_characters:
                    if char_lower in detected.lower().split():
                        skip = True
                        break

                if not skip:
                    entities["anthro_characters"].append(full_name)
                    detected_characters.add(full_name.lower())

    # Find anthro character patterns (for generic anthro types not in our database)
    for pattern in anthro_patterns:
        if pattern in text_lower:
            # Find the full anthro character description
            # Look for pattern + animal type
            for animal in feral_animals:
                anthro_pattern = f"{pattern} {animal}"
                if anthro_pattern in text_lower and anthro_pattern not in [
                    p.lower() for p in entities["anthro_characters"]
                ]:
                    entities["anthro_characters"].append(anthro_pattern)
                animal_pattern = f"{animal} {pattern}"
                if animal_pattern in text_lower and animal_pattern not in [
                    p.lower() for p in entities["anthro_characters"]
                ]:
                    entities["anthro_characters"].append(animal_pattern)

            # If no animal specified, just add the anthro term
            if pattern not in [
                part for entity in entities["anthro_characters"] for part in entity.split()
            ]:
                entities["anthro_characters"].append(pattern)

    # Check for species terms
    for species in species_patterns:
        if species in words:
            entities["species"].append(species)

    # Find feral animals - but exclude animals that are part of anthro descriptions
    anthro_words = [
        word.lower() for entity in entities["anthro_characters"] for word in entity.split()
    ]
    for animal in feral_animals:
        if animal in words and animal not in anthro_words:
            # Check if the animal isn't already part of a character name
            is_character = False
            for char in entities["anthro_characters"]:
                if animal.lower() in char.lower().split():
                    is_character = True
                    break

            if not is_character:
                entities["feral_animals"].append(animal)

    # Find art styles
    for style in art_styles:
        if style in text_lower:
            entities["art_styles"].append(style)

    # Find companies
    for company in companies:
        if company in words:
            entities["companies"].append(company)
            # Remove from organizations if it was classified there
            if company in entities["organizations"]:
                entities["organizations"].remove(company)

    # Process any remaining capitalized words that weren't caught above
    for sentence in sentences:
        for word in word_tokenize(sentence):
            # Check if it's capitalized and not at the start of the sentence and not already classified
            is_entity = False
            for entity_list in entities.values():
                if word.lower() in [entity.lower() for entity in entity_list]:
                    is_entity = True
                    break

            if (
                word[0].isupper()
                and word.lower() not in stopwords.words("english")
                and len(word) > 1
                and not is_entity
            ):

                # Try to classify based on word
                word_lower = word.lower()

                # Check if it's a known anthro character (using more strict checking)
                found_character = False
                for char in sorted_characters:
                    if word_lower == char.lower() or (
                        word_lower in char.lower().split() and len(word) > 3
                    ):
                        # Check for contextual clues
                        if word[0].isupper():  # It's capitalized
                            full_name = CHARACTER_TO_FULL_NAME.get(char, char)
                            found_character = True

                            # Check if a form of this name was already detected
                            skip = False
                            for detected in [e.lower() for e in entities["anthro_characters"]]:
                                if word_lower in detected.split():
                                    skip = True
                                    break

                            if not skip:
                                entities["anthro_characters"].append(full_name)
                            break

                if found_character:
                    continue
                # Otherwise check other categories
                elif word_lower in [a.lower() for a in feral_animals]:
                    entities["feral_animals"].append(word)
                elif word_lower in [s.lower() for s in species_patterns]:
                    entities["species"].append(word)
                elif word_lower in [c.lower() for c in companies]:
                    entities["companies"].append(word)
                else:
                    entities["other"].append(word)

    # Remove duplicates while preserving order
    for category in entities:
        unique_entities = []
        seen = set()
        for entity in entities[category]:
            if entity.lower() not in seen:
                unique_entities.append(entity)
                seen.add(entity.lower())
        entities[category] = unique_entities

    return entities


def generate_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Generate n-grams from text

    Args:
        text: Text to analyze
        n: Size of n-grams

    Returns:
        List of n-grams
    """
    tokens = word_tokenize(text)

    # Generate n-grams
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n])
        ngrams.append(ngram)

    return ngrams


def find_collocations(
    text: str, window_size: int = 2, min_freq: int = 2
) -> List[Tuple[str, str, int]]:
    """
    Find word collocations (words that frequently appear together)

    Args:
        text: Text to analyze
        window_size: Size of window to consider
        min_freq: Minimum frequency to include

    Returns:
        List of (word1, word2, frequency) tuples
    """
    # Tokenize and remove stopwords/punctuation
    tokens = [
        word.lower()
        for word in word_tokenize(text)
        if word.isalpha() and word.lower() not in stopwords.words("english")
    ]

    # Find collocations
    collocations = defaultdict(int)

    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]

        # Create all pairs in the window
        for j in range(len(window)):
            for k in range(j + 1, len(window)):
                pair = (window[j], window[k])
                collocations[pair] += 1

    # Filter by minimum frequency
    filtered_collocations = [
        (word1, word2, freq) for (word1, word2), freq in collocations.items() if freq >= min_freq
    ]

    # Sort by frequency
    return sorted(filtered_collocations, key=lambda x: x[2], reverse=True)
