"""
Prompt analysis utilities for analyzing and improving image generation prompts.

This module provides tools to:
1. Analyze the semantic content of prompts
2. Identify prompt structure and components
3. Detect potential issues and improvement opportunities
4. Provide suggestions for prompt enhancement
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Common prompt prefixes and suffixes that don't add semantic value
COMMON_PREFIXES = [
    "a photo of",
    "a picture of",
    "an image of",
    "a rendering of",
    "a painting of",
    "a drawing of",
    "a sketch of",
    "an illustration of",
    "a digital art of",
]

# Specific prompt sections for different generation models
PROMPT_SECTIONS = {
    "quality": ["masterpiece", "best quality", "high quality", "detailed", "intricate"],
    "style": ["digital art", "oil painting", "watercolor", "sketch", "3d render"],
    "composition": ["portrait", "full body", "close-up", "wide shot", "landscape"],
    "lighting": ["studio lighting", "natural lighting", "dramatic lighting", "rim lighting"],
    "camera": ["dslr", "4k", "8k", "hdr", "sharp focus", "bokeh"],
    "negative": ["low quality", "blurry", "pixelated", "distorted", "deformed"],
}

# Common issues in prompts
PROMPT_ISSUES = {
    "repetition": "Repeated words or phrases",
    "contradictory": "Contradictory descriptions",
    "over_specification": "Too many specific details",
    "under_specification": "Too vague or general",
    "grammar": "Grammatical issues",
    "unbalanced": "Unbalanced focus on aspects",
    "negative_in_positive": "Negative terms in positive prompt",
}


class PromptAnalysis:
    """Class for storing and accessing prompt analysis results"""

    def __init__(self, prompt: str):
        """
        Initialize prompt analysis

        Args:
            prompt: The prompt text to analyze
        """
        self.prompt = prompt
        self.tokens = word_tokenize(prompt.lower())
        self.pos_tags = pos_tag(self.tokens)
        self.sentences = sent_tokenize(prompt)

        # Analysis components to be filled
        self.word_count = len(self.tokens)
        self.char_count = len(prompt)
        self.section_presence = {}
        self.complexity_score = 0
        self.detected_issues = []
        self.keyword_density = {}
        self.suggestions = []

        # Perform analysis
        self._analyze()

    def _analyze(self) -> None:
        """Perform complete analysis of the prompt"""
        self._analyze_sections()
        self._calculate_complexity()
        self._analyze_issues()
        self._analyze_keyword_density()
        self._generate_suggestions()

    def _analyze_sections(self) -> None:
        """Analyze presence of different prompt sections"""
        prompt_lower = self.prompt.lower()
        for section, keywords in PROMPT_SECTIONS.items():
            self.section_presence[section] = any(kw in prompt_lower for kw in keywords)

    def _calculate_complexity(self) -> None:
        """Calculate prompt complexity score"""
        # Based on several factors:
        # 1. Length of prompt
        # 2. Vocabulary diversity
        # 3. Sentence structure

        # Length factor (0-5)
        length_factor = min(5, self.word_count / 20)

        # Vocabulary diversity (0-5)
        unique_words = set(
            word.lower()
            for word in self.tokens
            if word.isalpha() and word.lower() not in stopwords.words("english")
        )
        diversity_factor = min(5, len(unique_words) / 10)

        # Sentence complexity (0-5)
        avg_sentence_length = self.word_count / max(1, len(self.sentences))
        sentence_factor = min(5, avg_sentence_length / 5)

        # Calculate final score (0-15)
        self.complexity_score = length_factor + diversity_factor + sentence_factor

    def _analyze_issues(self) -> None:
        """Detect potential issues in the prompt"""
        prompt_lower = self.prompt.lower()

        # Check for repetition
        word_counts = Counter(self.tokens)
        repeats = [
            word
            for word, count in word_counts.items()
            if count > 1 and word.isalpha() and len(word) > 3
        ]

        if repeats:
            self.detected_issues.append(
                {
                    "type": "repetition",
                    "description": f"Repeated words: {', '.join(repeats)}",
                    "severity": "medium",
                }
            )

        # Check for contradictions (basic implementation)
        contradictions = []
        if "small" in prompt_lower and "large" in prompt_lower:
            contradictions.append("small/large")
        if "bright" in prompt_lower and "dark" in prompt_lower:
            contradictions.append("bright/dark")

        if contradictions:
            self.detected_issues.append(
                {
                    "type": "contradictory",
                    "description": f"Potential contradictions: {', '.join(contradictions)}",
                    "severity": "high",
                }
            )

        # Check for over-specification
        if self.word_count > 50:
            self.detected_issues.append(
                {
                    "type": "over_specification",
                    "description": "Prompt is very long and may have too many details",
                    "severity": "low",
                }
            )

        # Check for under-specification
        if self.word_count < 5:
            self.detected_issues.append(
                {
                    "type": "under_specification",
                    "description": "Prompt is very short and may lack details",
                    "severity": "medium",
                }
            )

        # Check for negative terms in positive prompt
        negative_terms = PROMPT_SECTIONS["negative"]
        found_negatives = [term for term in negative_terms if term in prompt_lower]

        if found_negatives:
            self.detected_issues.append(
                {
                    "type": "negative_in_positive",
                    "description": f"Negative terms in positive prompt: {', '.join(found_negatives)}",
                    "severity": "high",
                }
            )

    def _analyze_keyword_density(self) -> None:
        """Analyze keyword density in the prompt"""
        # Remove stopwords
        content_words = [
            word.lower()
            for word in self.tokens
            if word.isalpha() and word.lower() not in stopwords.words("english")
        ]

        # Calculate frequency distribution
        fdist = FreqDist(content_words)

        # Get top keywords
        self.keyword_density = {
            word: count / len(content_words) for word, count in fdist.most_common(10)
        }

    def _generate_suggestions(self) -> None:
        """Generate improvement suggestions based on analysis"""
        # Check for missing sections
        for section, present in self.section_presence.items():
            if not present and section != "negative":
                self.suggestions.append(
                    {
                        "type": "add_section",
                        "section": section,
                        "description": f"Consider adding {section} details to your prompt",
                    }
                )

        # Address detected issues
        for issue in self.detected_issues:
            if issue["type"] == "repetition":
                self.suggestions.append(
                    {
                        "type": "fix_issue",
                        "issue": "repetition",
                        "description": "Remove repeated words or use synonyms",
                    }
                )
            elif issue["type"] == "contradictory":
                self.suggestions.append(
                    {
                        "type": "fix_issue",
                        "issue": "contradictory",
                        "description": "Resolve contradictory descriptions",
                    }
                )
            elif issue["type"] == "over_specification":
                self.suggestions.append(
                    {
                        "type": "fix_issue",
                        "issue": "over_specification",
                        "description": "Consider simplifying the prompt by focusing on key elements",
                    }
                )
            elif issue["type"] == "under_specification":
                self.suggestions.append(
                    {
                        "type": "fix_issue",
                        "issue": "under_specification",
                        "description": "Add more details about subject, style, or composition",
                    }
                )
            elif issue["type"] == "negative_in_positive":
                self.suggestions.append(
                    {
                        "type": "fix_issue",
                        "issue": "negative_in_positive",
                        "description": "Move negative terms to the negative prompt",
                    }
                )

        # Check prompt structure
        if not any(prefix in self.prompt.lower() for prefix in COMMON_PREFIXES):
            self.suggestions.append(
                {
                    "type": "structure",
                    "description": "Consider starting with a clear subject description",
                }
            )


def analyze_prompt(prompt: str) -> PromptAnalysis:
    """
    Analyze a prompt and return detailed analysis

    Args:
        prompt: The prompt text to analyze

    Returns:
        PromptAnalysis object with detailed analysis
    """
    return PromptAnalysis(prompt)


def get_prompt_structure(prompt: str) -> Dict[str, Any]:
    """
    Extract the structure of a prompt

    Args:
        prompt: The prompt text to analyze

    Returns:
        Dictionary with identified prompt structure components
    """
    # Initialize structure components
    structure = {
        "subject": None,
        "quality": [],
        "style": [],
        "composition": [],
        "lighting": [],
        "camera": [],
        "other": [],
    }

    # Tokenize and tag parts of speech
    tokens = word_tokenize(prompt.lower())
    pos_tags = pos_tag(tokens)

    # Extract subject (first noun phrase, typically)
    subject_pattern = r"(?:a|an|the)?\s?(?:\w+\s)?(?:\w+\s)?\w+"
    subject_match = re.search(subject_pattern, prompt.lower())
    if subject_match:
        structure["subject"] = subject_match.group(0).strip()

    # Extract other components
    prompt_lower = prompt.lower()

    # Check for quality terms
    for term in PROMPT_SECTIONS["quality"]:
        if term in prompt_lower:
            structure["quality"].append(term)

    # Check for style terms
    for term in PROMPT_SECTIONS["style"]:
        if term in prompt_lower:
            structure["style"].append(term)

    # Check for composition terms
    for term in PROMPT_SECTIONS["composition"]:
        if term in prompt_lower:
            structure["composition"].append(term)

    # Check for lighting terms
    for term in PROMPT_SECTIONS["lighting"]:
        if term in prompt_lower:
            structure["lighting"].append(term)

    # Check for camera terms
    for term in PROMPT_SECTIONS["camera"]:
        if term in prompt_lower:
            structure["camera"].append(term)

    return structure


def compare_prompts(prompt1: str, prompt2: str) -> Dict[str, Any]:
    """
    Compare two prompts and identify differences

    Args:
        prompt1: First prompt to compare
        prompt2: Second prompt to compare

    Returns:
        Dictionary with comparison results
    """
    # Analyze both prompts
    analysis1 = analyze_prompt(prompt1)
    analysis2 = analyze_prompt(prompt2)

    # Get unique words in each prompt
    words1 = set(
        word.lower()
        for word in analysis1.tokens
        if word.isalpha() and word.lower() not in stopwords.words("english")
    )
    words2 = set(
        word.lower()
        for word in analysis2.tokens
        if word.isalpha() and word.lower() not in stopwords.words("english")
    )

    # Find differences
    unique_to_prompt1 = words1 - words2
    unique_to_prompt2 = words2 - words1
    common_words = words1 & words2

    # Compare section presence
    section_differences = {}
    for section in analysis1.section_presence:
        if analysis1.section_presence[section] != analysis2.section_presence[section]:
            section_differences[section] = {
                "prompt1": analysis1.section_presence[section],
                "prompt2": analysis2.section_presence[section],
            }

    # Compare complexity
    complexity_diff = abs(analysis1.complexity_score - analysis2.complexity_score)

    return {
        "unique_to_prompt1": sorted(list(unique_to_prompt1)),
        "unique_to_prompt2": sorted(list(unique_to_prompt2)),
        "common_words": sorted(list(common_words)),
        "word_count_diff": analysis2.word_count - analysis1.word_count,
        "section_differences": section_differences,
        "complexity_diff": complexity_diff,
    }


def suggest_improvements(prompt: str) -> List[Dict[str, str]]:
    """
    Suggest improvements for a prompt

    Args:
        prompt: The prompt text to analyze

    Returns:
        List of improvement suggestions
    """
    analysis = analyze_prompt(prompt)
    return analysis.suggestions


def simplify_prompt(prompt: str, target_length: int = 30) -> str:
    """
    Simplify a prompt while preserving its core meaning

    Args:
        prompt: The prompt text to simplify
        target_length: Target word count

    Returns:
        Simplified prompt
    """
    # Get current word count
    tokens = word_tokenize(prompt)
    current_length = len(tokens)

    # If already shorter than target, return as is
    if current_length <= target_length:
        return prompt

    # Get prompt structure
    structure = get_prompt_structure(prompt)

    # Build simplified prompt starting with the most important elements
    simplified_parts = []

    # Add subject
    if structure["subject"]:
        simplified_parts.append(structure["subject"])

    # Add one quality term
    if structure["quality"]:
        simplified_parts.append(structure["quality"][0])

    # Add one style term
    if structure["style"]:
        simplified_parts.append(structure["style"][0])

    # Add one composition term
    if structure["composition"]:
        simplified_parts.append(structure["composition"][0])

    # Add one lighting term if space permits
    if structure["lighting"] and len(simplified_parts) < target_length:
        simplified_parts.append(structure["lighting"][0])

    # Add one camera term if space permits
    if structure["camera"] and len(simplified_parts) < target_length:
        simplified_parts.append(structure["camera"][0])

    # Join parts into a simplified prompt
    return ", ".join(simplified_parts)


def extract_keywords(prompt: str, top_n: int = 5) -> List[str]:
    """
    Extract the most important keywords from a prompt

    Args:
        prompt: The prompt text to analyze
        top_n: Number of top keywords to extract

    Returns:
        List of top keywords
    """
    # Tokenize and remove stopwords
    tokens = word_tokenize(prompt.lower())
    content_words = [
        word for word in tokens if word.isalpha() and word.lower() not in stopwords.words("english")
    ]

    # Calculate frequency distribution
    fdist = FreqDist(content_words)

    # Return top keywords
    return [word for word, _ in fdist.most_common(top_n)]


def detect_prompt_style(prompt: str) -> str:
    """
    Detect the predominant style of a prompt

    Args:
        prompt: The prompt text to analyze

    Returns:
        Detected style category
    """
    prompt_lower = prompt.lower()

    # Check for artistic style indicators
    if any(term in prompt_lower for term in ["oil painting", "watercolor", "acrylic", "painting"]):
        return "traditional_art"

    if any(term in prompt_lower for term in ["digital art", "digital painting", "concept art"]):
        return "digital_art"

    if any(term in prompt_lower for term in ["photo", "photograph", "photorealistic", "realistic"]):
        return "photorealistic"

    if any(term in prompt_lower for term in ["3d", "render", "octane", "blender", "cinema4d"]):
        return "3d_render"

    if any(term in prompt_lower for term in ["anime", "manga", "japanese", "cartoon"]):
        return "anime"

    if any(term in prompt_lower for term in ["sketch", "drawing", "line art", "illustration"]):
        return "illustration"

    # Default if no specific style detected
    return "general"
