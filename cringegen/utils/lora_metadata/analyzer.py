"""
LoRA Analyzer Module - Detects LoRA types and provides recommendations.

This module provides functionality to analyze LoRA files and classify them as:
- Style LoRAs: Focus on artistic styles
- Character LoRAs: Focus on specific characters
- Concept LoRAs: Focus on specific themes, actions, or scenarios
- Kink LoRAs: Focus on specific fetishes or kinks
"""

import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ...data.lora import (
    STYLE_KEYWORDS,
    CHARACTER_KEYWORDS,
    KINK_KEYWORDS,
    CONCEPT_KEYWORDS,
    SPECIFIC_KINK_LORAS,
    SPECIFIC_CHARACTER_LORAS,
    SPECIFIC_CONCEPT_LORAS,
    STYLE_PATTERNS,
    CHARACTER_PATTERNS,
    CONCEPT_PATTERNS,
)
from ...utils.logger import get_logger
from .extractor import DB_PATH, extract_lora_metadata, get_lora_path

logger = get_logger(__name__)

def analyze_lora_type(lora_path_or_name: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Analyze a LoRA file and determine its type (style, character, or concept).

    Args:
        lora_path_or_name: Path to the LoRA file or its name (basename)
        use_cache: Whether to use cached results (if available)

    Returns:
        Dictionary containing analysis results with structure:
        {
            "name": str,             # LoRA name
            "path": str,             # Full path to LoRA file
            "type": str,             # "style", "character", "concept", or "unknown"
            "confidence": float,     # Confidence level (0.0-1.0)
            "evidence": List[str],   # Reasons for the classification
            "recommendations": Dict   # Type-specific recommendations
        }
    """
    # Initialize result structure
    result = {
        "name": os.path.basename(lora_path_or_name),
        "path": lora_path_or_name,
        "type": "unknown",
        "confidence": 0.0,
        "evidence": [],
        "recommendations": {},
    }

    # Get full path if name was provided
    if not os.path.exists(lora_path_or_name):
        lora_path = get_lora_path(lora_path_or_name)
        if not lora_path:
            logger.error(f"LoRA file not found: {lora_path_or_name}")
            return result
        result["path"] = lora_path

    # Check cache if enabled
    if use_cache:
        cached_result = _get_cached_analysis(result["path"])
        if cached_result:
            logger.debug(f"Using cached analysis for {result['name']}")
            return cached_result

    # Extract metadata from LoRA file
    metadata = extract_lora_metadata(result["path"])

    # Analyze metadata to determine type
    lora_type, confidence, evidence = _determine_lora_type(metadata)

    result["type"] = lora_type
    result["confidence"] = confidence
    result["evidence"] = evidence

    # Generate type-specific recommendations
    result["recommendations"] = _generate_recommendations(lora_type, metadata)

    # Cache the result
    _cache_analysis_result(result)

    return result

def _determine_lora_type(metadata: Dict[str, Any]) -> Tuple[str, float, List[str]]:
    """Determine the type of a LoRA based on its metadata

    Args:
        metadata: The metadata to analyze

    Returns:
        A tuple containing the LoRA type, confidence, and evidence for the determination
    """
    lora_name = metadata.get("name", "").lower()
    lora_filename = metadata.get("filename", "").lower()
    lora_basename = os.path.basename(lora_filename).lower() if lora_filename else ""
    
    # First check for explicit kink-related keywords in the filename or name
    kink_terms = ["fetish", "kink", "fart", "bdsm", "bondage", "latex", "paw_fetish", "foot_fetish"]
    for term in kink_terms:
        if term in lora_basename or term in lora_name:
            evidence = [f"Matched kink-related term in name: {term}"]
            return "kink", 1.0, evidence
            
    # Check for specific kink loras by name or basename
    for kink_lora in SPECIFIC_KINK_LORAS:
        if (kink_lora in lora_basename or 
            kink_lora in lora_name or 
            kink_lora in lora_filename):
            evidence = [f"Matched specific kink LoRA name: {kink_lora}"]
            return "kink", 1.0, evidence
    
    # Check for specific character loras by name
    for character_lora in SPECIFIC_CHARACTER_LORAS:
        if character_lora in lora_basename or character_lora in lora_name:
            evidence = [f"Matched specific character LoRA name: {character_lora}"]
            return "character", 1.0, evidence
    
    # Check for specific concept loras by name
    for concept_lora in SPECIFIC_CONCEPT_LORAS:
        if concept_lora in lora_basename or concept_lora in lora_name:
            evidence = [f"Matched specific concept LoRA name: {concept_lora}"]
            return "concept", 1.0, evidence
    
    # Extract keywords from metadata
    keywords = set()
    evidence = []

    # Process name and filename
    if lora_name:
        keywords.update(re.findall(r"\b\w+\b", lora_name.replace("_", " ").replace("-", " ")))
        evidence.append(f"Name: {lora_name}")
    if lora_basename:
        keywords.update(
            re.findall(r"\b\w+\b", lora_basename.replace("_", " ").replace("-", " "))
        )
        evidence.append(f"Basename: {lora_basename}")

    # Process activation text if available
    activation_text = metadata.get("activation_text", "")
    if activation_text:
        if isinstance(activation_text, str):
            keywords.update(re.findall(r"\b\w+\b", activation_text.lower()))
            evidence.append(f"Activation text: {activation_text}")
        elif isinstance(activation_text, list):
            # Handle activation_text as a list
            for text in activation_text:
                if isinstance(text, str):
                    keywords.update(re.findall(r"\b\w+\b", text.lower()))
            evidence.append(f"Activation text list: {len(activation_text)} items")

    # Process tags
    tags = metadata.get("tags", [])
    if tags:
        for tag in tags:
            if isinstance(tag, str):
                keywords.update(re.findall(r"\b\w+\b", tag.lower().replace("_", " ")))
            else:
                # Handle non-string tags (convert to string if possible)
                try:
                    tag_str = str(tag)
                    keywords.update(re.findall(r"\b\w+\b", tag_str.lower().replace("_", " ")))
                except:
                    pass
        evidence.append(f"Tags: {', '.join(str(t) for t in tags[:5])}{'...' if len(tags) > 5 else ''}")

    # Process description
    description = metadata.get("description", "")
    if description:
        if isinstance(description, str):
            keywords.update(re.findall(r"\b\w+\b", description.lower()))
            evidence.append(f"Description excerpt: {description[:100]}...")
        else:
            # Handle non-string description
            try:
                desc_str = str(description)
                keywords.update(re.findall(r"\b\w+\b", desc_str.lower()))
                evidence.append(f"Description excerpt (converted): {desc_str[:100]}...")
            except:
                pass

    # Count keyword matches for each type
    style_matches = len(keywords.intersection(STYLE_KEYWORDS))
    character_matches = len(keywords.intersection(CHARACTER_KEYWORDS))
    concept_matches = len(keywords.intersection(CONCEPT_KEYWORDS))
    kink_matches = len(keywords.intersection(KINK_KEYWORDS))

    # Check for strongest match
    total_matches = style_matches + character_matches + concept_matches + kink_matches
    if total_matches == 0:
        # If no matches, fall back to pattern recognition in the name/filename
        pattern_evidence = []
        
        # Check for kink-related patterns first
        kink_patterns = [
            (r"(.+)_(?:fetish|kink)", r"(?:fetish|kink)_(.+)", r"(.+)-(?:fetish|kink)", r"(?:fetish|kink)-(.+)")
        ]
        for pattern_group in kink_patterns:
            for pattern in pattern_group:
                if lora_name and re.match(pattern, lora_name):
                    pattern_evidence.append(f"Kink pattern in name: {pattern}")
                if lora_basename and re.match(pattern, lora_basename):
                    pattern_evidence.append(f"Kink pattern in filename: {pattern}")
            if pattern_evidence:
                return "kink", 0.7, pattern_evidence
        
        # Check for character-based patterns
        character_patterns = [
            (r"(.+)_(?:character|person|oc)", r"(?:character|person|oc)_(.+)"),
            (r"(.+)-(?:character|person|oc)", r"(?:character|person|oc)-(.+)"),
        ]
        for pattern_group in character_patterns:
            for pattern in pattern_group:
                if lora_name and re.match(pattern, lora_name):
                    pattern_evidence.append(f"Character pattern in name: {pattern}")
                if lora_basename and re.match(pattern, lora_basename):
                    pattern_evidence.append(f"Character pattern in filename: {pattern}")
            if pattern_evidence:
                return "character", 0.7, pattern_evidence

        # Check for style-based patterns
        style_patterns = [
            (r"(.+)_(?:style|artist)", r"(?:style|artist)_(.+)"),
            (r"(.+)-(?:style|artist)", r"(?:style|artist)-(.+)"),
        ]
        for pattern_group in style_patterns:
            for pattern in pattern_group:
                if lora_name and re.match(pattern, lora_name):
                    pattern_evidence.append(f"Style pattern in name: {pattern}")
                if lora_basename and re.match(pattern, lora_basename):
                    pattern_evidence.append(f"Style pattern in filename: {pattern}")
            if pattern_evidence:
                return "style", 0.7, pattern_evidence

        # Check for concept-based patterns
        concept_patterns = [
            (r"(.+)_(?:fetish|action|pose|activity|concept)", r"(?:fetish|action|pose|activity|concept)_(.+)"),
            (r"(.+)-(?:fetish|action|pose|activity|concept)", r"(?:fetish|action|pose|activity|concept)-(.+)"),
        ]
        for pattern_group in concept_patterns:
            for pattern in pattern_group:
                if lora_name and re.match(pattern, lora_name):
                    pattern_evidence.append(f"Concept pattern in name: {pattern}")
                if lora_basename and re.match(pattern, lora_basename):
                    pattern_evidence.append(f"Concept pattern in filename: {pattern}")
            if pattern_evidence:
                return "concept", 0.7, pattern_evidence

        # If no patterns matched either, default to concept with low confidence
        return "concept", 0.3, ["No clear classification indicators, defaulting to concept"]

    # Add evidence of keyword matches
    if style_matches > 0:
        evidence.append(f"Style keywords: {style_matches}")
    if character_matches > 0:
        evidence.append(f"Character keywords: {character_matches}")
    if concept_matches > 0:
        evidence.append(f"Concept keywords: {concept_matches}")
    if kink_matches > 0:
        evidence.append(f"Kink keywords: {kink_matches}")

    # Calculate confidence
    confidence = max(style_matches, character_matches, concept_matches, kink_matches) / total_matches
    confidence = min(max(confidence, 0.3), 1.0)  # Clamp between 0.3 and 1.0

    # Determine type based on most matches
    if kink_matches >= max(style_matches, character_matches, concept_matches):
        return "kink", confidence, evidence
    elif style_matches >= max(character_matches, concept_matches):
        return "style", confidence, evidence
    elif character_matches >= concept_matches:
        return "character", confidence, evidence
    else:
        return "concept", confidence, evidence

def _generate_recommendations(lora_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate recommendations for a LoRA

    Args:
        lora_type: The type of the LoRA
        metadata: The metadata for the LoRA

    Returns:
        A dictionary of recommendations
    """
    recommendations = {
        "usage_examples": [],
        "complementary_loras": [],
        "tips": [],
    }

    lora_name = metadata.get("name", "")
    lora_filename = metadata.get("filename", "")
    lora_basename = os.path.basename(lora_filename) if lora_filename else ""
    
    # Common tips for all LoRA types
    recommendations["tips"].append(
        "Start with a low strength (0.5-0.7) and adjust as needed for optimal results."
    )
    recommendations["tips"].append(
        "Use the 'resolve-path' command to verify the LoRA path is recognized by ComfyUI."
    )

    if lora_type == "style":
        # Style LoRA recommendations
        recommendations["usage_examples"].append(
            f"--lora {lora_basename} --lora-strength 0.7 --no-art-style"
        )
        recommendations["usage_examples"].append(
            f"--checkpoint CHECKPOINT_NAME --lora {lora_basename} --lora-strength 0.8"
        )
        recommendations["tips"].append(
            "Style LoRAs work best when the 'no-art-style' option is used to prevent conflicting styles."
        )
        recommendations["tips"].append(
            "Try different checkpoints to see which ones work best with this style."
        )
        
        # Suggest combining with character or concept LoRAs
        recommendations["complementary_loras"].append(
            {"type": "character", "suggestion": "Try combining with a character LoRA for consistent styling"}
        )
        recommendations["complementary_loras"].append(
            {"type": "concept", "suggestion": "Concept LoRAs can enhance specific elements in this style"}
        )

    elif lora_type == "character":
        # Character LoRA recommendations
        recommendations["usage_examples"].append(
            f"--lora {lora_basename} --lora-strength 0.8"
        )
        recommendations["usage_examples"].append(
            f"--checkpoint CHECKPOINT_NAME --lora {lora_basename} --lora-strength 0.9"
        )
        recommendations["tips"].append(
            "Character LoRAs generally work best at higher strengths (0.8-1.0)."
        )
        recommendations["tips"].append(
            "Be specific in your prompt about the character's poses, expressions, and setting."
        )
        
        # Suggest combining with style or concept LoRAs
        recommendations["complementary_loras"].append(
            {"type": "style", "suggestion": "Style LoRAs can give this character a unique artistic look"}
        )
        recommendations["complementary_loras"].append(
            {"type": "concept", "suggestion": "Add concept LoRAs for specific poses, activities, or settings"}
        )

    elif lora_type == "kink":
        # Kink LoRA recommendations
        recommendations["usage_examples"].append(
            f"--lora {lora_basename} --lora-strength 0.7 --intensity explicit"
        )
        recommendations["usage_examples"].append(
            f"--checkpoint noobaiXLVpredv10.safetensors --lora {lora_basename} --lora-strength 0.8"
        )
        recommendations["tips"].append(
            "Kink LoRAs work best with the noobai checkpoint and explicit intensity settings."
        )
        recommendations["tips"].append(
            "Use specific kink-related terms in your prompt to enhance the effect."
        )
        
        # Special recommendation for fart fetish
        if "fart" in lora_basename.lower() or "fart_fetish" in lora_basename.lower():
            recommendations["tips"].append(
                "For the fart_fetish LoRA, adding 'gas cloud', 'visible fart', or 'embarrassed expression' to your prompt enhances results."
            )
            recommendations["tips"].append(
                "The noobaiXLVpredv10.safetensors checkpoint works particularly well with this kink LoRA."
            )
        
        # Suggest combining with character LoRAs
        recommendations["complementary_loras"].append(
            {"type": "character", "suggestion": "Character LoRAs can be combined for specific kink scenarios"}
        )

    else:  # concept
        # Concept LoRA recommendations
        recommendations["usage_examples"].append(
            f"--lora {lora_basename} --lora-strength 0.6"
        )
        recommendations["usage_examples"].append(
            f"--checkpoint CHECKPOINT_NAME --lora {lora_basename} --lora-strength 0.7"
        )
        recommendations["tips"].append(
            "Concept LoRAs typically work well at moderate strengths (0.5-0.8)."
        )
        recommendations["tips"].append(
            "Be explicit in your prompt about the specific concept elements you want to emphasize."
        )
        
        # Check for specific concept types and provide tailored recommendations
        for concept in ["realistic", "fetish", "holding_object", "space", "landscape"]:
            if concept in lora_basename.lower():
                recommendations["tips"].append(
                    f"This is a {concept}-focused concept LoRA. Emphasize {concept}-related terms in your prompt."
                )
        
        # Suggest combining with other LoRA types
        recommendations["complementary_loras"].append(
            {"type": "style", "suggestion": "Style LoRAs can enhance the overall aesthetic of this concept"}
        )
        recommendations["complementary_loras"].append(
            {"type": "character", "suggestion": "Add character LoRAs to place specific characters in this concept"}
        )

    return recommendations

def _get_cached_analysis(lora_path: str) -> Optional[Dict[str, Any]]:
    """
    Get cached analysis for a LoRA.

    Args:
        lora_path: Path to the LoRA file

    Returns:
        Cached analysis result or None if no cache found
    """
    try:
        # Skip if DB_PATH doesn't exist
        if not os.path.exists(DB_PATH):
            return None

        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get the cached analysis
        cursor.execute(
            "SELECT analysis FROM lora_analysis WHERE lora_path = ?", (lora_path,)
        )
        result = cursor.fetchone()

        # Close connection
        conn.close()

        if result:
            # Parse JSON and return
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                logger.warning(f"Error parsing cached analysis for {lora_path}")
                return None
        return None
    except Exception as e:
        logger.warning(f"Error retrieving cached analysis: {e}")
        return None

def _ensure_db_schema():
    """
    Ensure the database schema is correct.
    Creates or resets the table if needed.
    """
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if the table exists with the correct schema
        cursor.execute("PRAGMA table_info(lora_analysis)")
        columns = cursor.fetchall()
        
        # Column names from the schema
        expected_columns = {'lora_path', 'analysis', 'updated_at'}
        
        # Extract column names from the result
        existing_columns = {col[1] for col in columns}
        
        # If the table doesn't exist or has incorrect columns, recreate it
        if not columns or existing_columns != expected_columns:
            # Drop the table if it exists
            cursor.execute("DROP TABLE IF EXISTS lora_analysis")
            
            # Create the table with the correct schema
            cursor.execute("""
                CREATE TABLE lora_analysis (
                    lora_path TEXT PRIMARY KEY,
                    analysis TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Created or reset lora_analysis table with correct schema")
        
        conn.close()
    except Exception as e:
        logger.warning(f"Error ensuring DB schema: {e}")

# Ensure the DB schema is correct on module import
_ensure_db_schema()

def _cache_analysis_result(result: Dict[str, Any]) -> None:
    """Cache LoRA analysis result in database

    Args:
        result: Analysis result to cache
    """
    try:
        # Don't try to cache if there's no path
        if not result.get("path"):
            return

        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS lora_analysis (
                lora_path TEXT PRIMARY KEY,
                analysis TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Convert result to JSON
        analysis_json = json.dumps(result)

        # Insert or update analysis
        cursor.execute(
            """
            INSERT OR REPLACE INTO lora_analysis (lora_path, analysis, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (result["path"], analysis_json),
        )

        # Commit changes
        conn.commit()
        conn.close()
    except Exception as e:
        # Log error but don't crash
        logger.warning(f"Error caching analysis result: {e}")
        # Continue execution without caching

def analyze_multiple_loras(
    lora_dir: str, pattern: str = "*.safetensors", force_refresh: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple LoRAs in a directory and determine their types.

    Args:
        lora_dir: Directory containing LoRA files
        pattern: Glob pattern for finding LoRA files
        force_refresh: Whether to force refresh analysis (ignore cache)

    Returns:
        Dictionary mapping LoRA names to their analysis results
    """
    results = {}

    # Ensure directory exists
    if not os.path.isdir(lora_dir):
        logger.error(f"Directory not found: {lora_dir}")
        return results

    # Find all LoRA files
    lora_files = list(Path(lora_dir).glob(pattern))
    logger.info(f"Found {len(lora_files)} LoRA files in {lora_dir}")

    # Analyze each LoRA
    for lora_file in lora_files:
        try:
            analysis = analyze_lora_type(str(lora_file), not force_refresh)
            results[analysis["name"]] = analysis
        except Exception as e:
            logger.error(f"Error analyzing LoRA {lora_file}: {e}")

    return results

def get_loras_by_type(
    lora_dir: str, lora_type: str, min_confidence: float = 0.5, pattern: str = "*.safetensors"
) -> List[Dict[str, Any]]:
    """
    Get all LoRAs of a specific type in a directory.

    Args:
        lora_dir: Directory containing LoRA files
        lora_type: Type of LoRAs to find (style, character, concept)
        min_confidence: Minimum confidence threshold for type detection
        pattern: Glob pattern for finding LoRA files

    Returns:
        List of LoRA analysis results of the specified type
    """
    # Analyze all LoRAs
    all_loras = analyze_multiple_loras(lora_dir, pattern)

    # Filter by type and confidence
    return [
        lora
        for lora in all_loras.values()
        if lora["type"] == lora_type and lora["confidence"] >= min_confidence
    ]

def suggest_lora_combinations(
    lora_path_or_name: str, lora_dir: str, max_suggestions: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Suggest LoRA combinations based on a primary LoRA.

    Args:
        lora_path_or_name: Path or name of the primary LoRA
        lora_dir: Directory containing LoRA files
        max_suggestions: Maximum number of suggestions per category

    Returns:
        Dictionary mapping suggestion types to lists of LoRA analysis results
    """
    # Analyze the primary LoRA
    primary_analysis = analyze_lora_type(lora_path_or_name)
    primary_type = primary_analysis["type"]

    # Initialize suggestion categories
    suggestions = {
        "complementary": [],  # LoRAs that work well with the primary
        "similar": [],  # Similar LoRAs (same type)
        "contrasting": [],  # LoRAs of different types that could create interesting results
    }

    # Get all LoRAs
    all_loras = analyze_multiple_loras(lora_dir)

    # Filter out the primary LoRA itself
    other_loras = [lora for lora in all_loras.values() if lora["path"] != primary_analysis["path"]]

    # Generate suggestions based on primary LoRA type
    if primary_type == "style":
        # For style LoRAs, suggest character and concept LoRAs
        suggestions["complementary"] = [
            lora
            for lora in other_loras
            if lora["type"] == "character" and lora["confidence"] >= 0.6
        ][:max_suggestions]

        suggestions["similar"] = [
            lora for lora in other_loras if lora["type"] == "style" and lora["confidence"] >= 0.6
        ][:max_suggestions]

        suggestions["contrasting"] = [
            lora for lora in other_loras if lora["type"] == "concept" and lora["confidence"] >= 0.6
        ][:max_suggestions]

    elif primary_type == "character":
        # For character LoRAs, suggest style and concept LoRAs
        suggestions["complementary"] = [
            lora for lora in other_loras if lora["type"] == "style" and lora["confidence"] >= 0.6
        ][:max_suggestions]

        suggestions["similar"] = [
            lora
            for lora in other_loras
            if lora["type"] == "character" and lora["confidence"] >= 0.6
        ][:max_suggestions]

        suggestions["contrasting"] = [
            lora for lora in other_loras if lora["type"] == "concept" and lora["confidence"] >= 0.6
        ][:max_suggestions]

    elif primary_type == "concept":
        # For concept LoRAs, suggest character and style LoRAs
        suggestions["complementary"] = [
            lora
            for lora in other_loras
            if lora["type"] == "character" and lora["confidence"] >= 0.6
        ][:max_suggestions]

        suggestions["similar"] = [
            lora for lora in other_loras if lora["type"] == "concept" and lora["confidence"] >= 0.6
        ][:max_suggestions]

        suggestions["contrasting"] = [
            lora for lora in other_loras if lora["type"] == "style" and lora["confidence"] >= 0.6
        ][:max_suggestions]

    return suggestions
