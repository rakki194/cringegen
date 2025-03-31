"""
Functions for fuzzy matching and autocompletion of LoRA metadata tags
"""

import difflib
import os
import re
import sqlite3
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Get logger
from ...utils.logger import get_logger

logger = get_logger(__name__)

# Import from extractor
from .extractor import DB_PATH, get_lora_tag_frequencies, initialize_db


def fuzzy_match(query: str, choices: List[str], cutoff: float = 0.6) -> List[Tuple[str, float]]:
    """
    Perform fuzzy matching on a list of choices

    Args:
        query: The string to match
        choices: List of choices to match against
        cutoff: Minimum similarity score (0-1)

    Returns:
        List of tuples (match, score) sorted by score
    """
    if not query or not choices:
        return []

    # Use difflib's get_close_matches as a base
    matches = difflib.get_close_matches(query, choices, n=20, cutoff=cutoff)

    # Get scores for each match
    scored_matches = []
    for match in matches:
        score = difflib.SequenceMatcher(None, query, match).ratio()
        scored_matches.append((match, score))

    # Sort by score
    scored_matches.sort(key=lambda x: x[1], reverse=True)

    return scored_matches


def match_tags(
    query: str, min_count: int = 1, max_results: int = 10, cutoff: float = 0.5
) -> List[Tuple[str, float, int]]:
    """
    Match tags from the database using fuzzy matching

    Args:
        query: The tag prefix or pattern to match
        min_count: Minimum tag frequency
        max_results: Maximum number of results to return
        cutoff: Minimum similarity score (0-1)

    Returns:
        List of tuples (tag, score, frequency) sorted by score and frequency
    """
    # Initialize database if it doesn't exist
    initialize_db()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all tags with count >= min_count
    cursor.execute("SELECT tag, count FROM tags WHERE count >= ? ORDER BY count DESC", (min_count,))
    tag_counts = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    # If query is empty, return most common tags
    if not query:
        return [
            (tag, 1.0, count)
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[
                :max_results
            ]
        ]

    # Try exact prefix matching first (case insensitive)
    prefix_matches = []
    query_lower = query.lower()
    for tag, count in tag_counts.items():
        if tag.lower().startswith(query_lower):
            # Calculate a score based on how close the length is to the query
            # Exact matches get a score of 1.0
            if tag.lower() == query_lower:
                score = 1.0
            else:
                # Score decreases as tag gets longer compared to query
                score = 0.9 * (len(query) / len(tag))
            prefix_matches.append((tag, score, count))

    # Sort prefix matches by score and count
    prefix_matches.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # If we have enough prefix matches, return them
    if len(prefix_matches) >= max_results:
        return prefix_matches[:max_results]

    # Otherwise, use fuzzy matching for additional matches
    remaining = max_results - len(prefix_matches)

    # Get tags that weren't matched by prefix
    unmatched_tags = [
        tag for tag in tag_counts.keys() if not any(tag == m[0] for m in prefix_matches)
    ]

    # Perform fuzzy matching
    fuzzy_matches = fuzzy_match(query, unmatched_tags, cutoff)

    # Convert to the same format as prefix matches
    fuzzy_match_tuples = [(tag, score, tag_counts[tag]) for tag, score in fuzzy_matches]

    # Combine results and sort
    all_matches = prefix_matches + fuzzy_match_tuples
    all_matches.sort(key=lambda x: (x[1], x[2]), reverse=True)

    return all_matches[:max_results]


def get_tag_suggestions(input_tags: List[str], max_results: int = 5) -> List[str]:
    """
    Suggest additional tags based on co-occurrence with input tags

    Args:
        input_tags: List of tags already selected
        max_results: Maximum number of suggestions to return

    Returns:
        List of suggested tags
    """
    if not input_tags:
        return []

    # Initialize database if it doesn't exist
    initialize_db()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find LoRAs that have the input tags
    placeholders = ", ".join(["?"] * len(input_tags))
    query = f"""
    SELECT DISTINCT l.id
    FROM loras l
    JOIN lora_tags lt ON l.id = lt.lora_id
    JOIN tags t ON lt.tag_id = t.id
    WHERE t.tag IN ({placeholders})
    """
    cursor.execute(query, input_tags)
    lora_ids = [row[0] for row in cursor.fetchall()]

    if not lora_ids:
        conn.close()
        return []

    # Find tags that co-occur with the input tags
    placeholders = ", ".join(["?"] * len(lora_ids))
    query = f"""
    SELECT t.tag, COUNT(DISTINCT lt.lora_id) as freq
    FROM tags t
    JOIN lora_tags lt ON t.id = lt.tag_id
    WHERE lt.lora_id IN ({placeholders})
    AND t.tag NOT IN ({', '.join(['?'] * len(input_tags))})
    GROUP BY t.tag
    ORDER BY freq DESC
    LIMIT ?
    """

    # Combine lora_ids and input_tags for the query
    params = lora_ids + input_tags + [max_results]
    cursor.execute(query, params)

    # Get results
    suggestions = [row[0] for row in cursor.fetchall()]

    conn.close()

    return suggestions


def get_activation_text(lora_name: str) -> List[str]:
    """
    Get activation text for a specific LoRA

    Args:
        lora_name: Name of the LoRA

    Returns:
        List of activation text strings
    """
    # Initialize database if it doesn't exist
    initialize_db()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Debug logging
    logger.debug(f"Looking for activation text for LoRA: {lora_name}")

    # Try to find the LoRA
    query = "SELECT activation_text, path FROM loras WHERE name = ? OR lower(name) = lower(?)"
    logger.debug(f"Executing SQL query: {query} with params ({lora_name}, {lora_name})")

    cursor.execute(query, (lora_name, lora_name))
    result = cursor.fetchone()

    if result:
        logger.debug(f"Found result: {result}")
    else:
        logger.debug(f"No result found for {lora_name}")
        # Try with file extension variations
        if not lora_name.endswith(".safetensors"):
            lora_with_ext = f"{lora_name}.safetensors"
            logger.debug(f"Trying with extension: {lora_with_ext}")
            cursor.execute(query, (lora_with_ext, lora_with_ext))
            result = cursor.fetchone()
            if result:
                logger.debug(f"Found result with extension: {result}")

    conn.close()

    if result and result[0]:
        import json

        return json.loads(result[0])

    return []


def get_trigger_phrases(lora_name: str) -> List[str]:
    """
    Get trigger phrases for a specific LoRA

    Args:
        lora_name: Name of the LoRA

    Returns:
        List of trigger phrases
    """
    # Initialize database if it doesn't exist
    initialize_db()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Debug logging
    logger.debug(f"Looking for trigger phrases for LoRA: {lora_name}")

    # Try to find the LoRA
    query = "SELECT trigger_phrases, path FROM loras WHERE name = ? OR lower(name) = lower(?)"
    logger.debug(f"Executing SQL query: {query} with params ({lora_name}, {lora_name})")

    cursor.execute(query, (lora_name, lora_name))
    result = cursor.fetchone()

    if result:
        logger.debug(f"Found result: {result}")
    else:
        logger.debug(f"No result found for {lora_name}")
        # Try with file extension variations
        if not lora_name.endswith(".safetensors"):
            lora_with_ext = f"{lora_name}.safetensors"
            logger.debug(f"Trying with extension: {lora_with_ext}")
            cursor.execute(query, (lora_with_ext, lora_with_ext))
            result = cursor.fetchone()
            if result:
                logger.debug(f"Found result with extension: {result}")

    conn.close()

    if result and result[0]:
        import json

        return json.loads(result[0])

    return []
