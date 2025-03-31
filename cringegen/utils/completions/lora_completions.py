"""
LoRA completion utilities for shell auto-completion.
"""

import os
from typing import Dict, List, Optional, Tuple

from ...utils.logger import get_logger
from ...utils.lora_metadata import (
    analyze_lora_type,
    analyze_multiple_loras,
    extract_lora_metadata,
    get_lora_path,
    get_lora_tag_frequencies,
)

logger = get_logger(__name__)


def get_lora_completions(
    partial_name: str = "",
    include_type_info: bool = True,
    lora_dir: Optional[str] = None,
    confidence_threshold: float = 0.5,
    lora_types: Optional[List[str]] = None,
    max_results: int = 50,
) -> List[Tuple[str, str]]:
    """
    Get LoRA completions with optional type information.

    Args:
        partial_name: Partial LoRA name to filter by
        include_type_info: Whether to include LoRA type information
        lora_dir: Directory containing LoRA files (defaults to ComfyUI LoRA dir)
        confidence_threshold: Minimum confidence for type classification
        lora_types: List of LoRA types to filter by (style, character, concept)
        max_results: Maximum number of results to return

    Returns:
        List of tuples (lora_name, description) for shell completion
    """
    # Use the provided directory or a default path if not specified
    lora_directory = lora_dir
    if not lora_directory:
        try:
            from ...utils.comfy_api import get_lora_directory

            lora_directory = get_lora_directory()
        except ImportError:
            # Fallback to a common location
            lora_directory = os.path.expanduser("~/comfyui/models/loras")

    if not lora_directory or not os.path.isdir(lora_directory):
        logger.warning(f"LoRA directory not found: {lora_directory}")
        return []

    completions = []

    try:
        # Get all LoRA files
        lora_files = []
        for root, _, files in os.walk(lora_directory):
            for file in files:
                if file.endswith(".safetensors") and partial_name.lower() in file.lower():
                    lora_path = os.path.join(root, file)
                    # Strip file extension for completion
                    lora_name = os.path.splitext(file)[0]
                    lora_files.append((lora_name, lora_path))

        # Sort by name and limit results
        lora_files = sorted(lora_files, key=lambda x: x[0])

        # Add type information if requested
        if include_type_info:
            filtered_files = []
            for lora_name, lora_path in lora_files:
                try:
                    # Check if file is valid (not empty)
                    if os.path.getsize(lora_path) < 1024:  # Skip files smaller than 1KB
                        filtered_files.append((lora_name, lora_path, "unknown"))
                        continue

                    analysis = analyze_lora_type(lora_path)
                    if analysis["confidence"] >= confidence_threshold:
                        lora_type = analysis["type"]

                        # Filter by type if specified
                        if lora_types and lora_type not in lora_types:
                            continue

                        filtered_files.append((lora_name, lora_path, lora_type))
                    elif not lora_types:  # Include if no type filter and below confidence
                        filtered_files.append((lora_name, lora_path, "unknown"))
                except Exception as e:
                    logger.error(f"Error analyzing LoRA {lora_name}: {e}")
                    if not lora_types:  # Include if no type filter
                        filtered_files.append((lora_name, lora_path, "unknown"))

            # Apply limit after filtering
            filtered_files = filtered_files[:max_results]

            # Create completions with type info
            for lora_name, _, lora_type in filtered_files:
                if lora_type != "unknown":
                    completions.append((lora_name, f"{lora_type.capitalize()} LoRA"))
                else:
                    completions.append((lora_name, "LoRA"))
        else:
            # If no type info requested, limit first and then create completions
            lora_files = lora_files[:max_results]
            completions = [(name, "LoRA") for name, _ in lora_files]

    except Exception as e:
        logger.error(f"Error generating LoRA completions: {e}")

    return completions


def get_lora_types(lora_dir: Optional[str] = None, min_confidence: float = 0.6) -> Dict[str, str]:
    """
    Get a dictionary mapping LoRA names to their types.

    Args:
        lora_dir: Directory containing LoRA files
        min_confidence: Minimum confidence threshold for classification

    Returns:
        Dictionary mapping LoRA names to types (style, character, concept)
    """
    # Use the provided directory or a default path if not specified
    lora_directory = lora_dir
    if not lora_directory:
        try:
            from ...utils.comfy_api import get_lora_directory

            lora_directory = get_lora_directory()
        except ImportError:
            # Fallback to a common location
            lora_directory = os.path.expanduser("~/comfyui/models/loras")

    if not lora_directory or not os.path.isdir(lora_directory):
        logger.warning(f"LoRA directory not found: {lora_directory}")
        return {}

    lora_types = {}

    try:
        # Analyze all LoRAs
        analyses = analyze_multiple_loras(lora_directory)

        # Filter by confidence and create mapping
        for lora_name, analysis in analyses.items():
            if analysis["confidence"] >= min_confidence:
                # Strip file extension if present
                name = os.path.splitext(lora_name)[0]
                lora_types[name] = analysis["type"]

    except Exception as e:
        logger.error(f"Error analyzing LoRAs: {e}")

    return lora_types


def get_lora_keywords(
    lora_name: str, top_n: int = 10, include_activation_text: bool = True
) -> List[str]:
    """
    Get keywords for a LoRA, including top tags and activation text.

    Args:
        lora_name: Name of the LoRA
        top_n: Number of top tags to include
        include_activation_text: Whether to include activation text

    Returns:
        List of keywords for the LoRA
    """
    keywords = []

    try:
        # Get LoRA path
        lora_path = get_lora_path(lora_name)
        if not lora_path:
            return keywords

        # Extract metadata
        metadata = extract_lora_metadata(lora_path)

        # Add activation text if requested
        if include_activation_text and metadata.get("activation_text"):
            keywords.extend(metadata["activation_text"])

        # Add top tags
        tag_frequencies = get_lora_tag_frequencies(lora_name)
        if tag_frequencies:
            # Sort by frequency and take top_n
            sorted_tags = sorted(tag_frequencies.items(), key=lambda x: x[1], reverse=True)
            top_tags = [tag for tag, _ in sorted_tags[:top_n]]
            keywords.extend(top_tags)

    except Exception as e:
        logger.error(f"Error getting LoRA keywords for {lora_name}: {e}")

    return keywords
