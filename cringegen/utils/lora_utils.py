"""
LoRA Utility Functions for CringeGen

This module provides utility functions for working with the LoRA taxonomy system.
It includes functions for selecting, filtering, and combining LoRAs in an optimal way.
"""

import random
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import re

from cringegen.data.lora_taxonomy import (
    LoRACategory, 
    LoRASubcategory,
    LoRAMetadata,
    ALL_LORAS,
    get_loras_by_category,
    get_loras_by_tags,
    suggest_lora_strength,
    categorize_lora_by_filename,
    get_compatible_loras
)

def select_random_lora(
    category: Optional[LoRACategory] = None, 
    tags: Optional[List[str]] = None,
    exclude_nsfw: bool = True
) -> str:
    """
    Select a random LoRA, optionally filtered by category and tags.
    
    Args:
        category: Optional category to filter by
        tags: Optional list of tags to filter by
        exclude_nsfw: Whether to exclude NSFW LoRAs
        
    Returns:
        The name of a randomly selected LoRA
    """
    # Start with all LoRAs
    filtered_loras = {}
    
    # Filter by category if provided
    if category:
        filtered_loras = get_loras_by_category(category)
    else:
        filtered_loras = ALL_LORAS
    
    # Filter by tags if provided
    if tags:
        tag_filtered = get_loras_by_tags(tags, match_all=False)
        filtered_loras = {name: lora for name, lora in filtered_loras.items() if name in tag_filtered}
    
    # Remove NSFW LoRAs if requested
    if exclude_nsfw:
        filtered_loras = {name: lora for name, lora in filtered_loras.items() 
                          if not lora.nsfw and not lora.excluded_from_random}
    
    if not filtered_loras:
        return None
    
    # Randomly select a LoRA
    return random.choice(list(filtered_loras.keys()))

def create_balanced_lora_set(
    style_count: int = 1,
    character_count: int = 1,
    concept_count: int = 1,
    pose_count: int = 0,
    exclude_nsfw: bool = True
) -> Dict[str, float]:
    """
    Create a balanced set of LoRAs with appropriate weights.
    
    Args:
        style_count: Number of style LoRAs to include
        character_count: Number of character LoRAs to include
        concept_count: Number of concept LoRAs to include
        pose_count: Number of pose LoRAs to include
        exclude_nsfw: Whether to exclude NSFW LoRAs
        
    Returns:
        A dictionary mapping LoRA names to their suggested strengths
    """
    result = {}
    
    # Add style LoRAs
    style_loras = []
    for _ in range(style_count):
        if not style_loras:
            # First style LoRA is completely random
            lora = select_random_lora(category=LoRACategory.STYLE, exclude_nsfw=exclude_nsfw)
            if lora:
                style_loras.append(lora)
        else:
            # Additional style LoRAs should be compatible with existing ones
            compatible = get_compatible_loras(
                LoRACategory.STYLE, 
                existing_loras=list(result.keys()) + style_loras,
                exclude_nsfw=exclude_nsfw
            )
            if compatible:
                style_loras.append(compatible[0])
    
    # Add character LoRAs
    character_loras = []
    for _ in range(character_count):
        if not character_loras:
            # First character LoRA considers styles already selected
            compatible = get_compatible_loras(
                LoRACategory.CHARACTER, 
                existing_loras=style_loras,
                exclude_nsfw=exclude_nsfw
            )
            if compatible:
                character_loras.append(compatible[0])
            else:
                lora = select_random_lora(category=LoRACategory.CHARACTER, exclude_nsfw=exclude_nsfw)
                if lora:
                    character_loras.append(lora)
        else:
            # Additional character LoRAs should be compatible with everything
            compatible = get_compatible_loras(
                LoRACategory.CHARACTER, 
                existing_loras=list(result.keys()) + style_loras + character_loras,
                exclude_nsfw=exclude_nsfw
            )
            if compatible:
                character_loras.append(compatible[0])
    
    # Add concept LoRAs
    concept_loras = []
    for _ in range(concept_count):
        existing = list(result.keys()) + style_loras + character_loras + concept_loras
        compatible = get_compatible_loras(
            LoRACategory.CONCEPT, 
            existing_loras=existing,
            exclude_nsfw=exclude_nsfw
        )
        if compatible:
            concept_loras.append(compatible[0])
    
    # Add pose LoRAs
    pose_loras = []
    for _ in range(pose_count):
        existing = list(result.keys()) + style_loras + character_loras + concept_loras + pose_loras
        compatible = get_compatible_loras(
            LoRACategory.POSE, 
            existing_loras=existing,
            exclude_nsfw=exclude_nsfw
        )
        if compatible:
            pose_loras.append(compatible[0])
    
    # Combine all selections
    all_selections = style_loras + character_loras + concept_loras + pose_loras
    
    # Assign appropriate strengths
    for lora_name in all_selections:
        if lora_name in ALL_LORAS:
            result[lora_name] = suggest_lora_strength(lora_name)
    
    return result

def extract_lora_info_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract useful information from a LoRA filename.
    
    Args:
        filename: The filename to analyze
        
    Returns:
        A dictionary with extracted information
    """
    info = {
        "filename": filename,
        "category": categorize_lora_by_filename(filename),
        "version": "1.0",
        "steps": None,
        "creator": None,
        "name": filename.split('/')[-1].split('-')[0] if '/' in filename else filename.split('-')[0]
    }
    
    # Extract version
    version_match = re.search(r"v(\d+[a-z]?)", filename)
    if version_match:
        info["version"] = version_match.group(1)
    
    # Extract training steps
    steps_match = re.search(r"s(\d+)", filename)
    if steps_match:
        info["steps"] = int(steps_match.group(1))
    
    # Extract creator if in format creator/name-v1
    if '/' in filename:
        parts = filename.split('/')
        if len(parts) >= 2:
            info["creator"] = parts[-2]
    
    return info

def find_loras_for_prompt(prompt: str) -> Dict[str, float]:
    """
    Analyze a prompt and suggest appropriate LoRAs based on content.
    
    Args:
        prompt: The text prompt to analyze
        
    Returns:
        A dictionary mapping suggested LoRA names to strengths
    """
    result = {}
    
    # Look for LoRA trigger terms in the prompt
    for lora_name, lora in ALL_LORAS.items():
        for trigger in lora.trigger_terms:
            if trigger.lower() in prompt.lower():
                result[lora_name] = suggest_lora_strength(lora_name)
                break
    
    # If no matches found based on triggers, look for style and concept tags
    if not result:
        # Extract potential style keywords
        style_keywords = ["in the style of", "drawn in", "painted in", "art style", 
                         "rendered in", "with a", "aesthetic"]
        
        for keyword in style_keywords:
            if keyword in prompt.lower():
                # Extract the part after the keyword
                parts = prompt.lower().split(keyword)
                if len(parts) > 1:
                    style_description = parts[1].split('.')[0].split(',')[0].strip()
                    # Search LoRA tags that might match
                    for lora_name, lora in ALL_LORAS.items():
                        if lora.category == LoRACategory.STYLE:
                            for tag in lora.tags:
                                if tag.lower() in style_description:
                                    result[lora_name] = suggest_lora_strength(lora_name)
                                    break
    
    # If still no results, suggest a balanced default set
    if not result:
        result = create_balanced_lora_set(style_count=1, character_count=0, concept_count=1)
    
    return result

def generate_lora_prompt_addition(lora_name: str) -> str:
    """
    Generate additional prompt text for a specific LoRA to enhance its effect.
    
    Args:
        lora_name: The name of the LoRA
        
    Returns:
        Additional prompt text to enhance the LoRA's effect
    """
    if lora_name not in ALL_LORAS:
        return ""
    
    lora = ALL_LORAS[lora_name]
    
    if lora.category == LoRACategory.STYLE:
        if lora.trigger_terms:
            return f", {lora.trigger_terms[0]} style"
        return f", {lora.name.lower()} style"
    
    elif lora.category == LoRACategory.CHARACTER:
        if lora.trigger_terms:
            return f", {lora.trigger_terms[0]}"
        return f", {lora.name}"
    
    elif lora.category == LoRACategory.CONCEPT:
        if lora.trigger_terms:
            return f", {lora.trigger_terms[0]}"
        return ""
    
    elif lora.category == LoRACategory.POSE:
        if lora.trigger_terms:
            return f", {lora.trigger_terms[0]}"
        return ""
    
    return ""

def upgrade_legacy_lora_selection(legacy_loras: Dict[str, float]) -> Dict[str, float]:
    """
    Upgrade a legacy LoRA selection to use the new taxonomy system.
    
    Args:
        legacy_loras: A dictionary of legacy LoRA filenames to strengths
        
    Returns:
        An upgraded dictionary with better balanced strengths
    """
    result = {}
    
    for filename, strength in legacy_loras.items():
        # Extract base filename without path and extension
        base_filename = filename.split('/')[-1]
        if '.' in base_filename:
            base_filename = base_filename.split('.')[0]
        
        # Check if we have this LoRA in our taxonomy
        found = False
        for lora_name, lora in ALL_LORAS.items():
            if lora.filename.endswith(base_filename) or base_filename in lora.filename:
                # Use the proper name and optimal strength
                result[lora_name] = suggest_lora_strength(lora_name)
                found = True
                break
        
        if not found:
            # Keep the original but categorize and suggest better strength
            category = categorize_lora_by_filename(base_filename)
            min_strength, max_strength = (0.3, 0.7)  # Default range
            
            # Adjust strength based on categorization
            optimal_strength = (min_strength + max_strength) / 2
            
            # Keep within reasonable bounds
            result[base_filename] = min(max(optimal_strength, 0.1), 1.0)
    
    return result 