"""
Model utilities for cringegen.

This module provides functions for model-specific optimizations and prompt prefix injection.
"""

import os
import re
from typing import Dict, List, Tuple, Any, Optional, Union

from ..data.model_taxonomy import (
    get_model_architecture_defaults,
    get_model_family_prefix,
    get_optimal_resolution_for_model,
    get_combined_prefix,
    get_generation_parameters,
)

from .logger import get_model_info, print_colored_warning

class ModelOptimizer:
    """
    A class that handles model-specific optimizations and prompt tag injection.
    """
    
    def __init__(self, checkpoint_name: str = None, disable_tag_injection: bool = False):
        """
        Initialize the model optimizer.
        
        Args:
            checkpoint_name: Name of the checkpoint file to optimize for
            disable_tag_injection: Whether to disable automatic tag injection
        """
        self.checkpoint_name = checkpoint_name
        self.disable_tag_injection = disable_tag_injection
        self.architecture = "unknown"
        self.family = "unknown"
        self.family_settings = {}
        
        if checkpoint_name:
            self.detect_model()
    
    def detect_model(self) -> Tuple[str, str]:
        """
        Detect the model architecture and family from the checkpoint name.
        
        Returns:
            Tuple of (architecture, family)
        """
        if not self.checkpoint_name:
            return "unknown", "unknown"
        
        self.architecture, self.family = get_model_info(self.checkpoint_name)
        self.family_settings = get_model_family_prefix(self.family)
        return self.architecture, self.family
    
    def get_optimal_resolution(self, width: int = None, height: int = None) -> Tuple[int, int]:
        """
        Get the optimal resolution for the detected model architecture.
        
        Args:
            width: Current width (optional)
            height: Current height (optional)
            
        Returns:
            Tuple of (optimal_width, optimal_height)
        """
        if self.architecture == "unknown":
            # Default to SD 1.5 if unknown
            self.architecture = "sd15"
        
        # If dimensions are provided, calculate aspect ratio
        if width and height:
            aspect_ratio = width / height
        else:
            aspect_ratio = 1.0  # Default to square
        
        return get_optimal_resolution_for_model(self.architecture, aspect_ratio)
    
    def check_resolution(self, width: int, height: int) -> bool:
        """
        Check if the provided resolution is optimal for the detected model.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            True if resolution is optimal, False otherwise
        """
        if self.architecture == "unknown":
            return True  # Can't check unknown models
        
        # Get architecture defaults
        defaults = get_model_architecture_defaults(self.architecture)
        
        # Calculate total pixels
        total_pixels = width * height
        target_pixels = defaults["optimal_pixel_count"]
        margin = target_pixels * defaults["resolution_margin"]
        
        # Check if within margin of optimal
        is_optimal = abs(total_pixels - target_pixels) <= margin
        
        # If not optimal, show warning with suggestions
        if not is_optimal:
            # Calculate optimal dimensions for this aspect ratio
            aspect_ratio = width / height
            optimal_width, optimal_height = self.get_optimal_resolution(width, height)
            
            print_colored_warning(
                f"WARNING: Non-optimal resolution for {self.architecture} model ({width}×{height} = {total_pixels} pixels).\n"
                f"         Optimal pixel count is ~{target_pixels:,}. Consider using {optimal_width}×{optimal_height}."
            )
        
        return is_optimal
    
    def get_optimized_parameters(self) -> Dict[str, Any]:
        """
        Get optimized generation parameters for the detected model.
        
        Returns:
            Dictionary with optimized parameters
        """
        if self.architecture == "unknown" or self.family == "unknown":
            # Default to SD 1.5 if unknown
            return get_generation_parameters("sd15", "unknown")
        
        return get_generation_parameters(self.architecture, self.family)
    
    def detect_background_type(self, prompt: str) -> str:
        """
        Detect the type of background in the prompt.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            The detected background type or empty string if none found
        """
        # Skip if no background optimizations are available for this model family
        if not self.family_settings.get('background_optimizations'):
            return ""
        
        # Look for explicit background types
        background_types = self.family_settings['background_optimizations'].keys()
        for bg_type in background_types:
            if bg_type in prompt.lower():
                return bg_type
        
        # Check for common background keywords
        for bg_type in ['forest', 'city', 'beach', 'mountain', 'desert', 'space', 'castle', 'landscape']:
            if bg_type in prompt.lower() and bg_type in background_types:
                return bg_type
        
        return ""
    
    def get_background_optimization(self, bg_type: str) -> str:
        """
        Get the background optimization string for a specific background type.
        
        Args:
            bg_type: The background type
            
        Returns:
            The background optimization string
        """
        if not bg_type or not self.family_settings.get('background_optimizations'):
            return ""
        
        return self.family_settings['background_optimizations'].get(bg_type, "")
    
    def inject_model_prefix(self, prompt: str) -> str:
        """
        Inject model-specific prefix and background optimizations into the prompt.
        
        Args:
            prompt: The original prompt
            
        Returns:
            Prompt with model-specific prefix and background optimizations
        """
        if self.disable_tag_injection:
            return prompt
        
        if self.architecture == "unknown" or self.family == "unknown":
            return prompt
        
        # Store the prompt for negative prompt generation
        self._prompt_for_negative = prompt
        
        # Get the combined prefix
        prefix = get_combined_prefix(self.architecture, self.family)
        
        # Check if prefix is already in the prompt
        if prompt.startswith(prefix):
            prompt_without_prefix = prompt
        else:
            # Check if parts of the prefix are already in the prompt
            prefix_parts = [p.strip() for p in prefix.split(',')]
            filtered_prefix = prefix
            for part in prefix_parts:
                if part and part in prompt:
                    # Remove this part from the prefix
                    filtered_prefix = filtered_prefix.replace(f"{part}, ", "")
                    filtered_prefix = filtered_prefix.replace(f", {part}", "")
            
            # Apply the filtered prefix
            if filtered_prefix and filtered_prefix.strip(', '):
                prompt_without_prefix = prompt
                prompt = f"{filtered_prefix.rstrip(', ')}, {prompt}"
            else:
                prompt_without_prefix = prompt
        
        # Detect and optimize for background
        bg_type = self.detect_background_type(prompt_without_prefix)
        bg_optimization = self.get_background_optimization(bg_type)
        
        if bg_optimization and bg_optimization not in prompt:
            # Add background optimization at the end (before any final punctuation)
            if prompt.rstrip().endswith(('.', ',', '!', '?')):
                # Insert before the final punctuation
                last_char = prompt[-1]
                prompt = f"{prompt[:-1]}, {bg_optimization.rstrip(', ')}{last_char}"
            else:
                # Add to the end
                prompt = f"{prompt}, {bg_optimization.rstrip(', ')}"
        
        return prompt
    
    def inject_negative_prefix(self, negative_prompt: str) -> str:
        """
        Inject model-specific negative prefix into the negative prompt.
        
        Args:
            negative_prompt: The original negative prompt
            
        Returns:
            Negative prompt with model-specific prefix
        """
        if self.disable_tag_injection:
            return negative_prompt
        
        if self.architecture == "unknown":
            return negative_prompt
        
        # Try to get family-specific negative prefix first
        negative_prefix = self.family_settings.get("negative_prefix", "")
        
        # Fall back to architecture defaults if family doesn't have a negative prefix
        if not negative_prefix:
            defaults = get_model_architecture_defaults(self.architecture)
            negative_prefix = defaults.get("negative_prefix", "")
        
        # Create a working copy of the negative prefix
        working_prefix = negative_prefix
        
        # For NoobAI models, handle anthro/feral intelligently
        if self.family == "noob" and working_prefix:
            # Remove feral from negative prompt for feral subjects, remove anthro for anthro subjects
            if hasattr(self, '_prompt_for_negative') and self._prompt_for_negative:
                prompt_lower = self._prompt_for_negative.lower()
                
                if is_feral_subject(self._prompt_for_negative):
                    # For feral subjects, remove 'feral' from negative and add 'anthro'
                    working_prefix = working_prefix.replace("feral,", "").replace(", feral", "")
                    
                    # Make sure we remove any trailing commas and clean up whitespace
                    working_prefix = working_prefix.strip()
                    if working_prefix.endswith(','):
                        working_prefix = working_prefix[:-1].strip()
                    
                    # Add 'anthro' to negative prompt
                    working_prefix += ", anthro, anthropomorphic"
                    
                elif is_anthro_subject(self._prompt_for_negative):
                    # For anthro subjects, ensure 'feral' is in negative
                    if "feral" not in working_prefix:
                        working_prefix += ", feral"
        
        # Check if prefix is already in the prompt
        if negative_prompt and negative_prompt.startswith(working_prefix):
            return negative_prompt
        
        # Apply the prefix
        if working_prefix:
            # Ensure there's no double comma
            if negative_prompt:
                if working_prefix.endswith(',') or negative_prompt.startswith(','):
                    return f"{working_prefix.rstrip(', ')} {negative_prompt.lstrip(', ')}"
                else:
                    return f"{working_prefix}, {negative_prompt}"
            else:
                return working_prefix
        else:
            return negative_prompt

    def optimize_generation_request(self, 
                                   prompt: str, 
                                   negative_prompt: str = "", 
                                   width: int = None, 
                                   height: int = None,
                                   parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize a generation request for the detected model.
        
        Args:
            prompt: The generation prompt
            negative_prompt: The negative prompt (optional)
            width: Image width (optional)
            height: Image height (optional)
            parameters: Additional generation parameters (optional)
            
        Returns:
            Dictionary with optimized generation request
        """
        if not parameters:
            parameters = {}
        
        # Optimize the prompt
        optimized_prompt = self.inject_model_prefix(prompt)
        
        # Optimize the negative prompt
        optimized_negative_prompt = self.inject_negative_prefix(negative_prompt)
        
        # Get optimal model parameters
        model_params = self.get_optimized_parameters()
        
        # Merge with provided parameters (user params take precedence)
        for key, value in model_params.items():
            if key not in parameters:
                parameters[key] = value
        
        # Check and suggest optimal resolution
        if width and height:
            self.check_resolution(width, height)
        
        # Create the optimized request
        optimized_request = {
            "prompt": optimized_prompt,
            "negative_prompt": optimized_negative_prompt,
            **parameters
        }
        
        if width:
            optimized_request["width"] = width
        if height:
            optimized_request["height"] = height
        
        return optimized_request


def detect_model(checkpoint_name: str) -> Tuple[str, str]:
    """
    Shorthand function to detect model architecture and family.
    
    Args:
        checkpoint_name: Name of the checkpoint file
        
    Returns:
        Tuple of (architecture, family)
    """
    return get_model_info(checkpoint_name)

def optimize_prompt(checkpoint_name: str, prompt: str, disable_tag_injection: bool = False) -> str:
    """
    Optimize a prompt for a specific model.
    
    Args:
        checkpoint_name: Name of the checkpoint file
        prompt: The original prompt
        disable_tag_injection: Whether to disable automatic tag injection
        
    Returns:
        Optimized prompt
    """
    optimizer = ModelOptimizer(checkpoint_name, disable_tag_injection)
    return optimizer.inject_model_prefix(prompt)

def get_model_optimal_parameters(checkpoint_name: str) -> Dict[str, Any]:
    """
    Get optimal generation parameters for a model.
    
    Args:
        checkpoint_name: Name of the checkpoint file
        
    Returns:
        Dictionary with recommended generation parameters
    """
    optimizer = ModelOptimizer(checkpoint_name)
    return optimizer.get_optimized_parameters()

def get_model_optimal_resolution(checkpoint_name: str, width: int = None, height: int = None) -> Tuple[int, int]:
    """
    Get optimal resolution for a model.
    
    Args:
        checkpoint_name: Name of the checkpoint file
        width: Current width (optional)
        height: Current height (optional)
        
    Returns:
        Tuple of (optimal_width, optimal_height)
    """
    optimizer = ModelOptimizer(checkpoint_name)
    return optimizer.get_optimal_resolution(width, height)

def is_anthro_subject(prompt: str) -> bool:
    """
    Detect if the prompt contains an anthro subject.
    
    Args:
        prompt: The prompt to analyze
        
    Returns:
        True if the subject is anthro, False otherwise
    """
    prompt_lower = prompt.lower()
    anthro_indicators = [
        "anthro",
        "anthropomorphic",
        "humanoid",
        "furry"
    ]
    
    for indicator in anthro_indicators:
        if indicator in prompt_lower:
            return True
    
    return False

def is_feral_subject(prompt: str) -> bool:
    """
    Detect if the prompt contains a feral animal subject.
    
    Args:
        prompt: The prompt to analyze
        
    Returns:
        True if the subject is feral, False otherwise
    """
    prompt_lower = prompt.lower()
    
    # Direct feral indicators
    feral_indicators = [
        "feral",
        "quadruped",
        "quadrupedal",
        "four-legged",
        "non-anthro"
    ]
    
    for indicator in feral_indicators:
        if indicator in prompt_lower:
            return True
    
    return False 