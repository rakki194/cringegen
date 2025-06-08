"""
Resolution utilities for cringegen.

This module provides functions to determine optimal resolutions for different model architectures.
"""

from typing import List, Tuple

# List of functions we're implementing
__all__ = [
    "is_optimal_resolution",
    "get_optimal_resolution",
    "get_optimal_resolution_suggestions"
]

def is_optimal_resolution(width: int, height: int, model_type: str, use_deepshrink: bool = False) -> bool:
    """Check if dimensions are optimal for the given model type.
    
    Args:
        width: Image width
        height: Image height
        model_type: "sdxl" or "sd15" or other model type
        use_deepshrink: Whether DeepShrink is enabled (bypasses check if True)
        
    Returns:
        True if the resolution is optimal for the model type or DeepShrink is enabled
    """
    # Skip resolution check when DeepShrink is enabled
    if use_deepshrink:
        return True
        
    # Calculate total pixels
    total_pixels = width * height
    
    if model_type.lower() == "sdxl":
        # SDXL optimal: 1024*1024 (1,048,576 pixels)
        # Allow 5% margin for different aspect ratios
        target_pixels = 1048576
        margin = target_pixels * 0.05
        return abs(total_pixels - target_pixels) <= margin
    elif model_type.lower() == "sd15":
        # SD 1.5 optimal: 512*512 (262,144 pixels)
        # Allow 5% margin for different aspect ratios
        target_pixels = 262144
        margin = target_pixels * 0.05
        return abs(total_pixels - target_pixels) <= margin
    elif model_type.lower().startswith("sd35"):
        # SD 3.5 optimal: 1024*1024 (1,048,576 pixels) 
        # Similar to SDXL but might have different optimal resolutions for variants
        if "turbo" in model_type.lower():
            # SD 3.5 Turbo might work better with smaller resolutions
            target_pixels = 786432  # 768x1024
        else:
            target_pixels = 1048576  # 1024x1024
        margin = target_pixels * 0.05
        return abs(total_pixels - target_pixels) <= margin
    elif model_type.lower() == "flux":
        # FLUX models typically work well with SDXL resolutions
        target_pixels = 1048576  # 1024x1024
        margin = target_pixels * 0.05
        return abs(total_pixels - target_pixels) <= margin
    else:
        # For other models, assume any resolution is fine
        return True


def get_optimal_resolution(aspect_ratio: float, model_type: str) -> tuple:
    """Calculate optimal resolution for a given aspect ratio and model type.
    
    Args:
        aspect_ratio: Width to height ratio (width/height)
        model_type: "sdxl" or "sd15" or other model type
        
    Returns:
        Tuple of (width, height) that is optimal for the model type and aspect ratio
    """
    # First, determine the target pixel count
    if model_type.lower() == "sdxl":
        target_pixels = 1048576  # 1024*1024
    elif model_type.lower().startswith("sd35"):
        if "turbo" in model_type.lower():
            target_pixels = 786432  # 768x1024
        else:
            target_pixels = 1048576  # 1024x1024
    elif model_type.lower() == "flux":
        target_pixels = 1048576  # 1024x1024
    elif model_type.lower() == "sd15":
        target_pixels = 262144   # 512*512
    else:
        # Default to SD1.5 for unknown models
        target_pixels = 262144
    
    # Calculate dimensions based on aspect ratio and target pixels
    width = int((target_pixels * aspect_ratio) ** 0.5)
    height = int(width / aspect_ratio)
    
    # Ensure dimensions are divisible by 8 (required by Stable Diffusion)
    width = width - (width % 8)
    height = height - (height % 8)
    
    return (width, height)


def get_optimal_resolution_suggestions(width: int, height: int, model_type: str) -> list:
    """Get optimal resolution suggestions for a given aspect ratio and model type.
    
    Args:
        width: Current width
        height: Current height
        model_type: "sdxl" or "sd15" or other model type
        
    Returns:
        List of tuple suggestions (width, height) that are optimal
    """
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Get the base optimal resolution for this aspect ratio
    base_optimal = get_optimal_resolution(aspect_ratio, model_type)
    
    # Get some common variations with the same aspect ratio
    suggestions = [base_optimal]
    
    # Add common predefined resolutions that maintain the approximate pixel count
    if model_type.lower() in ["sdxl", "flux"] or model_type.lower().startswith("sd35"):
        if abs(aspect_ratio - 1.0) < 0.01:  # Square (1:1)
            suggestions = [(1024, 1024)]
        elif abs(aspect_ratio - (4/3)) < 0.01:  # 4:3
            suggestions = [(1184, 888)]
        elif abs(aspect_ratio - (3/4)) < 0.01:  # 3:4
            suggestions = [(888, 1184)]
        elif abs(aspect_ratio - (16/9)) < 0.01:  # 16:9
            suggestions = [(1360, 768)]
        elif abs(aspect_ratio - (9/16)) < 0.01:  # 9:16
            suggestions = [(768, 1360)]
        elif abs(aspect_ratio - (3/2)) < 0.01:  # 3:2
            suggestions = [(1248, 832)]
        elif abs(aspect_ratio - (2/3)) < 0.01:  # 2:3
            suggestions = [(832, 1248)]
        
        # Add common SDXL presets regardless of the input aspect ratio
        if len(suggestions) == 1:  # Only the base suggestion was added
            if aspect_ratio >= 1.0:  # Landscape
                suggestions.extend([
                    (1024, 1024),  # 1:1
                    (1152, 896),   # 9:7
                    (1216, 832),   # 19:13
                    (1344, 768),   # 7:4
                    (1536, 640),   # 12:5
                ])
            else:  # Portrait
                suggestions.extend([
                    (1024, 1024),  # 1:1
                    (896, 1152),   # 7:9
                    (832, 1216),   # 13:19
                    (768, 1344),   # 4:7
                    (640, 1536),   # 5:12
                ])
                
        # For SD3.5 Turbo, add its specific recommendations
        if model_type.lower() == "sd35_turbo":
            if aspect_ratio >= 1.0:  # Landscape
                suggestions.extend([
                    (1024, 768),   # 4:3
                    (1152, 768),   # 3:2 
                ])
            else:  # Portrait
                suggestions.extend([
                    (768, 1024),   # 3:4
                    (768, 1152),   # 2:3
                ])
    
    elif model_type.lower() == "sd15":
        if abs(aspect_ratio - 1.0) < 0.01:  # Square (1:1)
            suggestions = [(512, 512)]
        elif abs(aspect_ratio - (4/3)) < 0.01:  # 4:3
            suggestions = [(592, 444)]
        elif abs(aspect_ratio - (3/4)) < 0.01:  # 3:4
            suggestions = [(444, 592)]
        elif abs(aspect_ratio - (16/9)) < 0.01:  # 16:9
            suggestions = [(680, 384)]
        elif abs(aspect_ratio - (9/16)) < 0.01:  # 9:16
            suggestions = [(384, 680)]
        elif abs(aspect_ratio - (3/2)) < 0.01:  # 3:2
            suggestions = [(624, 416)]
        elif abs(aspect_ratio - (2/3)) < 0.01:  # 2:3
            suggestions = [(416, 624)]
        
        # Add common SD1.5 presets regardless of the input aspect ratio
        if len(suggestions) == 1:  # Only the base suggestion was added
            if aspect_ratio >= 1.0:  # Landscape
                suggestions.extend([
                    (512, 512),    # 1:1
                    (576, 448),    # 9:7
                    (608, 416),    # 19:13
                    (672, 384),    # 7:4
                ])
            else:  # Portrait
                suggestions.extend([
                    (512, 512),    # 1:1
                    (448, 576),    # 7:9
                    (416, 608),    # 13:19
                    (384, 672),    # 4:7
                ])
    
    # Ensure all suggestions are unique
    unique_suggestions = []
    for w, h in suggestions:
        if (w, h) not in unique_suggestions:
            unique_suggestions.append((w, h))
    
    return unique_suggestions 