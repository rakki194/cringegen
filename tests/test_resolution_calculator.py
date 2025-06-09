"""
Test script for the resolution calculator functions in logger.py
"""

import sys
from pathlib import Path

# Add the repository root to the Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root))

from cringegen.utils.logger import (
    get_optimal_resolution,
    get_optimal_resolution_suggestions,
    is_optimal_resolution,
)

def test_optimal_resolution_calculation():
    """Test the get_optimal_resolution function with various aspect ratios."""
    print("\n=== Testing get_optimal_resolution ===")
    
    # Test common aspect ratios for SDXL
    test_cases_sdxl = [
        (1, 1),       # 1:1
        (16, 9),      # 16:9
        (9, 16),      # 9:16
        (4, 3),       # 4:3
        (3, 4),       # 3:4
        (3, 2),       # 3:2
        (2, 3),       # 2:3
        (21, 9),      # 21:9 (Ultrawide)
        (1.618, 1),   # Golden ratio
    ]
    
    print("SDXL Optimal Resolutions:")
    for width_ratio, height_ratio in test_cases_sdxl:
        aspect_ratio = width_ratio / height_ratio
        width, height = get_optimal_resolution(aspect_ratio, "sdxl")
        pixels = width * height
        print(f"{width_ratio}:{height_ratio} → {width}*{height} = {pixels} pixels (target: 1,048,576)")
    
    print("\nSD1.5 Optimal Resolutions:")
    for width_ratio, height_ratio in test_cases_sdxl:
        aspect_ratio = width_ratio / height_ratio
        width, height = get_optimal_resolution(aspect_ratio, "sd15")
        pixels = width * height
        print(f"{width_ratio}:{height_ratio} → {width}*{height} = {pixels} pixels (target: 262,144)")


def test_optimal_resolution_suggestions():
    """Test the get_optimal_resolution_suggestions function."""
    print("\n=== Testing get_optimal_resolution_suggestions ===")
    
    # Test cases
    test_cases = [
        (1024, 768, "sdxl"),    # 4:3 SDXL
        (1920, 1080, "sdxl"),   # 16:9 SDXL
        (768, 1024, "sdxl"),    # 3:4 SDXL
        (512, 384, "sd15"),     # 4:3 SD1.5
        (960, 540, "sd15"),     # 16:9 SD1.5
        (384, 512, "sd15"),     # 3:4 SD1.5
    ]
    
    for width, height, model_type in test_cases:
        suggestions = get_optimal_resolution_suggestions(width, height, model_type)
        print(f"\nFor {width}*{height} ({width/height:.2f}:1) with {model_type}:")
        for i, (w, h) in enumerate(suggestions):
            pixels = w * h
            print(f"  {i+1}. {w}*{h} = {pixels} pixels")


def test_is_optimal_resolution():
    """Test the is_optimal_resolution function."""
    print("\n=== Testing is_optimal_resolution ===")
    
    # Test cases
    test_cases = [
        (1024, 1024, "sdxl"),   # Perfect SDXL
        (1152, 896, "sdxl"),    # Good SDXL
        (1920, 1080, "sdxl"),   # Bad SDXL (too many pixels)
        (512, 512, "sd15"),     # Perfect SD1.5
        (576, 448, "sd15"),     # Good SD1.5
        (1024, 1024, "sd15"),   # Bad SD1.5 (too many pixels)
    ]
    
    for width, height, model_type in test_cases:
        is_optimal = is_optimal_resolution(width, height, model_type)
        pixels = width * height
        status = "✓ Optimal" if is_optimal else "✗ Non-optimal"
        print(f"{status}: {width}*{height} = {pixels} pixels for {model_type}")


if __name__ == "__main__":
    # Run all tests
    test_optimal_resolution_calculation()
    test_optimal_resolution_suggestions()
    test_is_optimal_resolution()
    
    print("\nAll tests completed.") 