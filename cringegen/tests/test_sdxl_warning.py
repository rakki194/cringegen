#!/usr/bin/env python3
"""
Test script for SDXL model detection and warnings.
"""

import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cringegen.utils.logger import is_sdxl_model, print_colored_warning, is_optimal_resolution

def test_sdxl_detection():
    """Test SDXL model detection with various checkpoint names."""
    test_models = [
        "sdxl_base.safetensors",
        "sd-xl_base.safetensors",
        "stable-diffusion-xl-base.safetensors",
        "stable_diffusion_1_5.safetensors",
        "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        "deliberate.safetensors",
        "anythingV3_fp16.safetensors",
        "noobAI_v123.safetensors",
        "SDXL_1.0.safetensors",
        "pixartSigma_v1.safetensors",
    ]
    
    print("\n--- Testing SDXL Model Detection ---\n")
    for model in test_models:
        if is_sdxl_model(model):
            print_colored_warning(f"WARNING: Detected SDXL model architecture for checkpoint '{model}'")
        else:
            print(f"Not an SDXL model: {model}")

def test_resolution_check():
    """Test optimal resolution checks for different model architectures."""
    test_cases = [
        ("SDXL", 1024, 1024),  # Optimal SDXL
        ("SDXL", 768, 768),    # Non-optimal SDXL
        ("SDXL", 896, 1152),   # Optimal SDXL (different aspect ratio)
        ("SD15", 512, 512),    # Optimal SD1.5
        ("SD15", 768, 768),    # Non-optimal SD1.5
        ("SD15", 448, 576),    # Optimal SD1.5 (different aspect ratio)
    ]
    
    print("\n--- Testing Resolution Checks ---\n")
    for model_type, width, height in test_cases:
        pixels = width * height
        is_optimal = is_optimal_resolution(width, height, model_type)
        
        if is_optimal:
            print(f"✓ Optimal resolution for {model_type}: {width}×{height} = {pixels} pixels")
        else:
            if model_type.lower() == "sdxl":
                optimal_pixels = 1048576
                print_colored_warning(
                    f"WARNING: Non-optimal resolution for {model_type} model ({width}×{height} = {pixels} pixels).\n"
                    f"         Optimal pixel count is ~{optimal_pixels:,}. Consider using 1024×1024, 896×1152, or 768×1344."
                )
            elif model_type.lower() == "sd15":
                optimal_pixels = 262144
                print_colored_warning(
                    f"WARNING: Non-optimal resolution for {model_type} model ({width}×{height} = {pixels} pixels).\n"
                    f"         Optimal pixel count is ~{optimal_pixels:,}. Consider using 512×512, 448×576, or 384×640."
                )
            else:
                print_colored_warning(
                    f"WARNING: Non-optimal resolution for {model_type} model ({width}×{height} = {pixels} pixels)."
                )

if __name__ == "__main__":
    test_sdxl_detection()
    test_resolution_check() 