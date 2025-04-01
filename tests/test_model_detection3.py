#!/usr/bin/env python3
"""
Test script to verify model detection for NoobAI XL models.
"""

import os
import sys

# Add the cringegen directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cringegen"))

from cringegen.utils.logger import is_sdxl_model, get_model_info

def test_model_detection():
    """Test if model detection correctly identifies NoobAI XL models."""
    test_models = [
        "noobaiXLVpredv10.safetensors",  # The problematic model
        "noobai-XL-v1.0.safetensors",
        "noobaiV1.safetensors",
        "noobai_v1.safetensors",
        "sd_xl_base_1.0.safetensors",
        "dreamshaper_8.safetensors"
    ]
    
    print("\nTesting model detection with is_sdxl_model function:")
    print("=" * 60)
    for model in test_models:
        is_xl = is_sdxl_model(model)
        print(f"{model:<35} -> SDXL: {is_xl}")
    
    print("\nTesting model detection with get_model_info function:")
    print("=" * 60)
    for model in test_models:
        arch, family = get_model_info(model)
        print(f"{model:<35} -> Architecture: {arch:<5}, Family: {family}")

if __name__ == "__main__":
    test_model_detection() 