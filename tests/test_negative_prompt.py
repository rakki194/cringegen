#!/usr/bin/env python
"""
Test script for negative prompt handling with anthro vs feral subjects.

This script tests how negative prompts are adjusted based on the subject type.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cringegen.utils.model_utils import ModelOptimizer, is_anthro_subject, is_feral_subject

def test_negative_prompts():
    """Test negative prompt adjustments for different subject types."""
    print("\n=== Testing Subject Detection ===")
    
    test_prompts = [
        "anthro, male, wolf",
        "feral, male, wolf",
        "male wolf, quadruped",
        "humanoid, female, fox",
        "female fox, four-legged"
    ]
    
    for prompt in test_prompts:
        is_anthro = is_anthro_subject(prompt)
        is_feral = is_feral_subject(prompt)
        print(f"Prompt: {prompt}")
        print(f"  Anthro: {is_anthro}")
        print(f"  Feral: {is_feral}")
    
    print("\n=== Testing Negative Prompt Generation ===")
    
    # Create optimizer for NoobAI model
    optimizer = ModelOptimizer("noobai-XL-v1.0.safetensors")
    
    # Test with anthro subject
    anthro_prompt = "anthro, male, wolf"
    optimizer.inject_model_prefix(anthro_prompt)  # This stores the prompt for negative prompt generation
    anthro_negative = optimizer.inject_negative_prefix("")
    print(f"\nAnthro subject: {anthro_prompt}")
    print(f"Negative prompt: {anthro_negative}")
    
    # Test with feral subject
    feral_prompt = "feral, male, wolf"
    optimizer.inject_model_prefix(feral_prompt)  # This stores the prompt for negative prompt generation
    feral_negative = optimizer.inject_negative_prefix("")
    print(f"\nFeral subject: {feral_prompt}")
    print(f"Negative prompt: {feral_negative}")

if __name__ == "__main__":
    test_negative_prompts() 