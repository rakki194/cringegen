#!/usr/bin/env python
"""
Test script for NoobAI model optimizations with NSFW prompts.

This script directly uses the ModelOptimizer to demonstrate background optimizations
outside the main command workflow.

Usage:
    python -m tests.test_noobai_nsfw
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cringegen.utils.model_utils import ModelOptimizer

def test_simple_optimization():
    """Direct test of model optimization."""
    print("\n=== Testing Direct NoobAI Model Optimization ===\n")
    
    # Create optimizer for noobai model
    optimizer = ModelOptimizer("realnoob-v3.safetensors")
    
    # Sample prompt and negative prompt
    prompt = "anthro, male, fox, detailed background, forest"
    negative_prompt = "bad quality, deformed hands"
    
    # Optimize the prompt
    optimized_prompt = optimizer.inject_model_prefix(prompt)
    optimized_negative = optimizer.inject_negative_prefix(negative_prompt)
    
    # Print original and optimized prompts
    print(f"Original prompt: {prompt}")
    print(f"Optimized prompt: {optimized_prompt}")
    print()
    print(f"Original negative prompt: {negative_prompt}")
    print(f"Optimized negative prompt: {optimized_negative}")
    print()
    
    # Detect background
    bg_type = optimizer.detect_background_type(prompt)
    print(f"Background detected: {bg_type}")
    
    # Get optimal parameters
    params = optimizer.get_optimized_parameters()
    print("\nOptimized Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Test with a different background
    prompt2 = "anthro, female, wolf, beach scene"
    optimized_prompt2 = optimizer.inject_model_prefix(prompt2)
    bg_type2 = optimizer.detect_background_type(prompt2)
    
    print("\nSecond Test:")
    print(f"Original prompt: {prompt2}")
    print(f"Background detected: {bg_type2}")
    print(f"Optimized prompt: {optimized_prompt2}")

def main():
    """Run the test."""
    test_simple_optimization()
    print("\nTest completed!")

if __name__ == "__main__":
    main() 