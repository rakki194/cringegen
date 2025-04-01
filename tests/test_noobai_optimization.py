#!/usr/bin/env python
"""
Test script for NoobAI model optimizations.

This script specifically tests the NoobAI model prompt optimizations:
1. Prefix injection
2. Negative prompt generation
3. Background-specific enhancements

Usage:
    python -m tests.test_noobai_optimization
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cringegen.utils.model_utils import ModelOptimizer

def test_noobai_optimization():
    """Test NoobAI prompt optimization."""
    print("\n=== Testing NoobAI Model Optimizations ===\n")
    
    noob_optimizer = ModelOptimizer("noobai-XL-v1.0.safetensors")
    
    test_prompts = [
        # Simple prompts
        "anthro, male, fox, simple background",
        "anthro, female, wolf, cute",
        
        # Background variations
        "anthro, male, fox, detailed background",
        "anthro, male, fox, detailed background, forest",
        "anthro, female, wolf, beach",
        "anthro, male, dragon, city",
        "anthro, female, tiger, mountain",
        "anthro, male, lion, desert",
        "anthro, female, fox, space",
        "anthro, male, wolf, castle",
        "anthro, female, dragon, landscape",
        
        # With existing quality tags
        "masterpiece, anthro, male, fox, simple background",
        "highres, anthro, female, wolf, detailed background",
        "absurdres, anthro, male, fox, forest",
    ]
    
    # Test negative prompt
    negative_prompt = "bad quality, deformed paws"
    optimized_negative = noob_optimizer.inject_negative_prefix(negative_prompt)
    
    print("=== Negative Prompt Optimization ===")
    print(f"Original: {negative_prompt}")
    print(f"Optimized: {optimized_negative}")
    print()
    
    # Test prompt optimizations
    print("=== Prompt Optimizations ===")
    for prompt in test_prompts:
        optimized = noob_optimizer.inject_model_prefix(prompt)
        bg_type = noob_optimizer.detect_background_type(prompt)
        
        print(f"Prompt: {prompt}")
        print(f"Background type detected: {bg_type or 'None'}")
        print(f"Optimized: {optimized}")
        print()

def main():
    """Run the NoobAI optimization tests."""
    test_noobai_optimization()
    print("All tests completed!")

if __name__ == "__main__":
    main() 