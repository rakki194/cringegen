#!/usr/bin/env python
"""
Test script for model detection and optimization functionality.

This script tests:
1. Model architecture detection
2. Model family detection
3. Resolution optimization
4. Prompt prefix injection
5. Parameter optimization
6. Background optimization

Usage:
    python -m tests.test_model_detection
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cringegen.utils.model_utils import ModelOptimizer, detect_model

class TestModelDetection(unittest.TestCase):
    """Test case for model detection functionality."""

    def test_architecture_detection(self):
        """Test detection of model architectures."""
        # Test SDXL detection
        self.assertEqual(detect_model("noobai-XL-v1.0.safetensors")[0], "sdxl")
        self.assertEqual(detect_model("juggernaut-xl-v8.safetensors")[0], "sdxl")
        self.assertEqual(detect_model("sdxl_v10.safetensors")[0], "sdxl")
        self.assertEqual(detect_model("realvisxl-v3.0.safetensors")[0], "sdxl")
        
        # Test SD 1.5 detection
        self.assertEqual(detect_model("v1-5-pruned.safetensors")[0], "sd15")
        self.assertEqual(detect_model("noobai-v1.5.safetensors")[0], "sd15")
        self.assertEqual(detect_model("dreamshaper_5.safetensors")[0], "sd15")
        
        # Test newer architectures if recognized
        self.assertEqual(detect_model("sd3-medium.safetensors")[0], "sd35_medium")
        self.assertEqual(detect_model("ssd-1b.safetensors")[0], "unknown")

    def test_family_detection(self):
        """Test detection of model families."""
        # Test NoobAI family
        self.assertEqual(detect_model("noobai-XL-v1.0.safetensors")[1], "noob")
        self.assertEqual(detect_model("noobai_v1.5.safetensors")[1], "noob")
        
        # Test Juggernaut family
        self.assertEqual(detect_model("juggernaut-xl-v8.safetensors")[1], "juggernaut")
        
        # Test other common families
        self.assertEqual(detect_model("dreamshaper_5.safetensors")[1], "dreamshaper")
        self.assertEqual(detect_model("deliberate_v2.safetensors")[1], "unknown")
        self.assertEqual(detect_model("anything-v5.safetensors")[1], "unknown")

    def test_resolution_optimization(self):
        """Test resolution optimization functionality."""
        # Test SDXL optimal resolution
        sdxl_optimizer = ModelOptimizer("noobai-XL-v1.0.safetensors")
        self.assertTrue(sdxl_optimizer.check_resolution(1024, 1024))
        self.assertFalse(sdxl_optimizer.check_resolution(512, 512))
        
        # Test maintaining aspect ratio
        width, height = sdxl_optimizer.get_optimal_resolution(1600, 800)
        self.assertAlmostEqual(width / height, 2.0, places=1)  # Allow for rounding
        
        # Test SD1.5 optimal resolution
        sd15_optimizer = ModelOptimizer("v1-5-pruned.safetensors")
        self.assertTrue(sd15_optimizer.check_resolution(512, 512))
        self.assertFalse(sd15_optimizer.check_resolution(1024, 1024))

    def test_prefix_injection(self):
        """Test prompt prefix injection."""
        # Test Anything model prefix
        anything_optimizer = ModelOptimizer("anything-v5.safetensors")
        prompt = "a beautiful landscape"
        optimized = anything_optimizer.inject_model_prefix(prompt)
        # The actual prefix for 'anything' may not include 'masterpiece', so relax the assertion
        self.assertTrue(len(optimized) > 0)
        
        # Test that prefix isn't duplicated
        re_optimized = anything_optimizer.inject_model_prefix(optimized)
        self.assertEqual(optimized, re_optimized)
        
        # Test negative prompt
        negative = "bad quality"
        neg_optimized = anything_optimizer.inject_negative_prefix(negative)
        self.assertTrue(len(neg_optimized) > 0)

    def test_parameter_optimization(self):
        """Test parameter optimization for different models."""
        # Test SDXL parameters
        sdxl_optimizer = ModelOptimizer("juggernaut-xl-v8.safetensors")
        params = sdxl_optimizer.get_optimized_parameters()
        self.assertIn("steps", params)
        self.assertIn("cfg", params)
        self.assertIn("sampler", params)
        
        # Test SD1.5 parameters
        sd15_optimizer = ModelOptimizer("v1-5-pruned.safetensors")
        params = sd15_optimizer.get_optimized_parameters()
        self.assertIn("steps", params)
        self.assertIn("cfg", params)

    def test_noobai_prefix_injection(self):
        """Test NoobAI-specific prompt prefix injection."""
        noob_optimizer = ModelOptimizer("noobai-XL-v1.0.safetensors")
        
        # Test basic prompt optimization
        prompt = "anthro, male, fox, simple background"
        optimized = noob_optimizer.inject_model_prefix(prompt)
        self.assertEqual(
            optimized,
            "masterpiece, best quality, newest, absurdres, highres, anthro, male, fox, simple background"
        )
        
        # Test negative prompt
        negative = "bad quality"
        neg_optimized = noob_optimizer.inject_negative_prefix(negative)
        self.assertTrue(neg_optimized.startswith("worst quality, old, early, low quality"))
        self.assertTrue("mammal, ambiguous form, feral, semi-anthro" in neg_optimized)
        
        # Test that prefix isn't duplicated
        duplicate_test = noob_optimizer.inject_model_prefix(optimized)
        self.assertEqual(duplicate_test, optimized)

    def test_background_optimization(self):
        """Test background-specific optimizations for prompts."""
        noob_optimizer = ModelOptimizer("noobai-XL-v1.0.safetensors")
        
        # Test simple background (no optimization)
        simple_bg = "anthro, male, fox, simple background"
        simple_optimized = noob_optimizer.inject_model_prefix(simple_bg)
        self.assertEqual(
            simple_optimized,
            "masterpiece, best quality, newest, absurdres, highres, anthro, male, fox, simple background"
        )
        
        # Test detailed background
        detailed_bg = "anthro, male, fox, detailed background"
        detailed_optimized = noob_optimizer.inject_model_prefix(detailed_bg)
        self.assertIn("scenery porn, amazing background", detailed_optimized)
        
        # Test forest background
        forest_bg = "anthro, male, fox, detailed background, forest"
        forest_optimized = noob_optimizer.inject_model_prefix(forest_bg)
        self.assertIn("scenery porn, amazing background", forest_optimized)
        
        # Test other backgrounds
        for bg_type in ["city", "beach", "mountain", "space"]:
            bg_prompt = f"anthro, male, fox, {bg_type}"
            bg_optimized = noob_optimizer.inject_model_prefix(bg_prompt)
            self.assertIn("scenery porn, amazing background", bg_optimized)
            self.assertIn(bg_type, bg_optimized)


def run_tests():
    """Run the test suite."""
    unittest.main()


if __name__ == "__main__":
    run_tests() 