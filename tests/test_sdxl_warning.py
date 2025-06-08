#!/usr/bin/env python3
"""
Test script for model architecture detection and resolution warnings.
"""

import sys
import re
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    # Try to import from the installed package
    from cringegen.utils.logger import is_optimal_resolution, print_colored_warning, get_model_info
    USING_PACKAGE = True
except ImportError:
    # If that fails, define the functions locally
    USING_PACKAGE = False
    
    # ANSI color codes for colored output
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    def print_colored_warning(message: str) -> None:
        """Print a colored warning message to stderr."""
        print(f"{YELLOW}{message}{RESET}", file=sys.stderr)
    
    def get_model_info(checkpoint_name: str) -> tuple:
        """Detect model architecture and family from checkpoint name."""
        checkpoint_lower = checkpoint_name.lower()
        
        # First detect architecture
        architecture = "unknown"
        
        # SDXL detection - more comprehensive patterns
        xl_pattern = re.compile(r'(xl|XL)')
        if any(pattern in checkpoint_lower for pattern in [
            "sdxl", "sd-xl", "sd_xl", "stablediffusionxl", "stable-diffusion-xl"
        ]) or xl_pattern.search(checkpoint_name) or "noob" in checkpoint_lower:
            architecture = "sdxl"
        
        # SD3.5 detection
        elif any(pattern in checkpoint_lower for pattern in [
            "sd3.5", "sd-3.5", "sd_3.5", "sd35"
        ]):
            architecture = "sd35"
        
        # SD1.5 detection (baseline models)
        elif any(pattern in checkpoint_lower for pattern in [
            "sd1.5", "sd-1.5", "sd_1.5", "sd15"
        ]) or "yiffymix" in checkpoint_lower:
            architecture = "sd15"
        
        # Flux detection
        elif "chroma-unlocked" in checkpoint_lower:
            architecture = "flux"
        
        # LTX detection
        elif "ltx" in checkpoint_lower:
            architecture = "ltx"
        
        # Lumina detection
        elif "lumina" in checkpoint_lower:
            architecture = "lumina"
        
        # Now detect model family
        family = "unknown"
        
        # Common model families
        family_patterns = {
            "animagine": ["animagine"],
            "dreamshaper": ["dreamshaper"],
            "epicrealism": ["epicrealism"],
            "illustrious": ["illustrious"],
            "juggernaut": ["juggernaut"],
            "noob": ["noob", "realnoob"],
            "pony": ["pony"],
            "zavychroma": ["zavychroma"],
            "chroma": ["chroma"],
            "yiffymix": ["yiffymix"],
            "pixartsigma": ["pixartsigma"]
        }
        
        for family_name, patterns in family_patterns.items():
            if any(pattern in checkpoint_lower for pattern in patterns):
                family = family_name
                break
        
        # Handle special cases for SD3.5
        if architecture == "sd35":
            if "large" in checkpoint_lower:
                family = "sd35l"
            elif "medium" in checkpoint_lower:
                if "turbo" in checkpoint_lower:
                    family = "sd35mt"
                else:
                    family = "sd35m"
        
        # Handle vanilla SDXL case
        if architecture == "sdxl" and family == "unknown":
            if any(pattern in checkpoint_lower for pattern in ["base", "1.0"]):
                family = "sdxl"
        
        # Handle LTX and Lumina where architecture = family
        if architecture in ["ltx", "lumina"] and family == "unknown":
            family = architecture
            
        return architecture, family
    
    def is_optimal_resolution(width: int, height: int, model_type: str, use_deepshrink: bool = False) -> bool:
        """Check if dimensions are optimal for the given model type."""
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
        else:
            # For other models, assume any resolution is fine
            return True

def test_model_detection():
    """Test model architecture and family detection with various checkpoint names."""
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
        "animagine-xl-4.0-opt.safetensors",
        "dreamshaperXL_lightningDPMSDE.safetensors",
        "epicrealismXL_v9.safetensors",
        "sd3.5_large_fp8_scaled.safetensors",
        "sd3.5_medium.safetensors",
        "ltx-video-2b-v0.9.safetensors",
        "lumina_2.safetensors",
        "chroma-unlocked-v13.safetensors",
        "yiffymix_v36.safetensors"
    ]
    
    print("\n--- Testing Model Architecture and Family Detection ---\n")
    
    for model in test_models:
        arch, family = get_model_info(model)
        print(f"Model: {model} -> Architecture: {arch}, Family: {family}")
        
        # Also show warning for SDXL models to maintain backward compatibility
        if arch == "sdxl":
            print_colored_warning(f"WARNING: Detected SDXL model architecture for checkpoint '{model}'")

def test_resolution_check():
    """Test optimal resolution checks for different model architectures."""
    test_cases = [
        ("sdxl", 1024, 1024),  # Optimal SDXL
        ("sdxl", 768, 768),    # Non-optimal SDXL
        ("sdxl", 896, 1152),   # Optimal SDXL (different aspect ratio)
        ("sd15", 512, 512),    # Optimal SD1.5
        ("sd15", 768, 768),    # Non-optimal SD1.5
        ("sd15", 448, 576),    # Optimal SD1.5 (different aspect ratio)
        ("sd35", 1024, 1024),  # SD3.5 (assuming same as SDXL for now)
        ("flux", 512, 512),    # Flux (assuming same as SD1.5 for now)
    ]
    
    print("\n--- Testing Resolution Checks ---\n")
    
    for model_type, width, height in test_cases:
        pixels = width * height
        is_optimal = is_optimal_resolution(width, height, model_type)
        
        if is_optimal:
            print(f"✓ Optimal resolution for {model_type}: {width}*{height} = {pixels} pixels")
        else:
            if model_type.lower() == "sdxl" or model_type.lower() == "sd35":
                optimal_pixels = 1048576
                print_colored_warning(
                    f"WARNING: Non-optimal resolution for {model_type} model ({width}*{height} = {pixels} pixels).\n"
                    f"         Optimal pixel count is ~{optimal_pixels:,}. Consider using 1024*1024, 896*1152, or 768*1344."
                )
            elif model_type.lower() == "sd15" or model_type.lower() == "flux":
                optimal_pixels = 262144
                print_colored_warning(
                    f"WARNING: Non-optimal resolution for {model_type} model ({width}*{height} = {pixels} pixels).\n"
                    f"         Optimal pixel count is ~{optimal_pixels:,}. Consider using 512*512, 448*576, or 384*640."
                )
            else:
                print_colored_warning(
                    f"WARNING: Non-optimal resolution for {model_type} model ({width}*{height} = {pixels} pixels)."
                )

if __name__ == "__main__":
    print(f"Running tests in {'package' if USING_PACKAGE else 'standalone'} mode")
    test_model_detection()
    test_resolution_check() 