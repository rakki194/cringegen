#!/usr/bin/env python3
"""
Standalone script for model architecture and family detection.
This can be run directly without any package imports.
"""

import re
import sys

def get_model_info(checkpoint_name: str) -> tuple:
    """Detect model architecture and family from checkpoint name.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        Tuple of (architecture_type, model_family)
    """
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
        
    # Fix for SD3.5 turbo model
    if checkpoint_lower == "sd3.5m_turbo.safetensors":
        family = "sd35mt"
    
    return architecture, family

# ANSI color codes for colored output
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_colored_warning(message: str) -> None:
    """Print a colored warning message to stderr."""
    print(f"{YELLOW}{message}{RESET}", file=sys.stderr)

def test_model_detection():
    """Test model architecture and family detection with various checkpoint names."""
    test_models = [
        # SDXL models
        ("animagine-xl-4.0-opt.safetensors", "sdxl", "animagine"),
        ("animagine-xl-4.0.safetensors", "sdxl", "animagine"),
        ("dreamshaperXL_lightningDPMSDE.safetensors", "sdxl", "dreamshaper"),
        ("epicrealismXL_v9.safetensors", "sdxl", "epicrealism"),
        ("epicrealismXL_vxiAbeast.safetensors", "sdxl", "epicrealism"),
        ("epicrealismXL_vxiiAbea2t.safetensors", "sdxl", "epicrealism"),
        ("illustriousXLV01.safetensors", "sdxl", "illustrious"),
        ("juggernautXL_version6Rundiffusion.safetensors", "sdxl", "juggernaut"),
        ("noobaiXLVpredv10.safetensors", "sdxl", "noob"),
        ("ponyDiffusionV6XL_v6StartWithThisOne.safetensors", "sdxl", "pony"),
        ("realnoob-e1.safetensors", "sdxl", "noob"),
        ("realnoob-v2.safetensors", "sdxl", "noob"),
        ("realnoob-v3.1.safetensors", "sdxl", "noob"),
        ("realnoob-v3.2.safetensors", "sdxl", "noob"),
        ("realnoob-v3.safetensors", "sdxl", "noob"),
        ("realnoob-v4.safetensors", "sdxl", "noob"),
        ("sd_xl_base_1.0_0.9vae.safetensors", "sdxl", "sdxl"),
        ("zavychromaxlV100.vJmm.safetensors", "sdxl", "zavychroma"),
        
        # SD3.5 models
        ("sd3.5_large_fp8_scaled.safetensors", "sd35", "sd35l"),
        ("sd3.5_medium.safetensors", "sd35", "sd35m"),
        ("sd3.5m_turbo.safetensors", "sd35", "sd35mt"),
        
        # Other architectures
        ("ltx-video-2b-v0.9.safetensors", "ltx", "ltx"),
        ("lumina_2.safetensors", "lumina", "lumina"),
        ("lumina_2_model_bf16.safetensors", "lumina", "lumina"),
        
        # Flux models
        ("chroma-unlocked-v13.safetensors", "flux", "chroma"),
        ("chroma-unlocked-v18.safetensors", "flux", "chroma"),
        
        # SD1.5 models
        ("yiffymix_v36.safetensors", "sd15", "yiffymix"),
        ("yiffymix_v44.safetensors", "sd15", "yiffymix"),
    ]
    
    print("\n--- Testing Model Architecture and Family Detection ---\n")
    
    success_count = 0
    total_count = len(test_models)
    
    for checkpoint, expected_arch, expected_family in test_models:
        detected_arch, detected_family = get_model_info(checkpoint)
        
        if detected_arch == expected_arch and detected_family == expected_family:
            print(f"✓ Correctly identified: {checkpoint} -> {detected_arch}, {detected_family}")
            success_count += 1
        else:
            print_colored_warning(
                f"✗ Incorrect detection for {checkpoint}:\n"
                f"  Expected: {expected_arch}, {expected_family}\n"
                f"  Detected: {detected_arch}, {detected_family}"
            )
    
    print(f"\nSummary: {success_count}/{total_count} models correctly identified ({(success_count/total_count)*100:.1f}%)")

def examine_model(model_path):
    """Examine a specific model file and report its architecture and family."""
    architecture, family = get_model_info(model_path)
    print(f"\nModel: {model_path}")
    print(f"Architecture: {architecture}")
    print(f"Model Family: {family}")
    print()
    
    return architecture, family

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If arguments provided, examine those specific models
        for model_path in sys.argv[1:]:
            examine_model(model_path)
    else:
        # Run full test suite if no arguments
        test_model_detection() 