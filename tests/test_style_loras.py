#!/usr/bin/env python3
"""
Test script for style LoRA detection.
"""

import os
import sys
from cringegen.utils.lora_metadata import analyze_lora_type

# Style LoRAs to test
STYLE_LORA_PATHS = [
    "/home/kade/toolkit/diffusion/comfy/models/loras/noob/beksinski-v1s1600.safetensors",
    "/home/kade/toolkit/diffusion/comfy/models/loras/noob/gren_art-v1s2000.safetensors",
    "/home/kade/toolkit/diffusion/comfy/models/loras/noob/kenket-v1s3000.safetensors",
    "/home/kade/toolkit/diffusion/comfy/models/loras/noob/solisolsoli-v1s1600.safetensors",
    "/home/kade/toolkit/diffusion/comfy/models/loras/noob/noob-bill_watterson-saveus-v1s1600.safetensors",
]


def test_style_lora(lora_path):
    """Test a single style LoRA"""
    print(f"\n=== Testing style LoRA: {os.path.basename(lora_path)} ===")

    try:
        # Skip cache
        analysis = analyze_lora_type(lora_path, False)

        print(f"Type: {analysis['type'].upper()} (Confidence: {analysis['confidence']:.2f})")

        if analysis["evidence"]:
            print("\nEvidence:")
            for item in analysis["evidence"]:
                print(f"  • {item}")

        return analysis

    except Exception as e:
        print(f"Error analyzing {lora_path}: {e}")
        return None


def main():
    """Main test function"""
    results = []

    # Test each style LoRA
    for lora_path in STYLE_LORA_PATHS:
        if os.path.exists(lora_path):
            result = test_style_lora(lora_path)
            if result:
                results.append(result)
        else:
            print(f"LoRA file not found: {lora_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Tested {len(results)} LoRAs")

    correct_style_loras = [r for r in results if r["type"] == "style"]
    print(f"Correctly identified as style: {len(correct_style_loras)}")

    for lora in correct_style_loras:
        print(f"  • {lora['name']} (Confidence: {lora['confidence']:.2f})")

    other_loras = [r for r in results if r["type"] != "style"]
    print(f"Identified as other types: {len(other_loras)}")

    for lora in other_loras:
        print(
            f"  • {lora['name']} - Type: {lora['type'].upper()} (Confidence: {lora['confidence']:.2f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
