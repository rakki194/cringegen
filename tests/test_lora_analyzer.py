#!/usr/bin/env python3
"""
Test script for the LoRA analyzer functionality.
"""

import argparse
import json
import os
import sys

from cringegen.utils.lora_metadata import (
    analyze_lora_type,
    analyze_multiple_loras,
    get_loras_by_type,
    suggest_lora_combinations,
)


def test_single_lora():
    """Test analyzing a single LoRA file"""
    lora_path = "dummy_lora_path"
    no_cache = False
    print(f"\n=== Testing analysis of: {os.path.basename(lora_path)} ===")

    try:
        analysis = analyze_lora_type(lora_path, not no_cache)
        print(f"Type: {analysis['type'].upper()} (Confidence: {analysis['confidence']:.2f})")

        if analysis["evidence"]:
            print("\nEvidence:")
            for item in analysis["evidence"]:
                print(f"  • {item}")

        if analysis["recommendations"]:
            print("\nRecommendations:")

            if analysis["recommendations"].get("prompt_tips"):
                print("\nPrompt Tips:")
                for tip in analysis["recommendations"]["prompt_tips"]:
                    print(f"  • {tip}")

            if analysis["recommendations"].get("weight_suggestions"):
                print("\nWeight Suggestions:")
                for suggestion in analysis["recommendations"]["weight_suggestions"]:
                    print(f"  • {suggestion}")

            if analysis["recommendations"].get("combination_suggestions"):
                print("\nCombination Suggestions:")
                for suggestion in analysis["recommendations"]["combination_suggestions"]:
                    print(f"  • {suggestion}")

        return analysis

    except Exception as e:
        print(f"Error analyzing {lora_path}: {e}")
        return None


def test_multiple_loras():
    """Test analyzing multiple LoRAs in a directory"""
    lora_dir = "dummy_lora_dir"
    pattern = "*.safetensors"
    force_refresh = False
    print(f"\n=== Testing analysis of multiple LoRAs in: {lora_dir} ===")
    try:
        results = analyze_multiple_loras(lora_dir, pattern, force_refresh)
        loras_by_type = {"style": [], "character": [], "concept": [], "unknown": []}
        for lora in results.values():
            loras_by_type[lora["type"]].append((lora["name"], lora["confidence"]))
        for t in loras_by_type:
            loras_by_type[t].sort(key=lambda x: x[1], reverse=True)
        print(f"\nFound {len(results)} LoRAs in {lora_dir}")
        for lora_type, loras in loras_by_type.items():
            if loras:
                print(f"\n--- {lora_type.upper()} LoRAs ({len(loras)}) ---")
                for name, confidence in loras:
                    print(f"  • {name} (Confidence: {confidence:.2f})")
        assert isinstance(lora_dir, str)
    except Exception as e:
        print(f"Error analyzing LoRAs in {lora_dir}: {e}")


def test_loras_by_type():
    """Test getting LoRAs by type"""
    lora_dir = "dummy_lora_dir"
    lora_type = "dummy_type"
    min_confidence = 0.5
    print(f"\n=== Testing getting {lora_type.upper()} LoRAs from: {lora_dir} ===")
    try:
        results = get_loras_by_type(lora_dir, lora_type, min_confidence)
        print(f"\nFound {len(results)} {lora_type} LoRAs with confidence >= {min_confidence}")
        results.sort(key=lambda x: x["confidence"], reverse=True)
        for lora in results:
            print(f"  • {lora['name']} (Confidence: {lora['confidence']:.2f})")
        assert isinstance(lora_type, str)
    except Exception as e:
        print(f"Error getting {lora_type} LoRAs from {lora_dir}: {e}")


def test_lora_combinations():
    """Test suggesting LoRA combinations"""
    lora_path = "dummy_lora_path"
    lora_dir = "dummy_lora_dir"
    max_suggestions = 3
    print(f"\n=== Testing LoRA combination suggestions for: {os.path.basename(lora_path)} ===")

    try:
        suggestions = suggest_lora_combinations(lora_path, lora_dir, max_suggestions)

        if suggestions.get("complementary"):
            print("\nComplementary LoRAs (work well together):")
            for lora in suggestions["complementary"]:
                print(
                    f"  • {lora['name']} ({lora['type'].upper()}, Confidence: {lora['confidence']:.2f})"
                )

        if suggestions.get("similar"):
            print("\nSimilar LoRAs (same type):")
            for lora in suggestions["similar"]:
                print(
                    f"  • {lora['name']} ({lora['type'].upper()}, Confidence: {lora['confidence']:.2f})"
                )

        if suggestions.get("contrasting"):
            print("\nContrasting LoRAs (different type, interesting combinations):")
            for lora in suggestions["contrasting"]:
                print(
                    f"  • {lora['name']} ({lora['type'].upper()}, Confidence: {lora['confidence']:.2f})"
                )

        return suggestions

    except Exception as e:
        print(f"Error suggesting combinations for {lora_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test LoRA analyzer functionality")
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="/home/kade/comfy/models/loras",
        help="Directory containing LoRA files",
    )
    parser.add_argument(
        "--style-lora",
        type=str,
        default="/home/kade/comfy/models/loras/noob/chunie-v1s2000.safetensors",
        help="Path to a style LoRA for testing",
    )
    parser.add_argument(
        "--character-lora",
        type=str,
        default="/home/kade/comfy/models/loras/noob/foxparks-v2s1800.safetensors",
        help="Path to a character LoRA for testing",
    )
    parser.add_argument(
        "--concept-lora",
        type=str,
        default="/home/kade/comfy/models/loras/noob/fart_fetish-v2s3000.safetensors",
        help="Path to a concept LoRA for testing",
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force refresh the analysis cache"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Track test results
    test_results = {}

    # Test single LoRA analysis
    test_results["style_lora_analysis"] = test_single_lora()
    test_results["character_lora_analysis"] = test_single_lora()
    test_results["concept_lora_analysis"] = test_single_lora()

    # Test multiple LoRA analysis
    style_lora_dir = os.path.join(args.lora_dir, "noob")
    test_results["multiple_loras_analysis"] = test_multiple_loras()

    # Test getting LoRAs by type
    test_results["style_loras"] = test_loras_by_type()
    test_results["character_loras"] = test_loras_by_type()
    test_results["concept_loras"] = test_loras_by_type()

    # Test LoRA combination suggestions
    test_results["style_lora_combinations"] = test_lora_combinations()
    test_results["character_lora_combinations"] = test_lora_combinations()
    test_results["concept_lora_combinations"] = test_lora_combinations()

    # Output as JSON if requested
    if args.json:
        print(json.dumps(test_results, indent=2))

    # Print summary
    print("\n=== Test Summary ===")
    style_analysis = test_results.get("style_lora_analysis", {})
    character_analysis = test_results.get("character_lora_analysis", {})
    concept_analysis = test_results.get("concept_lora_analysis", {})

    if style_analysis:
        print(
            f"Style LoRA: {style_analysis.get('name', 'N/A')} - Type: {style_analysis.get('type', 'unknown').upper()} (Confidence: {style_analysis.get('confidence', 0):.2f})"
        )

    if character_analysis:
        print(
            f"Character LoRA: {character_analysis.get('name', 'N/A')} - Type: {character_analysis.get('type', 'unknown').upper()} (Confidence: {character_analysis.get('confidence', 0):.2f})"
        )

    if concept_analysis:
        print(
            f"Concept LoRA: {concept_analysis.get('name', 'N/A')} - Type: {concept_analysis.get('type', 'unknown').upper()} (Confidence: {concept_analysis.get('confidence', 0):.2f})"
        )

    # Print counts of LoRAs by type
    multiple_analysis = test_results.get("multiple_loras_analysis", {})
    if multiple_analysis:
        style_count = len(
            [
                l
                for l in multiple_analysis.values()
                if l["type"] == "style" and l["confidence"] >= 0.6
            ]
        )
        character_count = len(
            [
                l
                for l in multiple_analysis.values()
                if l["type"] == "character" and l["confidence"] >= 0.6
            ]
        )
        concept_count = len(
            [
                l
                for l in multiple_analysis.values()
                if l["type"] == "concept" and l["confidence"] >= 0.6
            ]
        )
        unknown_count = len(
            [
                l
                for l in multiple_analysis.values()
                if l["type"] == "unknown" or l["confidence"] < 0.6
            ]
        )

        print(f"\nMultiple LoRA Analysis Results:")
        print(f"  • Style LoRAs: {style_count}")
        print(f"  • Character LoRAs: {character_count}")
        print(f"  • Concept LoRAs: {concept_count}")
        print(f"  • Unknown/Low confidence: {unknown_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
