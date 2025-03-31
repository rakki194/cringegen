"""
Script to inspect LoRA metadata in detail
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional


def inspect_safetensors_metadata(file_path: str) -> Dict[str, Any]:
    """
    Inspect metadata from a safetensors file in detail

    Args:
        file_path: Path to the safetensors file

    Returns:
        Dictionary of metadata
    """
    try:
        import safetensors
        from safetensors.torch import load_file

        # Try to read header only first
        try:
            print(f"Reading metadata from: {file_path}")
            metadata = safetensors.safe_open(file_path, framework="pt").metadata()
            if metadata:
                print(f"Found {len(metadata)} metadata entries")
                return metadata
            else:
                print("No metadata found in header")
        except Exception as e:
            print(f"Error reading metadata from header: {e}")

            # Try loading the full file
            try:
                print("Attempting to load full file")
                tensors = load_file(file_path, device="cpu")
                if hasattr(tensors, "metadata") and tensors.metadata:
                    return tensors.metadata
                else:
                    print("No metadata found in full file")
            except Exception as e2:
                print(f"Error loading full file: {e2}")

    except ImportError:
        print("safetensors library not available. Install with: pip install safetensors")

    return {}


def read_civitai_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Read civitai.info file if available

    Args:
        file_path: Path to the safetensors file

    Returns:
        Dictionary of civitai info or None
    """
    civitai_path = f"{file_path}.civitai.info"
    if os.path.exists(civitai_path):
        try:
            print(f"Reading civitai info from: {civitai_path}")
            with open(civitai_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading civitai info: {e}")
    else:
        print(f"No civitai info file found at: {civitai_path}")

    return None


def print_metadata_details(metadata: Dict[str, Any], max_length: int = 200):
    """
    Print metadata details in a readable format

    Args:
        metadata: Dictionary of metadata
        max_length: Maximum length to print for each value
    """
    print("\n=== METADATA DETAILS ===")

    for key, value in sorted(metadata.items()):
        value_str = str(value)
        if len(value_str) > max_length:
            value_str = value_str[:max_length] + "... (truncated)"

        print(f"\n--- {key} ---")
        print(value_str)


def main():
    parser = argparse.ArgumentParser(description="Inspect LoRA metadata in detail")
    parser.add_argument("file_path", help="Path to the LoRA file")
    parser.add_argument("--json", action="store_true", help="Output metadata as JSON")
    parser.add_argument(
        "--max-length", type=int, default=500, help="Maximum length to print for each value"
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1

    # Get metadata from safetensors
    metadata = inspect_safetensors_metadata(args.file_path)

    # Get civitai info if available
    civitai_info = read_civitai_info(args.file_path)

    # If JSON output is requested
    if args.json:
        output = {"metadata": metadata, "civitai_info": civitai_info}
        print(json.dumps(output, indent=2))
    else:
        # Print metadata details
        if metadata:
            print_metadata_details(metadata, args.max_length)
        else:
            print("\nNo metadata found in safetensors file")

        # Print civitai info details
        if civitai_info:
            print("\n=== CIVITAI INFO ===")
            for key, value in sorted(civitai_info.items()):
                value_str = str(value)
                if len(value_str) > args.max_length:
                    value_str = value_str[: args.max_length] + "... (truncated)"

                print(f"\n--- {key} ---")
                print(value_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
