"""
Script to extract activation text from a LoRA file
"""

import argparse
import json
import os
import re
import sys


def extract_activation_text(file_path, verbose=False):
    """
    Extract activation text from a LoRA file

    Args:
        file_path: Path to the LoRA file
        verbose: Whether to print verbose output

    Returns:
        List of activation text strings
    """
    activation_texts = []
    trigger_phrases = []

    try:
        import safetensors
        from safetensors.torch import load_file

        # Try reading header only first
        if verbose:
            print(f"Reading metadata from: {file_path}")
        metadata = safetensors.safe_open(file_path, framework="pt").metadata()

        if metadata:
            if verbose:
                print(f"Found {len(metadata)} metadata entries")

            # Extract activation text from ss_tag_frequency
            ss_tag_frequency = metadata.get("ss_tag_frequency")
            if ss_tag_frequency:
                if verbose:
                    print(f"Found ss_tag_frequency field")

                # Try to parse as JSON
                try:
                    tag_data = json.loads(ss_tag_frequency)

                    # Extract captions from tag frequency data
                    for dataset, tags in tag_data.items():
                        if verbose:
                            print(f"Dataset: {dataset}")
                        for tag, count in tags.items():
                            # Long tags with multiple words are likely captions/prompts
                            if len(tag.split()) > 5 and "this is" in tag.lower():
                                activation_texts.append(tag)
                                if verbose:
                                    print(f"Found activation text: {tag[:100]}...")

                    # Extract artist names for triggers
                    for dataset, tags in tag_data.items():
                        for tag, count in tags.items():
                            if tag.lower().startswith(("by ", "art by")):
                                trigger_phrases.append(tag)
                                if verbose:
                                    print(f"Found trigger phrase: {tag}")
                            elif re.search(r"by\s+[a-z0-9]+", tag.lower()):
                                trigger_phrases.append(tag)
                                if verbose:
                                    print(f"Found artist trigger: {tag}")
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"Failed to parse ss_tag_frequency as JSON: {e}")

            # Check for other activation text fields
            for key in ["ss_activation_text", "activation_text", "prompt", "ss_prompt"]:
                if key in metadata and metadata[key]:
                    if verbose:
                        print(f"Found field {key}: {metadata[key]}")
                    texts = metadata[key].strip().split(",")
                    activation_texts.extend([t.strip() for t in texts if t.strip()])

            # Check for trigger phrases
            for key in ["ss_trigger_phrases", "trigger_phrases", "trigger_words", "ss_trigger"]:
                if key in metadata and metadata[key]:
                    if verbose:
                        print(f"Found field {key}: {metadata[key]}")
                    texts = metadata[key].strip().split(",")
                    trigger_phrases.extend([t.strip() for t in texts if t.strip()])

    except ImportError:
        if verbose:
            print("safetensors library not available. Install with: pip install safetensors")
    except Exception as e:
        if verbose:
            print(f"Error extracting activation text: {e}")

    # Remove duplicates
    activation_texts = list(set(activation_texts))
    trigger_phrases = list(set(trigger_phrases))

    return activation_texts, trigger_phrases


def main():
    parser = argparse.ArgumentParser(description="Extract activation text from a LoRA file")
    parser.add_argument("file_path", help="Path to the LoRA file")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1

    activation_texts, trigger_phrases = extract_activation_text(
        args.file_path, verbose=not args.quiet
    )

    print("\n=== Activation Text ===")
    if activation_texts:
        for text in activation_texts:
            print(f"  {text}")
    else:
        print("No activation text found")

    print("\n=== Trigger Phrases ===")
    if trigger_phrases:
        for text in trigger_phrases:
            print(f"  {text}")
    else:
        print("No trigger phrases found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
