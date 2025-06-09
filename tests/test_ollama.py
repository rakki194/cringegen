#!/usr/bin/env python
"""
Test script for Ollama API client
"""

import argparse
import sys
import json
from pathlib import Path
import pytest
import requests

# Add the parent directory to the path so we can import cringegen
sys.path.append(str(Path(__file__).parent.parent))

from cringegen.utils.ollama_api import default_client
from cringegen.prompt_generation.llm_generator import LLMPromptGenerator, BrainstormGenerator


def ollama_available():
    try:
        requests.get("http://localhost:11434")
        return True
    except Exception:
        return False

ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama API is not running on localhost:11434"
)

@ollama
def test_caption_generation():
    """Test caption generation"""
    model = "dummy_model"
    subject = "dummy_subject"
    species = "dummy_species"
    gender = "dummy_gender"
    background = "dummy_background"
    style = "dummy_style"
    temperature = 0.7
    thinking = False
    print(f"Testing caption generation with {model}...")
    print(f"Subject: {subject}")
    
    generator = LLMPromptGenerator(
        subject=subject,
        species=species,
        gender=gender,
        background=background,
        style=style,
        temperature=temperature,
        show_thinking=thinking,
    )
    
    caption = generator.generate()
    
    print("\nGenerated Caption:")
    print("=================")
    print(caption)
    print("\n")


@ollama
def test_nsfw_caption_generation():
    """Test NSFW caption generation"""
    model = "dummy_model"
    subject = "dummy_subject"
    nsfw_intensity = "medium"
    species = "dummy_species"
    gender = "dummy_gender"
    background = "dummy_background"
    style = "dummy_style"
    temperature = 0.7
    thinking = False
    print(f"Testing NSFW caption generation with {model}...")
    print(f"Subject: {subject}")
    print(f"NSFW Intensity: {nsfw_intensity}")
    
    generator = LLMPromptGenerator(
        subject=subject,
        species=species,
        gender=gender,
        background=background,
        style=style,
        temperature=temperature,
        nsfw=True,
        nsfw_intensity=nsfw_intensity,
        show_thinking=thinking,
    )
    
    caption = generator.generate()
    
    print("\nGenerated NSFW Caption:")
    print("=====================")
    print(caption)
    print("\n")


@ollama
def test_brainstorm():
    """Test brainstorm functionality"""
    model = "dummy_model"
    concept = "dummy_concept"
    temperature = 0.8
    thinking = False
    print(f"Testing brainstorm with {model}...")
    print(f"Concept: {concept}")
    
    generator = BrainstormGenerator(
        temperature=temperature,
        show_thinking=thinking,
    )
    variations = generator.generate_variations(concept)
    
    print("\nVariations:")
    print("===========")
    print(json.dumps(variations, indent=2))
    print("\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Ollama API client")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Caption generation command
    caption_parser = subparsers.add_parser("caption", help="Test caption generation")
    caption_parser.add_argument("--subject", required=True, help="Subject for caption")
    caption_parser.add_argument("--species", help="Species for caption")
    caption_parser.add_argument("--gender", help="Gender for caption")
    caption_parser.add_argument("--background", help="Background for caption")
    caption_parser.add_argument("--style", help="Style for caption")
    caption_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    caption_parser.add_argument("--model", default="qwq:latest", help="Model to use")
    caption_parser.add_argument("--thinking", action="store_true", help="Show model's thinking process")
    
    # NSFW Caption generation command
    nsfw_caption_parser = subparsers.add_parser("nsfw-caption", help="Test NSFW caption generation")
    nsfw_caption_parser.add_argument("--subject", required=True, help="Subject for caption")
    nsfw_caption_parser.add_argument("--species", help="Species for caption")
    nsfw_caption_parser.add_argument("--gender", help="Gender for caption")
    nsfw_caption_parser.add_argument("--background", help="Background for caption")
    nsfw_caption_parser.add_argument("--style", help="Style for caption")
    nsfw_caption_parser.add_argument("--nsfw-intensity", choices=["mild", "moderate", "explicit"], 
                                    default="moderate", help="NSFW intensity level")
    nsfw_caption_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    nsfw_caption_parser.add_argument("--model", default="qwq:latest", help="Model to use")
    nsfw_caption_parser.add_argument("--thinking", action="store_true", help="Show model's thinking process")
    
    # Brainstorm command
    brainstorm_parser = subparsers.add_parser("brainstorm", help="Test brainstorm")
    brainstorm_parser.add_argument("--concept", required=True, help="Concept for brainstorming")
    brainstorm_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
    brainstorm_parser.add_argument("--model", default="qwq:latest", help="Model to use")
    brainstorm_parser.add_argument("--thinking", action="store_true", help="Show model's thinking process")
    
    args = parser.parse_args()
    
    if args.command == "caption":
        test_caption_generation()
    elif args.command == "nsfw-caption":
        test_nsfw_caption_generation()
    elif args.command == "brainstorm":
        test_brainstorm()
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 