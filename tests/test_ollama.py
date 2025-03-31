#!/usr/bin/env python
"""
Test script for Ollama API client
"""

import argparse
import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import cringegen
sys.path.append(str(Path(__file__).parent.parent))

from cringegen.utils.ollama_api import default_client
from cringegen.prompt_generation.llm_generator import LLMPromptGenerator, BrainstormGenerator


def test_caption_generation(args):
    """Test caption generation"""
    print(f"Testing caption generation with {args.model}...")
    print(f"Subject: {args.subject}")
    
    if args.model:
        default_client.default_model = args.model
    
    generator = LLMPromptGenerator(
        subject=args.subject,
        species=args.species,
        gender=args.gender,
        background=args.background,
        style=args.style,
        temperature=args.temperature,
        show_thinking=args.thinking,
    )
    
    caption = generator.generate()
    
    print("\nGenerated Caption:")
    print("=================")
    print(caption)
    print("\n")


def test_nsfw_caption_generation(args):
    """Test NSFW caption generation"""
    print(f"Testing NSFW caption generation with {args.model}...")
    print(f"Subject: {args.subject}")
    print(f"NSFW Intensity: {args.nsfw_intensity}")
    
    if args.model:
        default_client.default_model = args.model
    
    generator = LLMPromptGenerator(
        subject=args.subject,
        species=args.species,
        gender=args.gender,
        background=args.background,
        style=args.style,
        temperature=args.temperature,
        nsfw=True,
        nsfw_intensity=args.nsfw_intensity,
        show_thinking=args.thinking,
    )
    
    caption = generator.generate()
    
    print("\nGenerated NSFW Caption:")
    print("=====================")
    print(caption)
    print("\n")


def test_brainstorm(args):
    """Test brainstorm functionality"""
    print(f"Testing brainstorm with {args.model}...")
    print(f"Concept: {args.concept}")
    
    if args.model:
        default_client.default_model = args.model
    
    generator = BrainstormGenerator(
        temperature=args.temperature,
        show_thinking=args.thinking,
    )
    variations = generator.generate_variations(args.concept)
    
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
        test_caption_generation(args)
    elif args.command == "nsfw-caption":
        test_nsfw_caption_generation(args)
    elif args.command == "brainstorm":
        test_brainstorm(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 