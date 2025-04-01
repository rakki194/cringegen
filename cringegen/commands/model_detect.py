"""
Command-line interface for model detection and optimization.

This command detects model architecture and family from a checkpoint name,
and suggests optimal parameters and prompt prefixes.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from ..utils.logger import print_colored_warning
from ..utils.model_utils import ModelOptimizer
from ..data.model_taxonomy import get_model_architecture_defaults, get_model_family_prefix

def add_subparser(subparsers, parent_parser=None):
    """Add the 'model-detect' subparser to the main parser."""
    parser = subparsers.add_parser(
        'model-detect',
        parents=[parent_parser] if parent_parser else [],
        help='Detect model architecture and suggest optimal parameters',
        description='Analyze a checkpoint file to detect its architecture and family, '
                    'and suggest optimal parameters for generation.'
    )
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint file or model name to analyze'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        help='Image width to check for optimization'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        help='Image height to check for optimization'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Example prompt to optimize for the model'
    )
    
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default='',
        help='Example negative prompt to optimize for the model'
    )
    
    parser.add_argument(
        '--no-tag-injection',
        action='store_true',
        default=False,
        help='Disable automatic tag injection in prompt optimization'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        default=False,
        help='Output results in JSON format'
    )
    
    parser.set_defaults(func=run_command)

def print_model_info(model_name: str, architecture: str, family: str) -> None:
    """Print information about the detected model."""
    print("\n=== Model Detection Results ===")
    print(f"Model:        {model_name}")
    print(f"Architecture: {architecture}")
    print(f"Family:       {family}")

def print_architecture_defaults(architecture: str) -> None:
    """Print default parameters for the detected architecture."""
    if architecture == "unknown":
        print("\nCould not detect model architecture.")
        return
    
    defaults = get_model_architecture_defaults(architecture)
    
    print("\n=== Architecture Default Parameters ===")
    print(f"Optimal resolution: {defaults['optimal_resolution'][0]}×{defaults['optimal_resolution'][1]} ({defaults['optimal_pixel_count']} pixels)")
    print(f"Default steps:      {defaults.get('default_steps', 'N/A')}")
    print(f"Default CFG:        {defaults.get('default_cfg', 'N/A')}")
    print(f"Default sampler:    {defaults.get('default_sampler', 'N/A')}")
    print(f"Default scheduler:  {defaults.get('default_scheduler', 'N/A')}")
    print(f"Prompt prefix:      {defaults.get('prompt_prefix', 'N/A')}")
    print(f"Negative prefix:    {defaults.get('negative_prefix', 'N/A')}")

def print_family_info(family: str) -> None:
    """Print information about the detected model family."""
    if family == "unknown":
        print("\nCould not detect model family.")
        return
    
    family_info = get_model_family_prefix(family)
    
    print("\n=== Model Family Information ===")
    print(f"Family:         {family}")
    print(f"Prompt prefix:  {family_info.get('prompt_prefix', 'N/A')}")
    
    if family_info.get('trigger_words'):
        print(f"Trigger words:  {', '.join(family_info['trigger_words'])}")
    
    if 'default_cfg' in family_info:
        print(f"Recommended CFG: {family_info['default_cfg']}")
    
    if 'default_steps' in family_info:
        print(f"Recommended steps: {family_info['default_steps']}")

def print_prompt_optimization(optimizer: ModelOptimizer, prompt: str, negative_prompt: str) -> None:
    """Print prompt optimization results."""
    if not prompt:
        return
    
    optimized_prompt = optimizer.inject_model_prefix(prompt)
    optimized_negative = optimizer.inject_negative_prefix(negative_prompt)
    
    print("\n=== Prompt Optimization ===")
    print(f"Original prompt:  {prompt}")
    print(f"Optimized prompt: {optimized_prompt}")
    
    if negative_prompt:
        print(f"\nOriginal negative prompt:  {negative_prompt}")
        print(f"Optimized negative prompt: {optimized_negative}")

def print_resolution_check(optimizer: ModelOptimizer, width: int, height: int) -> None:
    """Print resolution check results."""
    if not width or not height:
        return
    
    print("\n=== Resolution Check ===")
    is_optimal = optimizer.check_resolution(width, height)
    
    if is_optimal:
        print(f"Resolution {width}×{height} is optimal for this model.")
    else:
        optimal_width, optimal_height = optimizer.get_optimal_resolution(width, height)
        print(f"Suggested optimal resolution: {optimal_width}×{optimal_height}")

def run_command(args):
    """Run the model-detect command."""
    checkpoint_path = args.checkpoint
    
    # If path is provided, extract just the filename
    if os.path.exists(checkpoint_path):
        model_name = os.path.basename(checkpoint_path)
    else:
        model_name = checkpoint_path
    
    # Create optimizer instance
    optimizer = ModelOptimizer(model_name, args.no_tag_injection)
    architecture, family = optimizer.architecture, optimizer.family
    
    if args.json:
        import json
        
        result = {
            "model": model_name,
            "architecture": architecture,
            "family": family,
            "architecture_defaults": get_model_architecture_defaults(architecture),
            "family_settings": get_model_family_prefix(family),
        }
        
        if args.width and args.height:
            is_optimal = optimizer.check_resolution(args.width, args.height)
            optimal_width, optimal_height = optimizer.get_optimal_resolution(args.width, args.height)
            
            result["resolution"] = {
                "current": {"width": args.width, "height": args.height},
                "is_optimal": is_optimal,
                "suggested": {"width": optimal_width, "height": optimal_height}
            }
        
        if args.prompt:
            result["prompt"] = {
                "original": args.prompt,
                "optimized": optimizer.inject_model_prefix(args.prompt)
            }
            
            if args.negative_prompt:
                result["negative_prompt"] = {
                    "original": args.negative_prompt,
                    "optimized": optimizer.inject_negative_prefix(args.negative_prompt)
                }
        
        print(json.dumps(result, indent=2))
        return
    
    # Print standard output
    print_model_info(model_name, architecture, family)
    print_architecture_defaults(architecture)
    print_family_info(family)
    
    if args.width and args.height:
        print_resolution_check(optimizer, args.width, args.height)
    
    if args.prompt:
        print_prompt_optimization(optimizer, args.prompt, args.negative_prompt)
    
    print("\n=== Recommended Generation Parameters ===")
    parameters = optimizer.get_optimized_parameters()
    for key, value in parameters.items():
        print(f"{key}: {value}")
    
    # Print comprehensive command example
    if args.width and args.height and args.prompt:
        print("\n=== Example Generation Command ===")
        optimized_prompt = optimizer.inject_model_prefix(args.prompt)
        optimized_negative = optimizer.inject_negative_prefix(args.negative_prompt)
        
        # Get optimal resolution that maintains aspect ratio
        optimal_width, optimal_height = optimizer.get_optimal_resolution(args.width, args.height)
        
        print(f"cringegen generate \\")
        print(f"    --checkpoint \"{model_name}\" \\")
        print(f"    --prompt \"{optimized_prompt}\" \\")
        if optimized_negative:
            print(f"    --negative-prompt \"{optimized_negative}\" \\")
        print(f"    --width {optimal_width} \\")
        print(f"    --height {optimal_height} \\")
        print(f"    --steps {parameters.get('steps', 30)} \\")
        print(f"    --cfg {parameters.get('cfg', 7.0)} \\")
        print(f"    --sampler {parameters.get('sampler', 'euler_a')} \\")
        print(f"    --scheduler {parameters.get('scheduler', 'normal')}")
    
    print("")  # Add a newline at the end 