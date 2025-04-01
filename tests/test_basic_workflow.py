#!/usr/bin/env python
"""
Basic test script for cringegen
"""
import os
import sys
import argparse

# Add the parent directory to the path to import cringegen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cringegen.prompt_generation.furry_generator import FurryPromptGenerator, NsfwFurryPromptGenerator
from cringegen.utils.comfy_api import queue_prompt, get_image_path
from cringegen.workflows.furry import create_basic_furry_workflow, create_nsfw_furry_workflow

def test_prompt_generation():
    """Test prompt generation"""
    print("Testing FurryPromptGenerator...")
    
    # Test with default parameters
    generator = FurryPromptGenerator()
    prompt = generator.generate()
    negative_prompt = generator.get_negative_prompt()
    
    print(f"Generated prompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")
    print()
    
    # Test with specific parameters
    generator = FurryPromptGenerator(species="fox", gender="female")
    prompt = generator.generate()
    negative_prompt = generator.get_negative_prompt()
    
    print(f"Fox female prompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")
    print()
    
    # Test NSFW generator with different explicitness levels
    for level in range(1, 4):
        print(f"Testing NsfwFurryPromptGenerator with explicit_level={level}...")
        generator = NsfwFurryPromptGenerator(explicit_level=level)
        prompt = generator.generate()
        negative_prompt = generator.get_negative_prompt()
        
        print(f"NSFW (level {level}) prompt: {prompt}")
        print(f"Negative prompt: {negative_prompt}")
        print()

def test_workflow_generation():
    """Test workflow generation"""
    print("Testing workflow generation...")
    
    # Create a basic test workflow
    workflow = create_basic_furry_workflow(
        checkpoint="v1-5-pruned-emaonly.safetensors",  # Use a common checkpoint for testing
        lora="noob/beksinski-v1s1600.safetensors",  # Use a LoRA from user's collection
        prompt="masterpiece, best quality, wolf, anthro, forest, night",
        negative_prompt="low quality, worst quality, bad anatomy",
        seed=1234
    )
    
    # Print the workflow (simplified)
    print(f"Generated workflow with {len(workflow)} nodes")
    for node_id, node in workflow.items():
        print(f"Node {node_id}: {node['class_type']}")
    print()

def test_with_comfyui(checkpoint, lora):
    """Test with ComfyUI API
    
    Args:
        checkpoint: The checkpoint to use
        lora: The LoRA to use
    """
    print("Testing with ComfyUI API...")
    
    # Generate a prompt
    generator = FurryPromptGenerator(species="wolf", gender="male")
    prompt = generator.generate()
    negative_prompt = generator.get_negative_prompt()
    
    print(f"Generated prompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")
    
    # Create a workflow
    workflow = create_basic_furry_workflow(
        checkpoint=checkpoint,
        lora=lora,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=1234
    )
    
    # Queue the prompt
    print("Queueing prompt to ComfyUI...")
    try:
        result = queue_prompt(workflow)
        prompt_id = result.get('prompt_id')
        if prompt_id:
            print(f"Prompt queued with ID: {prompt_id}")
            print("Waiting for image...")
            image_paths = get_image_path(prompt_id)
            print(f"Image generated: {', '.join(image_paths)}")
        else:
            print(f"Error: Failed to get prompt ID")
    except Exception as e:
        print(f"Error communicating with ComfyUI: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test basic cringegen functionality")
    parser.add_argument("--test-comfyui", action="store_true", help="Test with ComfyUI API")
    parser.add_argument("--checkpoint", type=str, default="v1-5-pruned-emaonly.safetensors", help="Checkpoint model to use for ComfyUI test")
    parser.add_argument("--lora", type=str, default="noob/beksinski-v1s1600.safetensors", help="LoRA model to use for ComfyUI test")
    
    args = parser.parse_args()
    
    # Always run basic tests
    test_prompt_generation()
    test_workflow_generation()
    
    # Optionally test with ComfyUI
    if args.test_comfyui:
        test_with_comfyui(args.checkpoint, args.lora)

if __name__ == "__main__":
    main() 