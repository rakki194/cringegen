"""
LLM-powered NSFW furry prompt generation with NoobAI image generation.

This command uses qwq:latest to generate NSFW furry prompts specifically optimized for NoobAI models,
and then automatically generates an image using the NoobAI command.
"""

import logging
import random
import time
import os
import re
from typing import Dict, List, Any, Optional

from ..utils.ollama_api import OllamaAPIClient
from ..utils.comfy_api import (
    check_comfy_server,
    get_available_checkpoints,
    get_matching_checkpoint,
    queue_prompt,
    check_generation_status,
    get_matching_lora,
)
from ..utils.file_utils import (
    copy_image_from_comfyui,
    rsync_image_from_comfyui,
    open_images_with_imv,
)
from ..utils.model_utils import ModelOptimizer, is_anthro_subject, is_feral_subject
from ..workflows.furry import create_nsfw_furry_workflow

logger = logging.getLogger(__name__)

# Create a client instance for Ollama
ollama_client = OllamaAPIClient()

# System prompt for NoobAI NSFW prompt generation
NOOBAI_NSFW_SYSTEM_PROMPT = """You are an expert prompt creator for AI image generation, specializing in NSFW furry art with the NoobAI model family.

Your task is to create detailed and effective NSFW furry prompts that work well with NoobAI models. These models perform best with comma-separated tag-based prompts focused on:
- Character details (species, gender, physical attributes)
- Clothing and accessories (if applicable)
- Poses and expressions
- Explicit anatomical details (for NSFW content)
- Background/setting elements
- Artistic style indicators

FORMAT GUIDELINES:
- Structure prompts as comma-separated tags
- Place most important elements first (character description, pose, main action)
- Include NSFW-specific tags as appropriate for the requested intensity level
- Avoid extremely long descriptions of any single element
- No need to include basic quality boosters like "masterpiece, best quality" as these will be added automatically
- If the character is anthro, include "anthro" in the prompt
- If the character is feral, include relevant indicators like "feral" or "quadruped"

INTENSITY LEVELS:
- Suggestive: Implied nudity, flirtatious poses, minimal explicit content
- Mild: Partial nudity, suggestive poses, limited explicit content
- Explicit: Full nudity, sexual poses/acts clearly visible
- Hardcore: Explicit sexual activity with detailed anatomical focus

EXAMPLE GOOD PROMPT:
anthro, female, wolf, red fur, white underbelly, bedroom, laying on back, spread legs, paw on chest, bedroom, bedsheets, detailed background

OUTPUT FORMAT:
Provide only the final prompt with no explanations or commentary. Just the comma-separated tags.
"""

def add_subparser(subparsers, parent_parser=None):
    """Add the llm-noobai-nsfw command to the CLI.
    
    The following arguments are inherited from the parent parser:
    --verbose, --debug, --log-file, --log-level, --comfy-url, --show
    """
    parser = subparsers.add_parser(
        "llm-noobai-nsfw",
        parents=[parent_parser] if parent_parser else [],
        help="Generate NSFW furry prompts with LLM and create images with NoobAI",
        description="Use qwq:latest to generate NSFW furry prompts optimized for NoobAI models, and then generate images.",
        conflict_handler="resolve"  # Resolve conflicts with parent parser
    )
    
    # Core LLM parameters
    parser.add_argument(
        "--species",
        type=str,
        default="",
        help="Species for the character (e.g., wolf, fox, dragon)"
    )
    
    parser.add_argument(
        "--gender",
        type=str,
        default="",
        choices=["male", "female", "intersex", "ambiguous"],
        help="Gender for the character"
    )
    
    parser.add_argument(
        "--intensity",
        type=str,
        choices=["suggestive", "mild", "explicit", "hardcore"],
        default="explicit",
        help="Intensity of NSFW content"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of prompts/images to generate"
    )
    
    parser.add_argument(
        "--theme",
        type=str,
        help="Theme or setting for the image (e.g., beach, forest, bedroom)"
    )
    
    parser.add_argument(
        "--duo",
        action="store_true",
        help="Generate a scene with two characters"
    )
    
    parser.add_argument(
        "--species2",
        type=str,
        help="Species for the second character (when using --duo)"
    )
    
    parser.add_argument(
        "--gender2",
        type=str,
        choices=["male", "female", "intersex", "ambiguous"],
        help="Gender for the second character (when using --duo)"
    )
    
    parser.add_argument(
        "--kinks",
        type=str,
        help="Comma-separated list of kinks to include"
    )
    
    parser.add_argument(
        "--anthro",
        action="store_true",
        help="Generate anthro character(s)"
    )
    
    parser.add_argument(
        "--feral",
        action="store_true",
        help="Generate feral character(s)"
    )
    
    parser.add_argument(
        "--colors",
        type=str,
        help="Comma-separated list of colors for the character's fur/scales"
    )
    
    parser.add_argument(
        "--outfit",
        type=str,
        help="Description of clothing/outfit"
    )
    
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Custom prompt instructions to guide the LLM"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM generation (higher = more creative)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwq:latest",
        help="LLM model to use (default: qwq:latest)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for generation randomness"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="NoobAI checkpoint to use (defaults to first available NoobAI model)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (will be automatically optimized if needed)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (will be automatically optimized if needed)"
    )
    
    parser.add_argument(
        "--lora",
        type=str,
        default="",
        help="LoRA to use with the model (can be a name or path)"
    )
    
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=0.8,
        help="Strength of the LoRA (default: 0.8)"
    )
    
    parser.add_argument(
        "--loras",
        type=str,
        nargs="+",
        default=[],
        help="Additional LoRAs to use (can specify multiple)"
    )
    
    parser.add_argument(
        "--lora-weights",
        type=float,
        nargs="+",
        default=[],
        help="Weights for additional LoRAs (should match --loras count)"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Additional negative prompt (will be automatically optimized)"
    )
    
    # Output options (copied from noobai command)
    parser.add_argument(
        "--copy-output",
        action="store_true",
        help="Copy generated images to output directory"
    )
    
    parser.add_argument(
        "--comfy-output-dir",
        type=str,
        default="/home/kade/toolkit/diffusion/comfy/ComfyUI/output",
        help="ComfyUI output directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated images"
    )
    
    # Remote options (copied from noobai command)
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use SSH to copy images from a remote ComfyUI instance"
    )
    
    parser.add_argument(
        "--ssh-host",
        type=str,
        default="otter_den",
        help="SSH hostname for remote ComfyUI instance"
    )
    
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=1487,
        help="SSH port for remote ComfyUI instance"
    )
    
    parser.add_argument(
        "--ssh-user",
        type=str,
        default="kade",
        help="SSH username for remote ComfyUI instance"
    )
    
    parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key file for remote ComfyUI instance"
    )
    
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Don't generate the image, just output the LLM-generated prompt"
    )
    
    # Advanced generation options
    parser.add_argument(
        "--pag",
        action="store_true",
        help="Enable Perp. Attention Guidance (improves shapes and overall composition)"
    )
    
    parser.add_argument(
        "--split-sigmas",
        type=float,
        default=None,
        help="Enable split sigmas with specified value (e.g. 0.5, 0.75, etc.)"
    )
    
    parser.add_argument(
        "--detail-daemon",
        action="store_true",
        help="Enable Detail Daemon (improves fine details, especially in faces)"
    )
    
    parser.add_argument(
        "--deepshrink",
        action="store_true",
        help="Enable DeepShrink (generates at higher resolution and shrinks intelligently)"
    )
    
    # Set the function to call
    parser.set_defaults(func=generate_llm_noobai_nsfw)
    
    return parser


def strip_thinking_field(text: str) -> str:
    """Remove the LLM thinking field from generated text.
    
    The thinking field is enclosed in <think>...</think> tags.
    
    Args:
        text: The text to process
        
    Returns:
        The text with the thinking field removed
    """
    # Remove any content within <think>...</think> tags
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def generate_llm_noobai_nsfw(args):
    """Generate NSFW furry prompts with LLM and create images with NoobAI."""
    # Track copied images for --show flag
    copied_images = []
    
    # Set up seed
    if args.seed == -1:
        seed = random.randint(1, 1000000)
    else:
        seed = args.seed
        random.seed(seed)
    
    # Build LLM prompt
    llm_prompt = build_llm_prompt(args)
    
    # Generate prompts with LLM
    logger.info(f"Generating NSFW furry prompts with {args.model}...")
    
    for i in range(args.count):
        if args.count > 1:
            logger.info(f"Generating prompt {i+1}/{args.count}")
        
        # Get prompt from LLM - using stream mode
        try:
            # Start the streaming response
            logger.info("Streaming LLM response...")
            
            # Stream directly to the console as it's generated
            stream_response = ollama_client.generate(
                prompt=llm_prompt,
                model=args.model,
                system=NOOBAI_NSFW_SYSTEM_PROMPT,
                temperature=args.temperature,
                stream=True,  # Enable streaming
            )
            
            raw_generated_prompt = stream_response.get("response", "").strip()
            
            # Strip out the thinking field from the generated prompt
            generated_prompt = strip_thinking_field(raw_generated_prompt)
            
            # Log just the cleaned prompt
            logger.info(f"Using prompt: {generated_prompt}")
            
            # If no-generate flag is set, just output the prompt and continue
            if args.no_generate:
                continue
            
            # Generate the image with the NoobAI command
            logger.info("Generating image with NoobAI...")
            
            if i == 0 or args.count == 1:
                # Full image generation and display logic
                result = generate_noobai_image(args, generated_prompt, seed + i, copied_images)
            else:
                # For additional images, don't wait for completion if show/copy not needed
                if not (args.show or args.copy_output):
                    result = generate_noobai_image_without_waiting(args, generated_prompt, seed + i)
                else:
                    result = generate_noobai_image(args, generated_prompt, seed + i, copied_images)
            
        except Exception as e:
            logger.error(f"Error generating prompt or image: {e}")
            continue
    
    # Open images with imv if requested and we have any
    if args.show and copied_images:
        open_images_with_imv(copied_images)
    
    # Return the last result for reference
    return result


def build_llm_prompt(args):
    """Build the prompt for the LLM based on the command arguments."""
    prompt_parts = ["Generate a detailed NSFW furry prompt for NoobAI image generation."]
    
    if args.custom_prompt:
        # If custom prompt is provided, use it directly
        return args.custom_prompt
    
    # Add character information
    if args.species:
        character_type = "anthro" if args.anthro else ("feral" if args.feral else "")
        if character_type:
            prompt_parts.append(f"The character should be a {character_type} {args.gender} {args.species}.")
        else:
            prompt_parts.append(f"The character should be a {args.gender} {args.species}.")
    
    # Add duo information if requested
    if args.duo:
        if args.species2 and args.gender2:
            character_type = "anthro" if args.anthro else ("feral" if args.feral else "")
            if character_type:
                prompt_parts.append(f"Include a second {character_type} {args.gender2} {args.species2} character.")
            else:
                prompt_parts.append(f"Include a second {args.gender2} {args.species2} character.")
        else:
            prompt_parts.append("Include a second character with the first one.")
    
    # Add colors if specified
    if args.colors:
        prompt_parts.append(f"The character's fur/scales should be {args.colors}.")
    
    # Add outfit if specified
    if args.outfit:
        prompt_parts.append(f"The character should be wearing {args.outfit}.")
    
    # Add theme/setting if specified
    if args.theme:
        prompt_parts.append(f"The setting should be {args.theme}.")
    
    # Add NSFW intensity
    prompt_parts.append(f"The content should be {args.intensity} level NSFW.")
    
    # Add kinks if specified
    if args.kinks:
        prompt_parts.append(f"Include these kinks: {args.kinks}.")
    
    # Join all parts into a single prompt
    return " ".join(prompt_parts)


def generate_noobai_image(args, generated_prompt, seed, copied_images):
    """Generate an image with NoobAI using the generated prompt."""
    # Check if ComfyUI server is available
    is_available, message = check_comfy_server(args.comfy_url)
    if not is_available:
        logger.error(message)
        logger.error("Image generation skipped. Start ComfyUI and try again.")
        return
    
    # Find appropriate checkpoint
    checkpoint_name = ""
    if args.checkpoint:
        # Use specified checkpoint
        checkpoint_path = get_matching_checkpoint(args.checkpoint, args.comfy_url)
        if checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_path)
            logger.info(f"Using specified checkpoint: {checkpoint_name}")
        else:
            logger.warning(f"Specified checkpoint '{args.checkpoint}' not found.")
    
    if not checkpoint_name:
        # Look for any NoobAI model
        available_checkpoints = get_available_checkpoints(args.comfy_url)
        noobai_models = [cp for cp in available_checkpoints if "noob" in cp.lower()]
        
        if noobai_models:
            checkpoint_name = noobai_models[0]
            logger.info(f"Automatically selected NoobAI model: {checkpoint_name}")
        else:
            # Fall back to any available checkpoint
            if available_checkpoints:
                checkpoint_name = available_checkpoints[0]
                logger.warning(f"No NoobAI models found. Using: {checkpoint_name}")
            else:
                logger.error("No checkpoints found on the server.")
                return
    
    # Find LoRAs if specified
    lora_name = ""
    if args.lora:
        # Try to match the LoRA name to an available LoRA
        lora_path = get_matching_lora(args.lora, args.comfy_url)
        if lora_path:
            lora_name = lora_path
            logger.info(f"Found matching LoRA: {lora_name}")
        else:
            logger.warning(f"Specified LoRA '{args.lora}' not found. Proceeding without LoRA.")
    
    # Find additional LoRAs if specified
    loras = []
    lora_weights = []
    
    if args.loras:
        for lora in args.loras:
            lora_path = get_matching_lora(lora, args.comfy_url)
            if lora_path:
                loras.append(lora_path)
                logger.info(f"Found additional LoRA: {lora_path}")
            else:
                logger.warning(f"Additional LoRA '{lora}' not found and will be skipped.")
        
        # Use provided weights or default to 0.8 for all
        if args.lora_weights:
            if len(args.lora_weights) >= len(loras):
                lora_weights = args.lora_weights[:len(loras)]
            else:
                # If fewer weights than loras, use provided weights and default for the rest
                lora_weights = args.lora_weights + [0.8] * (len(loras) - len(args.lora_weights))
        else:
            lora_weights = [0.8] * len(loras)
    
    # Set up the ModelOptimizer for NoobAI models
    optimizer = ModelOptimizer(checkpoint_name, disable_tag_injection=False)
    
    # Process the LLM-generated prompt
    optimized_prompt = optimizer.inject_model_prefix(generated_prompt)
    
    # Optimize negative prompt
    optimized_negative = optimizer.inject_negative_prefix(args.negative_prompt or "")
    
    # Get and apply optimal parameters for generation
    optimal_params = optimizer.get_optimized_parameters()
    steps = optimal_params.get("steps", 30)
    cfg = optimal_params.get("cfg", 7.0)
    sampler = optimal_params.get("sampler", "dpmpp_2m")
    scheduler = optimal_params.get("scheduler", "karras")
    
    # Check and optimize resolution
    width, height = args.width, args.height
    if not optimizer.check_resolution(width, height):
        width, height = optimizer.get_optimal_resolution(width, height)
        logger.info(f"Optimizing resolution to {width}×{height}")
    
    # Detect background type
    bg_type = optimizer.detect_background_type(generated_prompt)
    
    # Detect subject type for negative prompt
    is_anthro = is_anthro_subject(generated_prompt)
    is_feral = is_feral_subject(generated_prompt)
    subject_type = "anthro" if is_anthro else ("feral" if is_feral else "unknown")
    
    # Store the prompt for negative prompt optimization
    optimizer._prompt_for_negative = generated_prompt
    
    # Log what we're going to do
    logger.info(f"Generating image with NoobAI optimizations")
    logger.info(f"Model: {checkpoint_name}")
    logger.info(f"Architecture: {optimizer.architecture}")
    logger.info(f"Model family: {optimizer.family}")
    logger.info(f"Resolution: {width}×{height}")
    logger.info(f"Seed: {seed}")
    
    if bg_type:
        logger.info(f"Background detected: {bg_type}")
    
    logger.info(f"Subject type: {subject_type}")
    
    # Only log the cleaned prompt here
    logger.info("Optimized prompt: " + optimized_prompt)
    
    if args.negative_prompt:
        logger.info("Original negative: " + args.negative_prompt)
    logger.info("Optimized negative: " + optimized_negative)
    
    # Log LoRA information if provided
    if args.lora:
        if lora_name:
            logger.info(f"Using LoRA: {lora_name} (strength: {args.lora_strength})")
        else:
            logger.warning("LoRA not found, generating without LoRA")
    
    # Log additional LoRAs if provided
    if loras:
        logger.info(f"Using {len(loras)} additional LoRAs:")
        for i, (lora, weight) in enumerate(zip(loras, lora_weights)):
            logger.info(f"  {i+1}. {lora} (weight: {weight})")
    
    # Create the workflow
    workflow = create_nsfw_furry_workflow(
        checkpoint=checkpoint_name,
        prompt=optimized_prompt,
        negative_prompt=optimized_negative,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        lora=lora_name,
        lora_strength=args.lora_strength,
        loras=loras,
        lora_weights=lora_weights,
        use_pag=args.pag,
        split_sigmas=args.split_sigmas,
        use_detail_daemon=args.detail_daemon,
        use_deepshrink=args.deepshrink,
    )
    
    # Queue the prompt
    prompt_id = queue_prompt(workflow, args.comfy_url)
    logger.info(f"Queued prompt with ID: {prompt_id}")
    
    # Extract prompt ID string if needed
    if isinstance(prompt_id, dict) and "prompt_id" in prompt_id:
        prompt_id_str = prompt_id["prompt_id"]
    else:
        prompt_id_str = str(prompt_id)
    
    # Handle image waiting, copying, and showing
    # We need to monitor completion if either --copy-output or --show are specified
    if args.copy_output or args.show:
        try:
            # Poll for generation status
            max_poll_time = 300  # 5 minutes max
            poll_interval = 2    # Check every 2 seconds
            start_time = time.time()
            
            logger.info(f"Monitoring generation status for prompt {prompt_id_str}")
            
            # Poll until completed or timeout
            last_progress = 0
            status = None
            while time.time() - start_time < max_poll_time:
                status = check_generation_status(prompt_id_str, args.comfy_url)
                
                # Print progress updates
                if status["status"] == "pending":
                    if time.time() - start_time > 10:
                        logger.info(f"Generation pending... ({int(time.time() - start_time)}s elapsed)")
                elif status["status"] == "processing":
                    current_progress = int(status["progress"] * 100)
                    if current_progress >= last_progress + 10 or (time.time() - start_time) % 10 < 2:
                        logger.info(f"Generation in progress: {current_progress}% ({int(time.time() - start_time)}s elapsed)")
                        last_progress = current_progress
                elif status["status"] == "completed":
                    logger.info(f"Generation completed in {int(time.time() - start_time)}s!")
                    break
                elif status["status"] == "error":
                    logger.error(f"Generation error: {status['error']}")
                    break
                
                # Wait before polling again
                time.sleep(poll_interval)
            
            # Check if we timed out
            if time.time() - start_time >= max_poll_time:
                logger.warning(f"Generation timed out after {max_poll_time}s")
                return
            
            # Get the image filename if status is completed
            if status and status["status"] == "completed" and status["images"]:
                image_filename = status["images"][0]["filename"]
                logger.info(f"Found image: {image_filename}")
                
                # Only copy the image if --copy-output is specified
                if args.copy_output:
                    output_filename = f"llm_noobai_nsfw_{seed}"
                    
                    if args.remote:
                        # Check for required SSH parameters
                        if not args.ssh_host:
                            logger.error("SSH host is required when using --remote")
                            return
                        
                        logger.info(f"Using rsync over SSH to copy image from {args.ssh_host}")
                        copied_path = rsync_image_from_comfyui(
                            image_filename,
                            args.ssh_host,
                            args.comfy_output_dir,
                            args.output_dir,
                            output_filename,
                            ssh_port=args.ssh_port,
                            ssh_user=args.ssh_user,
                            ssh_key=args.ssh_key,
                        )
                        
                        if copied_path:
                            logger.info(f"Copied image to {copied_path}")
                            copied_images.append(copied_path)
                        else:
                            logger.warning("Failed to copy image via rsync")
                    else:
                        # Local copy
                        copied_path = copy_image_from_comfyui(
                            image_filename,
                            args.comfy_output_dir,
                            args.output_dir,
                            output_filename,
                        )
                        
                        if copied_path:
                            logger.info(f"Copied image to {copied_path}")
                            copied_images.append(copied_path)
                        else:
                            logger.warning("Failed to copy image")
                
                # If --show is specified but we didn't copy the image, we need to get the image path
                elif args.show:
                    # For remote images, we need to copy them to display
                    if args.remote:
                        # Check for required SSH parameters
                        if not args.ssh_host:
                            logger.error("SSH host is required when using --remote")
                            return
                        
                        logger.info(f"Using rsync over SSH to copy image for viewing from {args.ssh_host}")
                        temp_output_dir = "/tmp/cringegen_view"
                        os.makedirs(temp_output_dir, exist_ok=True)
                        
                        copied_path = rsync_image_from_comfyui(
                            image_filename,
                            args.ssh_host,
                            args.comfy_output_dir,
                            temp_output_dir,
                            f"view_llm_noobai_nsfw_{seed}",
                            ssh_port=args.ssh_port,
                            ssh_user=args.ssh_user,
                            ssh_key=args.ssh_key,
                        )
                        
                        if copied_path:
                            logger.info(f"Copied image to temporary location for viewing: {copied_path}")
                            copied_images.append(copied_path)
                        else:
                            logger.warning("Failed to copy image via rsync for viewing")
                    else:
                        # For local images, we can just use the direct path
                        image_path = os.path.join(args.comfy_output_dir, image_filename)
                        if os.path.exists(image_path):
                            logger.info(f"Using direct path for viewing: {image_path}")
                            copied_images.append(image_path)
                        else:
                            logger.warning(f"Image file not found at {image_path}")
            else:
                logger.warning("No image was generated or image path is empty")
                
        except Exception as e:
            logger.error(f"Error during image generation or copying: {e}")
    else:
        logger.info("Image queued for generation. Use --copy-output to save the image or --show to view it.")
    
    # Return the result for reference
    return {
        "llm_generated_prompt": generated_prompt,
        "optimized_prompt": optimized_prompt,
        "original_negative": args.negative_prompt,
        "optimized_negative": optimized_negative,
        "seed": seed,
        "background_type": bg_type or "none",
        "model": checkpoint_name,
    }


def generate_noobai_image_without_waiting(args, generated_prompt, seed):
    """Generate an image with NoobAI without waiting for completion.
    Used for batch generation when not needing to display/copy images.
    """
    # Check if ComfyUI server is available
    is_available, message = check_comfy_server(args.comfy_url)
    if not is_available:
        logger.error(message)
        logger.error("Image generation skipped. Start ComfyUI and try again.")
        return
    
    # Find appropriate checkpoint
    checkpoint_name = ""
    if args.checkpoint:
        # Use specified checkpoint
        checkpoint_path = get_matching_checkpoint(args.checkpoint, args.comfy_url)
        if checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_path)
            logger.info(f"Using specified checkpoint: {checkpoint_name}")
        else:
            logger.warning(f"Specified checkpoint '{args.checkpoint}' not found.")
    
    if not checkpoint_name:
        # Look for any NoobAI model
        available_checkpoints = get_available_checkpoints(args.comfy_url)
        noobai_models = [cp for cp in available_checkpoints if "noob" in cp.lower()]
        
        if noobai_models:
            checkpoint_name = noobai_models[0]
            logger.info(f"Automatically selected NoobAI model: {checkpoint_name}")
        else:
            # Fall back to any available checkpoint
            if available_checkpoints:
                checkpoint_name = available_checkpoints[0]
                logger.warning(f"No NoobAI models found. Using: {checkpoint_name}")
            else:
                logger.error("No checkpoints found on the server.")
                return
    
    # Same LoRA handling as in full function
    lora_name = ""
    if args.lora:
        lora_path = get_matching_lora(args.lora, args.comfy_url)
        if lora_path:
            lora_name = lora_path
        else:
            logger.warning(f"Specified LoRA '{args.lora}' not found. Proceeding without LoRA.")
    
    loras = []
    lora_weights = []
    
    if args.loras:
        for lora in args.loras:
            lora_path = get_matching_lora(lora, args.comfy_url)
            if lora_path:
                loras.append(lora_path)
            else:
                logger.warning(f"Additional LoRA '{lora}' not found and will be skipped.")
        
        if args.lora_weights:
            if len(args.lora_weights) >= len(loras):
                lora_weights = args.lora_weights[:len(loras)]
            else:
                lora_weights = args.lora_weights + [0.8] * (len(loras) - len(args.lora_weights))
        else:
            lora_weights = [0.8] * len(loras)
    
    # Set up the ModelOptimizer and optimize the prompt
    optimizer = ModelOptimizer(checkpoint_name, disable_tag_injection=False)
    optimizer._prompt_for_negative = generated_prompt
    
    optimized_prompt = optimizer.inject_model_prefix(generated_prompt)
    optimized_negative = optimizer.inject_negative_prefix(args.negative_prompt or "")
    
    # Get optimal parameters
    optimal_params = optimizer.get_optimized_parameters()
    steps = optimal_params.get("steps", 30)
    cfg = optimal_params.get("cfg", 7.0)
    sampler = optimal_params.get("sampler", "dpmpp_2m")
    scheduler = optimal_params.get("scheduler", "karras")
    
    # Check and optimize resolution
    width, height = args.width, args.height
    if not optimizer.check_resolution(width, height):
        width, height = optimizer.get_optimal_resolution(width, height)
    
    # Create the workflow
    workflow = create_nsfw_furry_workflow(
        checkpoint=checkpoint_name,
        prompt=optimized_prompt,
        negative_prompt=optimized_negative,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        lora=lora_name,
        lora_strength=args.lora_strength,
        loras=loras,
        lora_weights=lora_weights,
        use_pag=args.pag,
        split_sigmas=args.split_sigmas,
        use_detail_daemon=args.detail_daemon,
        use_deepshrink=args.deepshrink,
    )
    
    # Queue the prompt without waiting for completion
    prompt_id = queue_prompt(workflow, args.comfy_url)
    logger.info(f"Queued prompt with ID: {prompt_id} (not waiting for completion)")
    
    # Return the result for reference
    return {
        "llm_generated_prompt": generated_prompt,
        "optimized_prompt": optimized_prompt,
        "original_negative": args.negative_prompt,
        "optimized_negative": optimized_negative,
        "seed": seed,
        "model": checkpoint_name,
    } 