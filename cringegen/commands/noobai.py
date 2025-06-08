"""
NoobAI-specific optimized image generation command for cringegen.

This command provides an ultra-streamlined way to generate images with NoobAI models,
automatically applying all the optimal prefixes, backgrounds, and parameters.
"""

import logging
import random
import os
import time

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
    copy_latest_images_from_comfyui,
    rsync_image_from_comfyui,
    rsync_latest_images_from_comfyui,
    open_images_with_imv,
)
from ..utils.model_utils import ModelOptimizer, is_anthro_subject, is_feral_subject
from ..workflows.furry import create_nsfw_furry_workflow

logger = logging.getLogger(__name__)


def add_subparser(subparsers, parent_parser=None):
    """Add the noobai command to the CLI.
    
    The following arguments are inherited from the parent parser:
    --verbose, --debug, --log-file, --log-level, --comfy-url, --show
    """
    parser = subparsers.add_parser(
        "noobai",
        parents=[parent_parser] if parent_parser else [],
        help="Generate an image using NoobAI models with fully optimized tags",
        description="Generate images with NoobAI models using optimal prefixes, backgrounds, and parameters.",
        conflict_handler="resolve"  # Resolve conflicts with parent parser
    )
    
    # Core parameters
    parser.add_argument(
        "prompt",
        type=str,
        help="Your base prompt (tags will be automatically optimized)"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt (will be automatically optimized)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="NoobAI checkpoint to use (defaults to first available NoobAI model)"
    )
    
    # Image parameters
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
        "--seed",
        type=int,
        default=-1,
        help="Seed for generation"
    )
    
    # Additional options
    parser.add_argument(
        "--no-scenery-porn",
        action="store_true",
        help="Disable 'scenery porn' background enhancement"
    )
    
    parser.add_argument(
        "--copy-output",
        action="store_true",
        help="Copy generated images to output directory"
    )
    
    # Output directories
    parser.add_argument(
        "--comfy-output-dir",
        type=str,
        default="/home/kade/comfy/ComfyUI/output",
        help="ComfyUI output directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated images"
    )
    
    # Remote options
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
    
    # LoRA parameters
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
    parser.set_defaults(func=generate_noobai_image)
    
    return parser


def generate_noobai_image(args):
    """Generate an image with NoobAI-specific optimizations."""
    # Track copied images for --show flag
    copied_images = []
    
    # Set up seed
    if args.seed == -1:
        seed = random.randint(1, 1000000)
    else:
        seed = args.seed
    
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
    
    # Set up the ModelOptimizer for NoobAI models
    optimizer = ModelOptimizer(checkpoint_name, disable_tag_injection=False)
    
    # Find appropriate LoRA if specified
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
    
    # Optimize the prompt with NoobAI prefix and background enhancements
    optimized_prompt = optimizer.inject_model_prefix(args.prompt)
    
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
        logger.info(f"Optimizing resolution to {width}*{height}")
    
    # Detect background type
    bg_type = optimizer.detect_background_type(args.prompt)
    
    # Detect subject type for negative prompt
    is_anthro = is_anthro_subject(args.prompt)
    is_feral = is_feral_subject(args.prompt)
    subject_type = "anthro" if is_anthro else ("feral" if is_feral else "unknown")
    
    # Store the prompt for negative prompt optimization
    optimizer._prompt_for_negative = args.prompt
    
    # Log what we're going to do
    logger.info(f"Generating image with NoobAI optimizations")
    logger.info(f"Model: {checkpoint_name}")
    logger.info(f"Architecture: {optimizer.architecture}")
    logger.info(f"Model family: {optimizer.family}")
    logger.info(f"Resolution: {width}*{height}")
    logger.info(f"Seed: {seed}")
    
    if bg_type:
        logger.info(f"Background detected: {bg_type}")
    
    logger.info(f"Subject type: {subject_type}")
    logger.info("Original prompt: " + args.prompt)
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
                    output_filename = f"noobai_{seed}"
                    
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
                            f"view_noobai_{seed}",
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
    
    # Open images with imv if requested and we have any
    if args.show and copied_images:
        open_images_with_imv(copied_images)
    
    # Return the optimized tags for reference
    return {
        "original_prompt": args.prompt,
        "optimized_prompt": optimized_prompt,
        "original_negative": args.negative_prompt,
        "optimized_negative": optimized_negative,
        "seed": seed,
        "background_type": bg_type or "none",
        "model": checkpoint_name,
    } 