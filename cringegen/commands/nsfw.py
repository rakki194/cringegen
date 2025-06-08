"""
NSFW generation commands for cringegen
"""

import logging
import random
import os
import time

from ..prompt_generation.generators.furry_generator import NsfwFurryPromptGenerator
from ..utils.comfy_api import (
    check_comfy_server,
    get_available_loras,
    get_matching_checkpoint,
    get_matching_lora,
    get_preferred_checkpoint,
    queue_prompt,
    check_generation_status,
)
from ..utils.file_utils import (
    copy_latest_images_from_comfyui,
    rsync_image_from_comfyui,
    rsync_latest_images_from_comfyui,
)
from ..utils.model_utils import ModelOptimizer
from ..workflows.furry import create_nsfw_furry_workflow

logger = logging.getLogger(__name__)


# Helper to add all shared NSFW arguments

def add_nsfw_shared_arguments(parser):
    parser.add_argument("--checkpoint", type=str, help="Checkpoint model to use")
    parser.add_argument("--lora", type=str, help="LoRA model to use")
    parser.add_argument(
        "--additional-loras",
        type=str,
        nargs="+",
        help="Additional LoRA models to stack (space separated)",
    )
    parser.add_argument(
        "--lora-weights",
        type=float,
        nargs="+",
        help="Weights for additional LoRAs (space separated, must match number of additional LoRAs)",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of prompts to generate")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for generation (ensures the same prompt is produced for the same seed value)",
    )
    parser.add_argument("--species", type=str, help="Species for the character")
    parser.add_argument("--gender", type=str, help="Gender for the character")
    parser.add_argument(
        "--duo",
        action="store_true",
        help="Generate a scene with two characters",
    )
    parser.add_argument(
        "--group",
        type=int,
        help="Generate a scene with specified number of characters (3-6)",
    )
    parser.add_argument(
        "--species2",
        type=str,
        help="Species for the second character (when using --duo)",
    )
    parser.add_argument(
        "--gender2",
        type=str,
        help="Gender for the second character (when using --duo)",
    )
    parser.add_argument(
        "--intensity",
        type=str,
        choices=["suggestive", "mild", "explicit", "hardcore"],
        default="explicit",
        help="Intensity of NSFW content",
    )
    parser.add_argument(
        "--kinks",
        type=str,
        help="Comma-separated list of kinks to include in the prompt",
    )
    parser.add_argument(
        "--no-kinks",
        action="store_true",
        help="Do not include any explicit or implicit kinks in the prompt",
    )
    parser.add_argument(
        "--colors",
        type=str,
        help="Comma-separated list of colors for the main character's fur/scales",
    )
    parser.add_argument(
        "--colors2",
        type=str,
        help="Comma-separated list of colors for the second character (when using --duo)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern type for the main character (e.g., spotted, striped, etc.)",
    )
    parser.add_argument(
        "--pattern2",
        type=str,
        help="Pattern type for the second character (when using --duo)",
    )
    parser.add_argument(
        "--anthro",
        action="store_true",
        help="Include 'anthro' tag in the prompt for the subject",
    )
    parser.add_argument(
        "--feral",
        action="store_true",
        help="Include 'feral' tag in the prompt for the subject",
    )
    parser.add_argument(
        "--prompt", type=str, help="Custom prompt to use (overrides generated prompt)"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        help="Custom negative prompt to use (overrides generated negative prompt)",
    )
    parser.add_argument("--width", type=int, default=1024, help="Width of the image")
    parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    parser.add_argument("--steps", type=int, default=40, help="Number of steps")
    parser.add_argument(
        "--cfg", type=float, default=None, help="CFG scale (defaults to model-specific value)"
    )
    parser.add_argument("--lora-strength", type=float, default=1.0, help="LoRA strength")
    parser.add_argument(
        "--sampler", type=str, help="Sampler to use (e.g., euler, euler_ancestral, ddim)"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler to use (use 'list-schedulers' command to see available options)",
    )
    parser.add_argument(
        "--no-generate", action="store_true", help="Only generate prompts, do not create images"
    )
    parser.add_argument(
        "--no-art-style", action="store_true", help="Don't include random art style in the prompt"
    )
    parser.add_argument(
        "--copy-output",
        action="store_true",
        help="Copy generated images to cringegen output directory",
    )
    parser.add_argument(
        "--pag", action="store_true", help="Use Perturbed-Attention Guidance for improved detail"
    )
    parser.add_argument(
        "--pag-scale", type=float, default=3.0, help="Scale for Perturbed-Attention Guidance"
    )
    parser.add_argument(
        "--pag-sigma-start", type=float, default=-1.0, help="Start sigma for PAG (default: auto)"
    )
    parser.add_argument(
        "--pag-sigma-end", type=float, default=-1.0, help="End sigma for PAG (default: auto)"
    )
    parser.add_argument(
        "--detail-daemon",
        action="store_true",
        help="Use DetailDaemonSamplerNode for enhanced details",
    )
    parser.add_argument(
        "--detail-amount",
        type=float,
        default=0.1,
        help="Detail amount for DetailDaemonSamplerNode (0.0-1.0)",
    )
    parser.add_argument(
        "--detail-start",
        type=float,
        default=0.5,
        help="Start percent for DetailDaemonSamplerNode (0.0-1.0)",
    )
    parser.add_argument(
        "--detail-end",
        type=float,
        default=0.8,
        help="End percent for DetailDaemonSamplerNode (0.0-1.0)",
    )
    parser.add_argument(
        "--split-sigmas", type=float, help="Value to split sigmas for multi-stage sampling"
    )
    parser.add_argument(
        "--split-first-cfg", type=float, help="CFG for first stage of split-sigma sampling"
    )
    parser.add_argument(
        "--split-second-cfg", type=float, help="CFG for second stage of split-sigma sampling"
    )
    parser.add_argument(
        "--split-first-sampler", type=str, help="Sampler for first stage of split-sigma sampling (e.g., euler, euler_ancestral)"
    )
    parser.add_argument(
        "--split-second-sampler", type=str, help="Sampler for second stage of split-sigma sampling (e.g., euler, dpm_2_ancestral)"
    )
    parser.add_argument(
        "--split-first-scheduler", type=str, help="Scheduler for first stage of split-sigma sampling (e.g., normal, karras)"
    )
    parser.add_argument(
        "--split-second-scheduler", type=str, help="Scheduler for second stage of split-sigma sampling (e.g., normal, karras)"
    )
    parser.add_argument(
        "--use-deepshrink",
        action="store_true",
        help="Use DeepShrink for improved high-frequency details",
    )
    parser.add_argument(
        "--deepshrink-factor", type=float, default=2.0, help="Downscale factor for DeepShrink"
    )
    parser.add_argument(
        "--deepshrink-start", type=float, default=0.0, help="Start percent for DeepShrink (0.0-1.0)"
    )
    parser.add_argument(
        "--deepshrink-end", type=float, default=0.35, help="End percent for DeepShrink (0.0-1.0)"
    )
    parser.add_argument(
        "--deepshrink-gradual",
        type=float,
        default=0.6,
        help="Gradual percent for DeepShrink (0.0-1.0)",
    )
    parser.add_argument(
        "--use-zsnr", action="store_true", help="Enable Zero SNR for potentially improved results"
    )
    parser.add_argument("--use-vpred", action="store_true", help="Use v-prediction sampling")
    parser.add_argument(
        "--comfy-output-dir",
        type=str,
        default="/home/kade/comfy/ComfyUI/output",
        help="ComfyUI output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--increment",
        action="store_true",
        help="Force seed increment by exactly +1 for each image when using --count",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use SSH to copy images from a remote ComfyUI instance",
    )
    parser.add_argument(
        "--ssh-host",
        type=str,
        default="otter_den",
        help="SSH hostname or IP address for remote ComfyUI instance",
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=1487,
        help="SSH port for remote ComfyUI instance",
    )
    parser.add_argument(
        "--ssh-user",
        type=str,
        default="kade",
        help="SSH username for remote ComfyUI instance",
    )
    parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key file for remote ComfyUI instance",
    )
    parser.add_argument(
        "--auto-optimize", 
        action="store_true",
        help="Automatically optimize parameters based on model architecture and family"
    )
    parser.add_argument(
        "--no-prefix-injection", 
        action="store_true",
        help="Disable automatic prefix injection for model-specific optimizations"
    )


def add_nsfw_command(subparsers, parent_parser):
    """Add the nsfw-furry command to the CLI"""
    nsfw_parser = subparsers.add_parser(
        "nsfw-furry", help="Generate a NSFW furry prompt", parents=[parent_parser]
    )
    add_nsfw_shared_arguments(nsfw_parser)
    nsfw_parser.set_defaults(func=generate_nsfw_furry)
    # Register the new character command
    add_nsfw_character_command(subparsers, parent_parser)
    return nsfw_parser


def generate_nsfw_furry(args):
    """Generate a NSFW furry prompt and optionally an image"""
    # To keep track of all copied images for --show flag
    copied_images = []

    # Set up seed if provided
    if args.seed == -1:
        seed = random.randint(1, 1000000)
    else:
        seed = args.seed
        random.seed(seed)

    # If increment flag is set and count > 1, log that we're using incremental mode
    if args.increment and args.count > 1:
        logger.info(
            f"Using seed increment mode: Starting with seed {seed} and incrementing by +1 for each image"
        )

    # Set up NSFW intensity
    intensity_levels = {"suggestive": 1, "mild": 2, "explicit": 3, "hardcore": 4}
    explicit_level = intensity_levels.get(args.intensity, 2)

    # Parse kinks
    kinks = []
    if args.kinks:
        kinks = [k.strip() for k in args.kinks.split(",")]

    # Check for fart-related kinks and recommend the fart_fetish LoRA
    fart_related_kinks = ["fart", "gas", "flatulence", "fart_fetish"]
    has_fart_kink = any(kink.lower() in fart_related_kinks for kink in kinks)

    if has_fart_kink and not args.lora:
        logger.info("Detected fart-related kink. Recommending fart_fetish LoRA for better results.")
        logger.info(
            "Example usage: --lora fart_fetish-v2s3000 --checkpoint noobaiXLVpredv10.safetensors"
        )

    # Parse colors
    colors = None
    if args.colors:
        colors = args.colors  # The FurryPromptGenerator will parse this

    # Parse colors2 for duo mode
    colors2 = None
    if args.duo and args.colors2:
        colors2 = args.colors2

    # Create generator instance with all parameters upfront
    generator = NsfwFurryPromptGenerator(
        species=args.species,
        gender=args.gender,
        explicit_level=explicit_level,
        use_anatomical_terms=not args.no_kinks,
        use_nlp=True,
        use_art_style=not args.no_art_style,
        is_anthro=True if args.anthro else (False if args.feral else True),
        is_feral=args.feral,
    )

    # Set additional properties that aren't in the constructor
    if kinks:
        generator.kinks = kinks

    # Set duo/group options
    if args.duo:
        generator.use_duo = True
        if args.species2:
            generator.species2 = args.species2
        if args.gender2:
            generator.gender2 = args.gender2
    elif args.group:
        generator.group_size = max(3, min(6, args.group))

    # Set colors and patterns
    if args.colors:
        generator.colors = [c.strip() for c in args.colors.split(",")]
    if args.pattern:
        generator.pattern = args.pattern

    if args.duo and args.colors2:
        generator.colors2 = [c.strip() for c in args.colors2.split(",")]
    if args.duo and args.pattern2:
        generator.pattern2 = args.pattern2

    # Log the species and gender being used
    logger.info(
        f"Generating NSFW prompts with species={generator.species}, gender={generator.gender}"
    )

    # Initialize model optimization if checkpoint is available and auto-optimize is enabled
    model_optimizer = None
    if args.checkpoint and (args.auto_optimize or not args.no_prefix_injection):
        model_name = os.path.basename(args.checkpoint)
        logger.info(f"Initializing model optimizer for: {model_name}")
        model_optimizer = ModelOptimizer(model_name, disable_tag_injection=args.no_prefix_injection)
        
        # Log model detection results
        logger.info(f"Model detected: {model_name}")
        logger.info(f"Architecture: {model_optimizer.architecture}")
        logger.info(f"Model family: {model_optimizer.family}")
        
        # Apply model-specific optimizations if auto-optimize is enabled
        if args.auto_optimize:
            # Check if resolution needs optimization
            if not model_optimizer.check_resolution(args.width, args.height):
                optimal_width, optimal_height = model_optimizer.get_optimal_resolution(args.width, args.height)
                logger.info(f"Optimizing resolution from {args.width}*{args.height} to {optimal_width}*{optimal_height}")
                args.width, args.height = optimal_width, optimal_height
            
            # Get optimal parameters
            params = model_optimizer.get_optimized_parameters()
            
            # Apply parameters if not explicitly set by user
            if not args.cfg and 'cfg' in params:
                args.cfg = params['cfg']
                logger.info(f"Using model-optimized CFG: {args.cfg}")
                
            if not args.steps and 'steps' in params:
                args.steps = params['steps']
                logger.info(f"Using model-optimized steps: {args.steps}")
                
            if not args.sampler and 'sampler' in params:
                args.sampler = params['sampler']
                logger.info(f"Using model-optimized sampler: {args.sampler}")
                
            if not args.scheduler and 'scheduler' in params:
                args.scheduler = params['scheduler']
                logger.info(f"Using model-optimized scheduler: {args.scheduler}")

    prompts = []
    for i in range(args.count):
        # Generate a unique seed for each prompt if multiple are requested
        if args.count > 1:
            # When increment flag is set, always add exactly 1 to the seed for each image
            if args.increment:
                curr_seed = seed + i
            else:
                # For non-increment mode, generate a completely new random seed for each image
                # but make it reproducible by deriving from the original seed
                random.seed(seed + i * 1000)  # Use a larger multiplier to get more variation
                curr_seed = random.randint(1, 1000000)
                random.seed(curr_seed)  # Set the random seed for prompt generation
        else:
            curr_seed = seed

        # Generate prompt
        if args.prompt:
            # Use custom prompt if provided
            prompt = args.prompt
            negative_prompt = args.negative_prompt or ""
        else:
            # Generate prompt with the generator
            prompt = generator.generate()
            negative_prompt = generator.get_negative_prompt()

        # Apply model-specific prompt optimizations if available
        if model_optimizer and not args.no_prefix_injection:
            original_prompt = prompt
            original_negative = negative_prompt
            
            prompt = model_optimizer.inject_model_prefix(prompt)
            negative_prompt = model_optimizer.inject_negative_prefix(negative_prompt)
            
            if prompt != original_prompt:
                logger.info(f"Model-optimized prompt: {prompt}")
            if negative_prompt != original_negative:
                logger.info(f"Model-optimized negative prompt: {negative_prompt}")
        
            # Log background detection
            bg_type = model_optimizer.detect_background_type(original_prompt)
            if bg_type:
                logger.info(f"Background detected: {bg_type}")
                logger.info(f"Applied background optimizations for {bg_type}")

        prompts.append((prompt, negative_prompt, curr_seed))

        # Print the generated prompt and seed
        logger.info(f"Generated NSFW prompt (seed {curr_seed}):")
        logger.info(f"PROMPT: {prompt}")
        logger.info(f"NEGATIVE PROMPT: {negative_prompt}")
        logger.info("---")

    # Generate images if requested
    if not args.no_generate:
        # Check if ComfyUI server is available
        is_available, message = check_comfy_server(args.comfy_url)
        if not is_available:
            logger.error(message)
            logger.error("Image generation skipped. Start ComfyUI and try again.")
            return prompts

        # Check for checkpoint and LoRA
        if not args.checkpoint:
            # Use preferred checkpoint instead of first available
            args.checkpoint = get_preferred_checkpoint(args.comfy_url)
        else:
            args.checkpoint = get_matching_checkpoint(args.checkpoint, args.comfy_url)

        if not args.lora and not args.checkpoint:
            available_loras = get_available_loras(args.comfy_url)
            if available_loras:
                args.lora = available_loras[0]
        elif args.lora:
            args.lora = get_matching_lora(args.lora, args.comfy_url)

        if not args.checkpoint and not args.lora:
            logger.error("No checkpoint or LoRA specified and none found automatically.")
            return

        # Process additional LoRAs
        additional_loras = []
        if args.additional_loras:
            for lora_name in args.additional_loras:
                lora_path = get_matching_lora(lora_name, args.comfy_url)
                if lora_path:
                    additional_loras.append(lora_path)

        # Process LoRA weights
        lora_weights = (
            args.lora_weights if args.lora_weights else [args.lora_strength] * len(additional_loras)
        )

        # Make sure we have the right number of weights
        if len(lora_weights) != len(additional_loras):
            logger.warning(
                f"Number of LoRA weights ({len(lora_weights)}) doesn't match number of additional LoRAs ({len(additional_loras)}). Using default weight."
            )
            lora_weights = [args.lora_strength] * len(additional_loras)

        for i, (prompt, negative_prompt, curr_seed) in enumerate(prompts):
            # Create workflow for this prompt
            workflow = create_nsfw_furry_workflow(
                checkpoint=args.checkpoint,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=args.width,
                height=args.height,
                seed=curr_seed,
                steps=args.steps,
                cfg=args.cfg,
                lora_strength=args.lora_strength,
                lora=args.lora,
                loras=additional_loras,
                lora_weights=lora_weights,
                sampler=args.sampler,
                scheduler=args.scheduler,
                use_pag=args.pag,
                pag_scale=args.pag_scale,
                pag_sigma_start=args.pag_sigma_start,
                pag_sigma_end=args.pag_sigma_end,
                use_detail_daemon=args.detail_daemon,
                detail_amount=args.detail_amount,
                detail_start=args.detail_start,
                detail_end=args.detail_end,
                split_sigmas=args.split_sigmas,
                split_first_cfg=args.split_first_cfg,
                split_second_cfg=args.split_second_cfg,
                split_first_sampler=args.split_first_sampler,
                split_second_sampler=args.split_second_sampler,
                split_first_scheduler=args.split_first_scheduler,
                split_second_scheduler=args.split_second_scheduler,
                use_deepshrink=args.use_deepshrink,
                deepshrink_factor=args.deepshrink_factor,
                deepshrink_start=args.deepshrink_start,
                deepshrink_end=args.deepshrink_end,
                deepshrink_gradual=args.deepshrink_gradual,
                use_zsnr=args.use_zsnr,
                use_vpred=args.use_vpred,
            )

            # Queue workflow
            prompt_id = queue_prompt(workflow, args.comfy_url)
            logger.info(f"Queued prompt {i+1}/{len(prompts)} with ID: {prompt_id}")

            # Extract the actual prompt_id string from the response
            if isinstance(prompt_id, dict) and "prompt_id" in prompt_id:
                prompt_id_str = prompt_id["prompt_id"]
                logger.info(f"Using prompt ID string: {prompt_id_str}")
            else:
                prompt_id_str = str(prompt_id)
                logger.warning(f"Unexpected prompt_id format, using as string: {prompt_id_str}")

            # Poll for job completion before proceeding
            start_time = time.time()
            max_wait_time = 300  # 5 minutes
            status = None
            while True:
                status = check_generation_status(prompt_id_str, args.comfy_url)
                print(f"DEBUG: ComfyUI status response (poll): {status} (type: {type(status)})")
                logger.debug(f"ComfyUI status response (poll): {status} (type: {type(status)})")
                if status.get("status") == "completed":
                    logger.info(f"ComfyUI job {prompt_id_str} completed.")
                    break
                if time.time() - start_time > max_wait_time:
                    logger.error(f"Timeout waiting for ComfyUI job {prompt_id_str} to complete.")
                    break
                time.sleep(2)  # Poll every 2 seconds
            # Now handle the status as before
            try:
                print(f"DEBUG: ComfyUI status response (final): {status} (type: {type(status)})")
                logger.debug(f"ComfyUI status response (final): {status} (type: {type(status)})")
                if status and status.get("status") == "completed":
                    if status.get("images"):
                        image_filename = status["images"][0]["filename"]
                        logger.info(f"Found image: {image_filename}")
                        logger.debug(f"Attempting to copy image: {image_filename} from remote output dir: {args.comfy_output_dir} to local dir: {args.output_dir}")
                        time.sleep(2)
                        try:
                            result_path = rsync_image_from_comfyui(
                                image_filename,
                                args.ssh_host,
                                args.comfy_output_dir,
                                args.output_dir,
                                f"nsfw_furry_{curr_seed}",
                                ssh_port=args.ssh_port,
                                ssh_user=args.ssh_user,
                                ssh_key=args.ssh_key,
                            )
                            print(f"DEBUG: rsync_image_from_comfyui returned: {result_path}")
                            logger.info(f"rsync_image_from_comfyui returned: {result_path}")
                            # Forcibly append/extend copied_images for debugging
                            if isinstance(result_path, str) and result_path:
                                copied_images.append(result_path)
                            elif isinstance(result_path, list) and result_path:
                                copied_images.extend(result_path)
                            print(f"DEBUG: copied_images after forced append: {copied_images}")
                            logger.info(f"copied_images after forced append: {copied_images}")
                        except Exception as e:
                            print(f"EXCEPTION in rsync_image_from_comfyui: {e}")
                            logger.error(f"Error during rsync_image_from_comfyui: {e}")
                            success = False
                    else:
                        print(f"Job completed but no images found in status. Status: {status}")
                        logger.warning(f"Job completed but no images found in status. Status: {status}")
                else:
                    print(f"Job not completed or unexpected status. Status: {status}")
                    logger.warning(f"Job not completed or unexpected status. Status: {status}")
            except Exception as e:
                print(f"EXCEPTION in status/image handling: {e}")
                logger.error(f"EXCEPTION in status/image handling: {e}")
                # Try copying the latest image as a fallback
                logger.info("Trying to copy the most recent image as fallback...")
                try:
                    if args.remote:
                        if not args.ssh_host:
                            logger.error("SSH host is required when using --remote")
                            return prompts

                        logger.info(
                            f"Using rsync over SSH to copy latest image from {args.ssh_host}"
                        )
                        copied = rsync_latest_images_from_comfyui(
                            args.ssh_host,
                            args.comfy_output_dir,
                            args.output_dir,
                            limit=1,
                            ssh_port=args.ssh_port,
                            ssh_user=args.ssh_user,
                            ssh_key=args.ssh_key,
                        )
                    else:
                        # Local copy
                        copied = copy_latest_images_from_comfyui(
                            args.comfy_output_dir, args.output_dir, limit=1
                        )

                    if copied:
                        logger.info(f"Copied most recent image to {args.output_dir}")
                        # Add to the list of copied images
                        copied_images.extend(copied)

                        # After rsync, log the copied_images list
                        print(f"DEBUG: copied_images after rsync: {copied_images}")
                        logger.info(f"copied_images after rsync: {copied_images}")
                except Exception as fallback_e:
                    logger.error(f"Fallback copy also failed: {fallback_e}")
                    logger.warning("Could not copy any images")

    # Open images with imv if requested and we have any
    if args.show and copied_images:
        print(f"DEBUG: Entering image-showing block with copied_images: {copied_images}")
        logger.info(f"Entering image-showing block with copied_images: {copied_images}")
        for local_image_path in copied_images:
            print(f"DEBUG: About to show image with imv: {local_image_path}")
            logger.info(f"About to show image with imv: {local_image_path}")
            try:
                import subprocess
                result = subprocess.run(["imv", local_image_path], check=False)
                logger.info(f"imv subprocess returned: {result.returncode}")
            except Exception as e:
                print(f"EXCEPTION in imv subprocess: {e}")
                logger.error(f"EXCEPTION in imv subprocess: {e}")

    return prompts


def create_nsfw_workflow(args):
    """Create a base NSFW workflow template for XY plotting

    Args:
        args: Command line arguments

    Returns:
        A workflow dictionary
    """
    # Use the more advanced furry workflow implementation for NSFW generation
    from ..workflows.furry import create_nsfw_furry_workflow
    import logging
    logger = logging.getLogger(__name__)
    
    import random

    # Collect all the parameters we need to pass to create_nsfw_furry_workflow
    checkpoint = (
        args.checkpoint
        if hasattr(args, "checkpoint") and args.checkpoint
        else "noobaiXLVpredv10.safetensors"
    )
    
    # Get prompt
    if hasattr(args, "prompt") and args.prompt:
        prompt = args.prompt
    else:
        from .utils import generate_random_prompt  # Import here to avoid circular imports
        prompt = generate_random_prompt(nsfw=True)

    # Get negative prompt
    if hasattr(args, "negative_prompt") and args.negative_prompt:
        negative_prompt = args.negative_prompt
    else:
        negative_prompt = "worst quality, low quality, medium quality, deleted, lowres, bad anatomy, bad hands, watermark"
    
    # Get basic parameters
    width = args.width if hasattr(args, "width") else 1024
    height = args.height if hasattr(args, "height") else 1024
    seed = args.seed if hasattr(args, "seed") and args.seed != -1 else random.randint(0, 2**32 - 1)
    steps = args.steps if hasattr(args, "steps") else 40
    cfg = args.cfg if hasattr(args, "cfg") else 3.5
    sampler = args.sampler if hasattr(args, "sampler") else "euler_ancestral"
    scheduler = args.scheduler if hasattr(args, "scheduler") else "normal"
    
    # Get LoRA parameters
    lora = args.lora if hasattr(args, "lora") else None
    lora_strength = args.lora_strength if hasattr(args, "lora_strength") else 1.0
    loras = args.loras if hasattr(args, "loras") else None
    lora_weights = args.lora_weights if hasattr(args, "lora_weights") else None
    
    # Get PAG parameters - default to True if it's being used in XY plot
    use_pag = True if (hasattr(args, "x_param") and args.x_param in ["pag_scale", "pag_sigma_start", "pag_sigma_end"]) or \
              (hasattr(args, "y_param") and args.y_param in ["pag_scale", "pag_sigma_start", "pag_sigma_end"]) else \
              (args.pag if hasattr(args, "pag") else False)
              
    pag_scale = args.pag_scale if hasattr(args, "pag_scale") else 3.0
    pag_sigma_start = args.pag_sigma_start if hasattr(args, "pag_sigma_start") else -1.0
    pag_sigma_end = args.pag_sigma_end if hasattr(args, "pag_sigma_end") else -1.0
    
    # Log PAG settings before creating workflow for debugging
    if use_pag:
        logger.debug(f"Using PAG with scale: {pag_scale}, sigma_start: {pag_sigma_start}, sigma_end: {pag_sigma_end}")
    else:
        logger.debug("PAG is disabled")
    
    # Get DeepShrink parameters
    use_deepshrink = args.use_deepshrink if hasattr(args, "use_deepshrink") else False
    deepshrink_factor = args.deepshrink_factor if hasattr(args, "deepshrink_factor") else 2.0
    deepshrink_start = args.deepshrink_start if hasattr(args, "deepshrink_start") else 0.0
    deepshrink_end = args.deepshrink_end if hasattr(args, "deepshrink_end") else 0.35
    deepshrink_gradual = args.deepshrink_gradual if hasattr(args, "deepshrink_gradual") else 0.6
    
    # Get Split Sigmas parameters
    split_sigmas = args.split_sigmas if hasattr(args, "split_sigmas") else None
    split_first_cfg = args.split_first_cfg if hasattr(args, "split_first_cfg") else None
    split_second_cfg = args.split_second_cfg if hasattr(args, "split_second_cfg") else None
    split_first_sampler = args.split_first_sampler if hasattr(args, "split_first_sampler") else None
    split_second_sampler = args.split_second_sampler if hasattr(args, "split_second_sampler") else None
    split_first_scheduler = args.split_first_scheduler if hasattr(args, "split_first_scheduler") else None
    split_second_scheduler = args.split_second_scheduler if hasattr(args, "split_second_scheduler") else None
    
    # Get Detail Daemon parameters
    use_detail_daemon = args.detail_daemon if hasattr(args, "detail_daemon") else False
    detail_amount = args.detail_amount if hasattr(args, "detail_amount") else 0.1
    detail_start = args.detail_start if hasattr(args, "detail_start") else 0.5
    detail_end = args.detail_end if hasattr(args, "detail_end") else 0.8
    detail_exponent = args.detail_exponent if hasattr(args, "detail_exponent") else 1.5
    
    # Get ZSNR and v-prediction parameters
    use_zsnr = args.use_zsnr if hasattr(args, "use_zsnr") else False
    use_vpred = args.use_vpred if hasattr(args, "use_vpred") else False
    
    # Call the more advanced implementation
    workflow = create_nsfw_furry_workflow(
        checkpoint=checkpoint,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        lora_strength=lora_strength,
        lora=lora,
        loras=loras,
        lora_weights=lora_weights,
        sampler=sampler,
        scheduler=scheduler,
        use_deepshrink=use_deepshrink,
        use_pag=use_pag,
        use_zsnr=use_zsnr,
        use_vpred=use_vpred,
        split_sigmas=split_sigmas,
        split_first_cfg=split_first_cfg,
        split_second_cfg=split_second_cfg,
        split_first_sampler=split_first_sampler,
        split_second_sampler=split_second_sampler,
        split_first_scheduler=split_first_scheduler,
        split_second_scheduler=split_second_scheduler,
        pag_scale=pag_scale,
        pag_sigma_start=pag_sigma_start,
        pag_sigma_end=pag_sigma_end,
        deepshrink_factor=deepshrink_factor,
        deepshrink_start=deepshrink_start,
        deepshrink_end=deepshrink_end,
        deepshrink_gradual=deepshrink_gradual,
        use_detail_daemon=use_detail_daemon,
        detail_amount=detail_amount,
        detail_start=detail_start,
        detail_end=detail_end,
        batch_size=1,
        show=False,
    )
    
    # Post-process the workflow to ensure the PAG node is connected correctly if it exists
    if use_pag:
        pag_node_id = None
        checkpoint_node_id = None
        
        # First, find the PAG and checkpoint nodes
        for node_id, node in workflow.items():
            if node["class_type"] == "PerturbedAttention":
                pag_node_id = node_id
                logger.debug(f"Found PAG node {node_id} in workflow: {node}")
            elif node["class_type"] == "CheckpointLoaderSimple":
                checkpoint_node_id = node_id
                logger.debug(f"Found checkpoint node {node_id} in workflow")
                
        # If PAG node exists, ensure its output is properly used
        if pag_node_id and checkpoint_node_id:
            pag_output = [pag_node_id, 0]
            checkpoint_output = [checkpoint_node_id, 0]
            
            # Check if any nodes use the checkpoint model output directly
            # if they should be using the PAG output
            for node_id, node in workflow.items():
                if node_id == pag_node_id:  # Skip the PAG node itself
                    continue
                    
                # Check model inputs in nodes like KSampler, CFGGuider, etc.
                for input_name, input_value in node["inputs"].items():
                    if input_name == "model" and isinstance(input_value, list) and len(input_value) == 2:
                        if input_value[0] == checkpoint_node_id and input_value[1] == 0:
                            # This node should use PAG output instead
                            node["inputs"][input_name] = pag_output
                            logger.debug(f"Updated node {node_id} to use PAG output for input '{input_name}'")
        elif use_pag and not pag_node_id:
            logger.warning("PAG was enabled but no PAG node was found in the workflow!")
    
    return workflow


def add_nsfw_character_command(subparsers, parent_parser):
    parser = subparsers.add_parser(
        "nsfw-character",
        help="Generate a thematically appropriate NSFW image for a character using NoobAI",
        parents=[parent_parser],
    )
    parser.add_argument("character", type=str, help="Character name (e.g., blaidd, krystal, legoshi)")
    add_nsfw_shared_arguments(parser)
    parser.set_defaults(func=generate_nsfw_character)


def generate_nsfw_character(args):
    import importlib
    # Try to import the character module
    try:
        char_mod = importlib.import_module(f"cringegen.data.characters.{args.character}")
    except ModuleNotFoundError:
        print(f"Character module for '{args.character}' not found.")
        return
    # Get the CHARACTER_TEMPLATE
    template = getattr(char_mod, "CHARACTER_TEMPLATE", None)
    if template is None:
        print(f"CHARACTER_TEMPLATE not found in module for '{args.character}'.")
        return
    # Set up args for NSFW generation
    args.species = template.species
    args.gender = template.gender.name.lower() if hasattr(template.gender, 'name') else str(template.gender)
    args.checkpoint = "noobaiXLVpredv10.safetensors"
    args.lora = "noob/fd-v3s6000.safetensors"
    args.lora_strength = 0.36
    args.lora_te_strength = 0.36
    # Compose a character-specific prompt prefix
    character_tags = []
    if hasattr(template, "model_tags") and "e621" in template.model_tags:
        character_tags.append(template.model_tags["e621"])
    if hasattr(template, "features"):
        character_tags.extend(template.features)
    if hasattr(template, "appearance_traits"):
        character_tags.extend(template.appearance_traits)
    # Only add clothing if not hardcore
    if getattr(args, 'intensity', None) != 'hardcore':
        if hasattr(template, "clothing"):
            character_tags.extend(template.clothing)
    if hasattr(template, "accessories"):
        character_tags.extend(template.accessories)
    if hasattr(template, "personality"):
        character_tags.extend(template.personality)
    # Remove duplicates and join
    character_tags = list(dict.fromkeys(character_tags))

    # Always include 'solo' in the prompt for single-character NSFW
    if 'solo' not in character_tags:
        character_tags.insert(0, 'solo')
    # Always include 'nsfw' in the prompt
    if 'nsfw' not in character_tags:
        character_tags.insert(0, 'nsfw')

    # Always include character nsfw_traits for explicit/hardcore
    if getattr(args, 'intensity', None) in ['explicit', 'hardcore']:
        if hasattr(template, "nsfw_traits"):
            for tag in template.nsfw_traits:
                if tag not in character_tags:
                    character_tags.append(tag)

    # For 'hardcore', add species- and gender-appropriate tags
    if getattr(args, 'intensity', None) == 'hardcore':
        # Prefer character-specific hardcore_tags if present
        if hasattr(template, "hardcore_tags") and template.hardcore_tags:
            for tag in template.hardcore_tags:
                if tag not in character_tags:
                    character_tags.append(tag)
        else:
            species = template.species.lower() if hasattr(template, "species") else ""
            gender = template.gender.name.lower() if hasattr(template.gender, "name") else str(template.gender).lower()

            species_hardcore_tags = {
                "wolf": ["canine genitalia"],
                "dog": ["canine genitalia"],
                "fox": ["canine genitalia"],
                "lion": ["feline genitalia", "barbed penis"],
                "tiger": ["feline genitalia", "barbed penis"],
                "cat": ["feline genitalia", "barbed penis"],
                "horse": ["equine genitalia"],
                "zebra": ["equine genitalia"],
                # Add more as needed
            }
            female_tags = ["animal pussy", "pussy", "vulva", "canine pussy", "feline pussy", "equine pussy"]
            # Add species-specific tags
            for key, tags in species_hardcore_tags.items():
                if key in species:
                    for tag in tags:
                        if tag not in character_tags:
                            character_tags.append(tag)
            # Add female tags if character is female
            if gender in ["female", "f"]:
                for tag in female_tags:
                    if tag not in character_tags:
                        character_tags.append(tag)
            # Add general explicit tags
            for tag in ["cum", "cum drip", "ejaculation", "cumshot", "cum on body", "cum inside", "penetration"]:
                if tag not in character_tags:
                    character_tags.append(tag)
    character_prefix = ", ".join(character_tags)

    # --- Add detailed, thematically accurate background prompt ---
    # Import here to avoid circular imports
    from cringegen.prompt_generation.nlp.background_utils import generate_background_description, get_complementary_locations
    character_name = getattr(template, 'model_tags', {}).get('e621', '').lower()
    # Character-specific backgrounds
    if 'legoshi' in character_name:
        location = 'school'
        background_desc = generate_background_description(location)
    elif 'krystal' in character_name:
        location = 'alien world'
        background_desc = generate_background_description(location)
    elif 'robin hood' in character_name:
        location = 'sherwood forest'
        background_desc = generate_background_description('forest')
    elif 'nick wilde' in character_name:
        location = 'city'
        background_desc = generate_background_description(location)
    elif 'sonic' in character_name:
        location = 'green hill zone'
        background_desc = generate_background_description('meadow')
    elif 'blaidd' in character_name:
        location = 'dark fantasy landscape'
        background_desc = generate_background_description('forest')
    else:
        # Get a thematically appropriate location for the character's species
        locations = get_complementary_locations(template.species or "forest")
        location = locations[0] if locations else "forest"
        background_desc = generate_background_description(location)
    # Append the background description to the prompt
    if hasattr(args, "prompt") and args.prompt:
        args.prompt = character_prefix + ", " + args.prompt + ", " + background_desc
    else:
        args.prompt = character_prefix + ", " + background_desc

    # Always add 'safe' to the negative prompt
    if hasattr(args, "negative_prompt") and args.negative_prompt:
        if 'safe' not in args.negative_prompt:
            args.negative_prompt = args.negative_prompt + ', safe'
    else:
        args.negative_prompt = 'safe'

    # Call the existing NSFW generation function
    return generate_nsfw_furry(args)
