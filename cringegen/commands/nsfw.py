"""
NSFW generation commands for CringeGen
"""

import logging
import random

from ..prompt_generation.generators.furry_generator import NsfwFurryPromptGenerator
from ..utils.comfy_api import (
    check_comfy_server,
    get_available_checkpoints,
    get_available_loras,
    get_image_path,
    get_matching_checkpoint,
    get_matching_lora,
    get_preferred_checkpoint,
    queue_prompt,
)
from ..utils.file_utils import (
    copy_image_from_comfyui, 
    copy_latest_images_from_comfyui,
    rsync_image_from_comfyui,
    rsync_latest_images_from_comfyui
)
from ..workflows.furry import create_nsfw_furry_workflow

logger = logging.getLogger(__name__)


def add_nsfw_command(subparsers, parent_parser):
    """Add the nsfw-furry command to the CLI"""
    nsfw_parser = subparsers.add_parser(
        "nsfw-furry", help="Generate a NSFW furry prompt", parents=[parent_parser]
    )
    nsfw_parser.add_argument("--checkpoint", type=str, help="Checkpoint model to use")
    nsfw_parser.add_argument("--lora", type=str, help="LoRA model to use")
    nsfw_parser.add_argument(
        "--additional-loras",
        type=str,
        nargs="+",
        help="Additional LoRA models to stack (space separated)",
    )
    nsfw_parser.add_argument(
        "--lora-weights",
        type=float,
        nargs="+",
        help="Weights for additional LoRAs (space separated, must match number of additional LoRAs)",
    )
    nsfw_parser.add_argument("--count", type=int, default=1, help="Number of prompts to generate")
    nsfw_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for generation (ensures the same prompt is produced for the same seed value)",
    )
    nsfw_parser.add_argument("--species", type=str, help="Species for the character")
    nsfw_parser.add_argument("--gender", type=str, help="Gender for the character")

    # Multi-subject options
    nsfw_parser.add_argument(
        "--duo",
        action="store_true",
        help="Generate a scene with two characters",
    )
    nsfw_parser.add_argument(
        "--group",
        type=int,
        help="Generate a scene with specified number of characters (3-6)",
    )
    nsfw_parser.add_argument(
        "--species2",
        type=str,
        help="Species for the second character (when using --duo)",
    )
    nsfw_parser.add_argument(
        "--gender2",
        type=str,
        help="Gender for the second character (when using --duo)",
    )

    # NSFW-specific options
    nsfw_parser.add_argument(
        "--intensity",
        type=str,
        choices=["suggestive", "mild", "explicit", "hardcore"],
        default="explicit",
        help="Intensity of NSFW content",
    )
    nsfw_parser.add_argument(
        "--kinks",
        type=str,
        help="Comma-separated list of kinks to include in the prompt",
    )
    nsfw_parser.add_argument(
        "--no-kinks",
        action="store_true",
        help="Do not include any explicit or implicit kinks in the prompt",
    )

    # Color options
    nsfw_parser.add_argument(
        "--colors",
        type=str,
        help="Comma-separated list of colors for the main character's fur/scales",
    )
    nsfw_parser.add_argument(
        "--colors2",
        type=str,
        help="Comma-separated list of colors for the second character (when using --duo)",
    )
    nsfw_parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern type for the main character (e.g., spotted, striped, etc.)",
    )
    nsfw_parser.add_argument(
        "--pattern2",
        type=str,
        help="Pattern type for the second character (when using --duo)",
    )

    nsfw_parser.add_argument(
        "--anthro",
        action="store_true",
        help="Include 'anthro' tag in the prompt for the subject",
    )
    nsfw_parser.add_argument(
        "--feral",
        action="store_true",
        help="Include 'feral' tag in the prompt for the subject",
    )
    nsfw_parser.add_argument(
        "--prompt", type=str, help="Custom prompt to use (overrides generated prompt)"
    )
    nsfw_parser.add_argument(
        "--negative-prompt",
        type=str,
        help="Custom negative prompt to use (overrides generated negative prompt)",
    )
    nsfw_parser.add_argument("--width", type=int, default=1024, help="Width of the image")
    nsfw_parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    nsfw_parser.add_argument("--steps", type=int, default=40, help="Number of steps")
    nsfw_parser.add_argument(
        "--cfg", type=float, default=None, help="CFG scale (defaults to model-specific value)"
    )
    nsfw_parser.add_argument("--lora-strength", type=float, default=1.0, help="LoRA strength")
    nsfw_parser.add_argument(
        "--sampler", type=str, help="Sampler to use (e.g., euler, euler_ancestral, ddim)"
    )
    nsfw_parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler to use (use 'list-schedulers' command to see available options)",
    )
    nsfw_parser.add_argument(
        "--no-generate", action="store_true", help="Only generate prompts, do not create images"
    )
    nsfw_parser.add_argument(
        "--no-art-style", action="store_true", help="Don't include random art style in the prompt"
    )
    nsfw_parser.add_argument(
        "--copy-output",
        action="store_true",
        help="Copy generated images to cringegen output directory",
    )
    
    # Advanced workflow options
    nsfw_parser.add_argument(
        "--pag", 
        action="store_true", 
        help="Use Perturbed-Attention Guidance for improved detail"
    )
    nsfw_parser.add_argument(
        "--pag-scale", 
        type=float, 
        default=3.0, 
        help="Scale for Perturbed-Attention Guidance"
    )
    nsfw_parser.add_argument(
        "--pag-sigma-start", 
        type=float, 
        default=-1.0, 
        help="Start sigma for PAG (default: auto)"
    )
    nsfw_parser.add_argument(
        "--pag-sigma-end", 
        type=float, 
        default=-1.0, 
        help="End sigma for PAG (default: auto)"
    )
    nsfw_parser.add_argument(
        "--detail-daemon", 
        action="store_true", 
        help="Use DetailDaemonSamplerNode for enhanced details"
    )
    nsfw_parser.add_argument(
        "--detail-amount", 
        type=float, 
        default=0.1, 
        help="Detail amount for DetailDaemonSamplerNode (0.0-1.0)"
    )
    nsfw_parser.add_argument(
        "--detail-start", 
        type=float, 
        default=0.5, 
        help="Start percent for DetailDaemonSamplerNode (0.0-1.0)"
    )
    nsfw_parser.add_argument(
        "--detail-end", 
        type=float, 
        default=0.8, 
        help="End percent for DetailDaemonSamplerNode (0.0-1.0)"
    )
    nsfw_parser.add_argument(
        "--split-sigmas", 
        type=float, 
        help="Value to split sigmas for multi-stage sampling"
    )
    nsfw_parser.add_argument(
        "--split-first-cfg", 
        type=float, 
        help="CFG for first stage of split-sigma sampling"
    )
    nsfw_parser.add_argument(
        "--split-second-cfg", 
        type=float, 
        help="CFG for second stage of split-sigma sampling"
    )
    nsfw_parser.add_argument(
        "--use-deepshrink", 
        action="store_true", 
        help="Use DeepShrink for improved high-frequency details"
    )
    nsfw_parser.add_argument(
        "--deepshrink-factor", 
        type=float, 
        default=2.0, 
        help="Downscale factor for DeepShrink"
    )
    nsfw_parser.add_argument(
        "--deepshrink-start", 
        type=float, 
        default=0.0, 
        help="Start percent for DeepShrink (0.0-1.0)"
    )
    nsfw_parser.add_argument(
        "--deepshrink-end", 
        type=float, 
        default=0.35, 
        help="End percent for DeepShrink (0.0-1.0)"
    )
    nsfw_parser.add_argument(
        "--deepshrink-gradual", 
        type=float, 
        default=0.6, 
        help="Gradual percent for DeepShrink (0.0-1.0)"
    )
    nsfw_parser.add_argument(
        "--use-zsnr", 
        action="store_true", 
        help="Enable Zero SNR for potentially improved results"
    )
    nsfw_parser.add_argument(
        "--use-vpred", 
        action="store_true", 
        help="Use v-prediction sampling"
    )
    
    nsfw_parser.add_argument(
        "--comfy-output-dir",
        type=str,
        default="/home/kade/toolkit/diffusion/comfy/ComfyUI/output",
        help="ComfyUI output directory",
    )
    nsfw_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated images",
    )
    nsfw_parser.add_argument(
        "--increment",
        action="store_true",
        help="Force seed increment by exactly +1 for each image when using --count",
    )

    # Remote ComfyUI options
    nsfw_parser.add_argument(
        "--remote",
        action="store_true",
        help="Use SSH to copy images from a remote ComfyUI instance",
    )
    nsfw_parser.add_argument(
        "--ssh-host",
        type=str,
        help="SSH hostname or IP address for remote ComfyUI instance",
    )
    nsfw_parser.add_argument(
        "--ssh-port",
        type=int,
        default=22,
        help="SSH port for remote ComfyUI instance",
    )
    nsfw_parser.add_argument(
        "--ssh-user",
        type=str,
        help="SSH username for remote ComfyUI instance",
    )
    nsfw_parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key file for remote ComfyUI instance",
    )

    nsfw_parser.set_defaults(func=generate_nsfw_furry)
    return nsfw_parser


def generate_nsfw_furry(args):
    """Generate a NSFW furry prompt and optionally an image"""
    # Set up seed if provided
    if args.seed == -1:
        seed = random.randint(1, 1000000)
    else:
        seed = args.seed
        random.seed(seed)
    
    # If increment flag is set and count > 1, log that we're using incremental mode
    if args.increment and args.count > 1:
        logger.info(f"Using seed increment mode: Starting with seed {seed} and incrementing by +1 for each image")

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
        logger.info("Example usage: --lora fart_fetish-v2s3000 --checkpoint noobaiXLVpredv10.safetensors")
    
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
        is_feral=args.feral
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
    logger.info(f"Generating NSFW prompts with species={generator.species}, gender={generator.gender}")

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
                prompt=prompt,
                negative_prompt=negative_prompt,
                checkpoint=args.checkpoint,
                lora=args.lora,
                loras=additional_loras,
                lora_weights=lora_weights,
                seed=curr_seed,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg=args.cfg,
                lora_strength=args.lora_strength,
                sampler=args.sampler,
                scheduler=args.scheduler,
                # Advanced workflow options
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
            if isinstance(prompt_id, dict) and 'prompt_id' in prompt_id:
                prompt_id_str = prompt_id['prompt_id']
                logger.info(f"Using prompt ID string: {prompt_id_str}")
            else:
                prompt_id_str = str(prompt_id)
                logger.warning(f"Unexpected prompt_id format, using as string: {prompt_id_str}")

            # Get image path if we need to copy it
            if args.copy_output and not args.no_generate:
                try:
                    # First try using the API to get the image path
                    image_path = get_image_path(prompt_id_str, args.comfy_url)
                    
                    if image_path and len(image_path) > 0:
                        logger.info(f"Found image path via API: {image_path[0]}")
                        success = False
                        
                        if args.remote:
                            # Check for required SSH parameters
                            if not args.ssh_host:
                                logger.error("SSH host is required when using --remote")
                                return prompts
                            
                            logger.info(f"Using rsync over SSH to copy image from {args.ssh_host}")
                            success = rsync_image_from_comfyui(
                                image_path[0],  # Use the first image
                                args.ssh_host,
                                args.comfy_output_dir,
                                args.output_dir,
                                f"nsfw_furry_{curr_seed}",
                                ssh_port=args.ssh_port,
                                ssh_user=args.ssh_user,
                                ssh_key=args.ssh_key
                            ) is not None
                        else:
                            # Local copy
                            success = copy_image_from_comfyui(
                                image_path[0],  # Use the first image
                                args.comfy_output_dir,
                                args.output_dir,
                                f"nsfw_furry_{curr_seed}",
                            ) is not None
                            
                        if success:
                            logger.info(f"Copied image to {args.output_dir}")
                        else:
                            # If copying via API path failed, try looking for the most recent image
                            logger.info("Trying to copy the most recent image instead...")
                            
                            if args.remote:
                                if not args.ssh_host:
                                    logger.error("SSH host is required when using --remote")
                                    return prompts
                                    
                                logger.info(f"Using rsync over SSH to copy latest image from {args.ssh_host}")
                                copied = rsync_latest_images_from_comfyui(
                                    args.ssh_host,
                                    args.comfy_output_dir,
                                    args.output_dir,
                                    limit=1,
                                    ssh_port=args.ssh_port,
                                    ssh_user=args.ssh_user,
                                    ssh_key=args.ssh_key
                                )
                            else:
                                # Local copy
                                copied = copy_latest_images_from_comfyui(
                                    args.comfy_output_dir, 
                                    args.output_dir, 
                                    limit=1
                                )
                                
                            if copied:
                                logger.info(f"Copied most recent image to {args.output_dir}")
                            else:
                                logger.warning("Failed to find or copy any images")
                    else:
                        # If API doesn't return image path, try looking for the most recent image
                        logger.info("No image path from API, trying to copy the most recent image...")
                        
                        if args.remote:
                            if not args.ssh_host:
                                logger.error("SSH host is required when using --remote")
                                return prompts
                                
                            logger.info(f"Using rsync over SSH to copy latest image from {args.ssh_host}")
                            copied = rsync_latest_images_from_comfyui(
                                args.ssh_host,
                                args.comfy_output_dir,
                                args.output_dir,
                                limit=1,
                                ssh_port=args.ssh_port,
                                ssh_user=args.ssh_user,
                                ssh_key=args.ssh_key
                            )
                        else:
                            # Local copy
                            copied = copy_latest_images_from_comfyui(
                                args.comfy_output_dir, 
                                args.output_dir, 
                                limit=1
                            )
                            
                        if copied:
                            logger.info(f"Copied most recent image to {args.output_dir}")
                        else:
                            logger.warning("No image was generated or image path is empty")
                except Exception as e:
                    logger.error(f"Error copying image: {e}")
                    # Try copying the latest image as a fallback
                    logger.info("Trying to copy the most recent image as fallback...")
                    try:
                        if args.remote:
                            if not args.ssh_host:
                                logger.error("SSH host is required when using --remote")
                                return prompts
                                
                            logger.info(f"Using rsync over SSH to copy latest image from {args.ssh_host}")
                            copied = rsync_latest_images_from_comfyui(
                                args.ssh_host,
                                args.comfy_output_dir,
                                args.output_dir,
                                limit=1,
                                ssh_port=args.ssh_port,
                                ssh_user=args.ssh_user,
                                ssh_key=args.ssh_key
                            )
                        else:
                            # Local copy
                            copied = copy_latest_images_from_comfyui(
                                args.comfy_output_dir, 
                                args.output_dir, 
                                limit=1
                            )
                            
                        if copied:
                            logger.info(f"Copied most recent image to {args.output_dir}")
                    except Exception as fallback_e:
                        logger.error(f"Fallback copy also failed: {fallback_e}")
                        logger.warning("Could not copy any images")

    return prompts
