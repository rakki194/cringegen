"""
Furry generation commands for cringegen
"""

import logging
import random
import time
import os

from ..prompt_generation.generators.furry_generator import FurryPromptGenerator
from ..utils.comfy_api import (
    check_comfy_server,
    check_generation_status,
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
    rsync_latest_images_from_comfyui,
    open_images_with_imv,
)
from ..workflows.furry import create_basic_furry_workflow

logger = logging.getLogger(__name__)


def add_furry_command(subparsers, parent_parser):
    """Add the furry command to the CLI"""
    furry_parser = subparsers.add_parser(
        "furry", help="Generate a furry prompt", parents=[parent_parser]
    )
    furry_parser.add_argument("--checkpoint", type=str, help="Checkpoint model to use")
    furry_parser.add_argument("--lora", type=str, help="LoRA model to use")
    furry_parser.add_argument(
        "--additional-loras",
        type=str,
        nargs="+",
        help="Additional LoRA models to stack (space separated)",
    )
    furry_parser.add_argument(
        "--lora-weights",
        type=float,
        nargs="+",
        help="Weights for additional LoRAs (space separated, must match number of additional LoRAs)",
    )
    furry_parser.add_argument("--count", type=int, default=1, help="Number of prompts to generate")
    furry_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for generation (ensures the same prompt is produced for the same seed value)",
    )
    furry_parser.add_argument("--species", type=str, help="Species for the character")
    furry_parser.add_argument("--gender", type=str, help="Gender for the character")

    # Multi-subject options
    furry_parser.add_argument(
        "--duo",
        action="store_true",
        help="Generate a scene with two characters",
    )
    furry_parser.add_argument(
        "--group",
        type=int,
        help="Generate a scene with specified number of characters (3-6)",
    )
    furry_parser.add_argument(
        "--species2",
        type=str,
        help="Species for the second character (when using --duo)",
    )
    furry_parser.add_argument(
        "--gender2",
        type=str,
        help="Gender for the second character (when using --duo)",
    )

    # Color options
    furry_parser.add_argument(
        "--colors",
        type=str,
        help="Comma-separated list of colors for the main character's fur/scales",
    )
    furry_parser.add_argument(
        "--colors2",
        type=str,
        help="Comma-separated list of colors for the second character (when using --duo)",
    )
    furry_parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern type for the main character (e.g., spotted, striped, etc.)",
    )
    furry_parser.add_argument(
        "--pattern2",
        type=str,
        help="Pattern type for the second character (when using --duo)",
    )

    furry_parser.add_argument(
        "--anthro",
        action="store_true",
        help="Include 'anthro' tag in the prompt for the subject",
    )
    furry_parser.add_argument(
        "--feral",
        action="store_true",
        help="Include 'feral' tag in the prompt for the subject",
    )
    furry_parser.add_argument(
        "--prompt", type=str, help="Custom prompt to use (overrides generated prompt)"
    )
    furry_parser.add_argument(
        "--negative-prompt",
        type=str,
        help="Custom negative prompt to use (overrides generated negative prompt)",
    )
    furry_parser.add_argument("--width", type=int, default=1024, help="Width of the image")
    furry_parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    furry_parser.add_argument("--steps", type=int, default=40, help="Number of steps")
    furry_parser.add_argument(
        "--cfg", type=float, default=None, help="CFG scale (defaults to model-specific value)"
    )
    furry_parser.add_argument("--lora-strength", type=float, default=1.0, help="LoRA strength")
    furry_parser.add_argument(
        "--sampler", type=str, help="Sampler to use (e.g., euler, euler_ancestral, ddim)"
    )
    furry_parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler to use (use 'list-schedulers' command to see available options)",
    )
    furry_parser.add_argument(
        "--no-generate", action="store_true", help="Only generate prompts, do not create images"
    )
    furry_parser.add_argument(
        "--no-art-style", action="store_true", help="Don't include random art style in the prompt"
    )
    furry_parser.add_argument(
        "--copy-output",
        action="store_true",
        help="Copy generated images to cringegen output directory",
    )
    furry_parser.add_argument(
        "--comfy-output-dir",
        type=str,
        default="/home/kade/toolkit/diffusion/comfy/ComfyUI/output",
        help="ComfyUI output directory",
    )
    furry_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated images",
    )
    furry_parser.add_argument(
        "--increment",
        action="store_true",
        help="Force seed increment by exactly +1 for each image when using --count",
    )

    # Remote ComfyUI options
    furry_parser.add_argument(
        "--remote",
        action="store_true",
        help="Use SSH to copy images from a remote ComfyUI instance",
    )
    furry_parser.add_argument(
        "--ssh-host",
        type=str,
        default="otter_den",
        help="SSH hostname or IP address for remote ComfyUI instance",
    )
    furry_parser.add_argument(
        "--ssh-port",
        type=int,
        default=1487,
        help="SSH port for remote ComfyUI instance",
    )
    furry_parser.add_argument(
        "--ssh-user",
        type=str,
        default="kade",
        help="SSH username for remote ComfyUI instance",
    )
    furry_parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key file for remote ComfyUI instance",
    )

    # Advanced workflow options
    furry_parser.add_argument(
        "--pag", action="store_true", help="Use Perturbed-Attention Guidance for improved detail"
    )
    furry_parser.add_argument(
        "--pag-scale", type=float, default=3.0, help="Scale for Perturbed-Attention Guidance"
    )
    furry_parser.add_argument(
        "--pag-sigma-start", type=float, default=-1.0, help="Start sigma for PAG (default: auto)"
    )
    furry_parser.add_argument(
        "--pag-sigma-end", type=float, default=-1.0, help="End sigma for PAG (default: auto)"
    )
    furry_parser.add_argument(
        "--detail-daemon",
        action="store_true",
        help="Use DetailDaemonSamplerNode for enhanced details",
    )
    furry_parser.add_argument(
        "--detail-amount",
        type=float,
        default=0.1,
        help="Detail amount for DetailDaemonSamplerNode (0.0-1.0)",
    )
    furry_parser.add_argument(
        "--detail-start",
        type=float,
        default=0.5,
        help="Start percent for DetailDaemonSamplerNode (0.0-1.0)",
    )
    furry_parser.add_argument(
        "--detail-end",
        type=float,
        default=0.8,
        help="End percent for DetailDaemonSamplerNode (0.0-1.0)",
    )
    furry_parser.add_argument(
        "--split-sigmas", type=float, help="Value to split sigmas for multi-stage sampling"
    )
    furry_parser.add_argument(
        "--split-first-cfg", type=float, help="CFG for first stage of split-sigma sampling"
    )
    furry_parser.add_argument(
        "--split-second-cfg", type=float, help="CFG for second stage of split-sigma sampling"
    )
    furry_parser.add_argument(
        "--split-first-sampler", type=str, help="Sampler for first stage of split-sigma sampling (e.g., euler, euler_ancestral)"
    )
    furry_parser.add_argument(
        "--split-second-sampler", type=str, help="Sampler for second stage of split-sigma sampling (e.g., euler, dpm_2_ancestral)"
    )
    furry_parser.add_argument(
        "--split-first-scheduler", type=str, help="Scheduler for first stage of split-sigma sampling (e.g., normal, karras)"
    )
    furry_parser.add_argument(
        "--split-second-scheduler", type=str, help="Scheduler for second stage of split-sigma sampling (e.g., normal, karras)"
    )
    furry_parser.add_argument(
        "--use-deepshrink",
        action="store_true",
        help="Use DeepShrink for improved high-frequency details",
    )
    furry_parser.add_argument(
        "--deepshrink-factor", type=float, default=2.0, help="Downscale factor for DeepShrink"
    )
    furry_parser.add_argument(
        "--deepshrink-start", type=float, default=0.0, help="Start percent for DeepShrink (0.0-1.0)"
    )
    furry_parser.add_argument(
        "--deepshrink-end", type=float, default=0.35, help="End percent for DeepShrink (0.0-1.0)"
    )
    furry_parser.add_argument(
        "--deepshrink-gradual",
        type=float,
        default=0.6,
        help="Gradual percent for DeepShrink (0.0-1.0)",
    )
    furry_parser.add_argument(
        "--use-zsnr", action="store_true", help="Enable Zero SNR for potentially improved results"
    )
    furry_parser.add_argument("--use-vpred", action="store_true", help="Use v-prediction sampling")

    furry_parser.set_defaults(func=generate_furry)
    return furry_parser


def generate_furry(args):
    """Generate a furry prompt and optionally an image"""
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

    # Create generator instance with all parameters upfront
    generator = FurryPromptGenerator(
        species=args.species,  # Pass species directly
        gender=args.gender,  # Pass gender directly
        colors=args.colors if args.colors else None,
        pattern=args.pattern,
        species2=args.species2 if args.duo else None,
        gender2=args.gender2 if args.duo else None,
        colors2=args.colors2 if args.duo and args.colors2 else None,
        pattern2=args.pattern2 if args.duo else None,
        use_duo=args.duo,
        group_size=max(3, min(6, args.group)) if args.group else 0,
        use_nlp=True,
        use_art_style=not args.no_art_style,
        is_anthro=True if args.anthro else (False if args.feral else True),
        is_feral=args.feral,
    )

    # Log the species and gender being used
    logger.info(f"Generating prompts with species={generator.species}, gender={generator.gender}")

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
        logger.info(f"Generated prompt (seed {curr_seed}):")
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

        # Continue with image generation
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
            workflow = create_basic_furry_workflow(
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
                use_pag=args.pag if hasattr(args, 'pag') else False,
                pag_scale=args.pag_scale if hasattr(args, 'pag_scale') else 3.0,
                pag_sigma_start=args.pag_sigma_start if hasattr(args, 'pag_sigma_start') else -1.0,
                pag_sigma_end=args.pag_sigma_end if hasattr(args, 'pag_sigma_end') else -1.0,
                use_detail_daemon=args.detail_daemon if hasattr(args, 'detail_daemon') else False,
                detail_amount=args.detail_amount if hasattr(args, 'detail_amount') else 0.1,
                detail_start=args.detail_start if hasattr(args, 'detail_start') else 0.5,
                detail_end=args.detail_end if hasattr(args, 'detail_end') else 0.8,
                split_sigmas=args.split_sigmas if hasattr(args, 'split_sigmas') else None,
                split_first_cfg=args.split_first_cfg if hasattr(args, 'split_first_cfg') else None,
                split_second_cfg=args.split_second_cfg if hasattr(args, 'split_second_cfg') else None,
                split_first_sampler=args.split_first_sampler if hasattr(args, 'split_first_sampler') else None,
                split_second_sampler=args.split_second_sampler if hasattr(args, 'split_second_sampler') else None,
                split_first_scheduler=args.split_first_scheduler if hasattr(args, 'split_first_scheduler') else None,
                split_second_scheduler=args.split_second_scheduler if hasattr(args, 'split_second_scheduler') else None,
                use_deepshrink=args.use_deepshrink if hasattr(args, 'use_deepshrink') else False,
                deepshrink_factor=args.deepshrink_factor if hasattr(args, 'deepshrink_factor') else 2.0,
                deepshrink_start=args.deepshrink_start if hasattr(args, 'deepshrink_start') else 0.0,
                deepshrink_end=args.deepshrink_end if hasattr(args, 'deepshrink_end') else 0.35,
                deepshrink_gradual=args.deepshrink_gradual if hasattr(args, 'deepshrink_gradual') else 0.6,
                use_zsnr=args.use_zsnr if hasattr(args, 'use_zsnr') else False,
                use_vpred=args.use_vpred if hasattr(args, 'use_vpred') else False,
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

            # Get image path if we need to copy it
            if args.copy_output:
                try:
                    # Poll for generation status with a reasonable timeout
                    status = None
                    max_poll_time = 300  # 5 minutes max
                    poll_interval = 2  # Check every 2 seconds
                    start_time = time.time()

                    logger.info(f"Monitoring generation status for prompt {prompt_id_str}")

                    # Poll until completed or timeout
                    while time.time() - start_time < max_poll_time:
                        status = check_generation_status(prompt_id_str, args.comfy_url)

                        # Print progress updates
                        if status["status"] == "pending":
                            logger.info(
                                f"Generation pending... ({int(time.time() - start_time)}s elapsed)"
                            )
                        elif status["status"] == "processing":
                            logger.info(
                                f"Generation in progress: {status['progress']*100:.1f}% ({int(time.time() - start_time)}s elapsed)"
                            )
                        elif status["status"] == "completed":
                            logger.info(
                                f"Generation completed in {int(time.time() - start_time)}s!"
                            )
                            break
                        elif status["status"] == "error":
                            logger.error(f"Generation error: {status['error']}")
                            break

                        # Wait before polling again
                        time.sleep(poll_interval)

                    # If generation completed successfully
                    if status and status["status"] == "completed" and status["images"]:
                        image_filename = status["images"][0]["filename"]
                        logger.info(f"Found image: {image_filename}")

                        success = False
                        if args.remote:
                            # Check for required SSH parameters
                            if not args.ssh_host:
                                logger.error("SSH host is required when using --remote")
                                return prompts

                            logger.info(f"Using rsync over SSH to copy image from {args.ssh_host}")
                            success = (
                                rsync_image_from_comfyui(
                                    image_filename,
                                    args.ssh_host,
                                    args.comfy_output_dir,
                                    args.output_dir,
                                    f"furry_{curr_seed}",
                                    ssh_port=args.ssh_port,
                                    ssh_user=args.ssh_user,
                                    ssh_key=args.ssh_key,
                                )
                                is not None
                            )
                        else:
                            # Local copy
                            success = (
                                copy_image_from_comfyui(
                                    image_filename,
                                    args.comfy_output_dir,
                                    args.output_dir,
                                    f"furry_{curr_seed}",
                                )
                                is not None
                            )

                        if success:
                            logger.info(f"Copied image to {args.output_dir}")
                            # If successful, add to the list of copied images
                            copied_path = os.path.join(
                                args.output_dir,
                                f"furry_{curr_seed}{os.path.splitext(image_filename)[1]}",
                            )
                            copied_images.append(copied_path)
                        else:
                            logger.warning("Failed to copy image, trying alternate methods...")
                            # Try with latest images as fallback
                            copied = []

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
                            else:
                                logger.warning("All image copy attempts failed")
                    else:
                        # If we couldn't get the image through the API status check
                        if status and status["status"] == "error":
                            logger.error(f"Generation failed: {status['error']}")
                        else:
                            logger.warning(
                                f"Generation timed out or didn't complete properly after {max_poll_time}s"
                            )

                        # Try to copy latest image as a fallback
                        logger.info("Trying to copy the most recent image instead...")

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
                        else:
                            logger.warning("No image found to copy")

                except Exception as e:
                    logger.error(f"Error during generation monitoring: {e}")
                    # Try copying the latest image as a fallback
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
                    except Exception as fallback_e:
                        logger.error(f"Fallback copy also failed: {fallback_e}")

        # Open images with imv if requested and we have any
        if args.show and copied_images:
            open_images_with_imv(copied_images)

    return prompts
