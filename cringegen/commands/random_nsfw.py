"""
Random NSFW generation command with Style LoRA randomization
"""

import logging
import os
import random
import re

from ..data.lora import (
    ARTIST_PATTERNS,
    EXCLUDED_LORAS,
    ARTIST_SELECTION_CONFIG,
    LORA_STRENGTH_RANGES,
    KINK_LORA_RECOMMENDATIONS,
)
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
    rsync_latest_images_from_comfyui,
    open_images_with_imv
)
from ..workflows.furry import create_nsfw_furry_workflow
from ..utils.lora_metadata.analyzer import analyze_lora_type, analyze_multiple_loras, get_loras_by_type

logger = logging.getLogger(__name__)

# Keep track of previously selected artists to encourage diversity
_previously_selected_artists = {}
_MAX_HISTORY = ARTIST_SELECTION_CONFIG["MAX_HISTORY"]
_SAME_RUN_PENALTY = ARTIST_SELECTION_CONFIG["SAME_RUN_PENALTY"]

def add_random_nsfw_command(subparsers, parent_parser):
    """Add the random-nsfw command to the CLI"""
    random_nsfw_parser = subparsers.add_parser(
        "random-nsfw",
        help="Generate a random NSFW furry prompt with randomized style LoRAs",
        parents=[parent_parser],
    )
    random_nsfw_parser.add_argument("--checkpoint", type=str, help="Checkpoint model to use")
    random_nsfw_parser.add_argument(
        "--count", type=int, default=1, help="Number of prompts to generate"
    )
    random_nsfw_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for generation (ensures the same prompt is produced for the same seed value)",
    )
    random_nsfw_parser.add_argument("--species", type=str, help="Species for the character")
    random_nsfw_parser.add_argument(
        "--gender", type=str, help="Gender for the character (male, female, etc.)"
    )

    # NSFW-specific options
    random_nsfw_parser.add_argument(
        "--intensity",
        type=str,
        choices=["suggestive", "mild", "explicit", "hardcore"],
        default="explicit",
        help="Intensity of NSFW content",
    )
    random_nsfw_parser.add_argument(
        "--kinks",
        type=str,
        help="Comma-separated list of kinks to include in the prompt",
    )
    random_nsfw_parser.add_argument(
        "--no-kinks",
        action="store_true",
        help="Do not include any explicit or implicit kinks in the prompt",
    )

    # Color options
    random_nsfw_parser.add_argument(
        "--colors",
        type=str,
        help="Comma-separated list of colors for the character's fur/scales",
    )
    random_nsfw_parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern type for the character (e.g., spotted, striped, etc.)",
    )

    random_nsfw_parser.add_argument(
        "--anthro",
        action="store_true",
        help="Include 'anthro' tag in the prompt for the subject",
    )
    random_nsfw_parser.add_argument(
        "--feral",
        action="store_true",
        help="Include 'feral' tag in the prompt for the subject",
    )
    random_nsfw_parser.add_argument(
        "--prompt", type=str, help="Custom prompt to use (overrides generated prompt)"
    )
    random_nsfw_parser.add_argument(
        "--negative-prompt",
        type=str,
        help="Custom negative prompt to use (overrides generated negative prompt)",
    )
    random_nsfw_parser.add_argument("--width", type=int, default=1024, help="Width of the image")
    random_nsfw_parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    random_nsfw_parser.add_argument("--steps", type=int, default=40, help="Number of steps")
    random_nsfw_parser.add_argument(
        "--cfg", type=float, default=None, help="CFG scale (defaults to model-specific value)"
    )
    random_nsfw_parser.add_argument(
        "--min-lora-strength",
        type=float,
        default=0.3,
        help="Minimum strength for randomly selected LoRAs",
    )
    random_nsfw_parser.add_argument(
        "--max-lora-strength",
        type=float,
        default=0.7,
        help="Maximum strength for randomly selected LoRAs",
    )
    random_nsfw_parser.add_argument(
        "--sampler", type=str, help="Sampler to use (e.g., euler, euler_ancestral, ddim)"
    )
    random_nsfw_parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler to use (use 'list-schedulers' command to see available options)",
    )
    random_nsfw_parser.add_argument(
        "--no-generate", action="store_true", help="Only generate prompts, do not create images"
    )
    random_nsfw_parser.add_argument(
        "--no-art-style", action="store_true", help="Don't include random art style in the prompt"
    )
    random_nsfw_parser.add_argument(
        "--copy-output",
        action="store_true",
        help="Copy generated images to cringegen output directory",
    )
    
    # Add advanced workflow options
    random_nsfw_parser.add_argument(
        "--pag", 
        action="store_true", 
        help="Use Perturbed-Attention Guidance for improved detail"
    )
    random_nsfw_parser.add_argument(
        "--pag-scale", 
        type=float, 
        default=3.0, 
        help="Scale for Perturbed-Attention Guidance"
    )
    random_nsfw_parser.add_argument(
        "--pag-sigma-start", 
        type=float, 
        default=-1.0, 
        help="Start sigma for PAG (default: auto)"
    )
    random_nsfw_parser.add_argument(
        "--pag-sigma-end", 
        type=float, 
        default=-1.0, 
        help="End sigma for PAG (default: auto)"
    )
    random_nsfw_parser.add_argument(
        "--detail-daemon", 
        action="store_true", 
        help="Use DetailDaemonSamplerNode for enhanced details"
    )
    random_nsfw_parser.add_argument(
        "--detail-amount", 
        type=float, 
        default=0.1, 
        help="Detail amount for DetailDaemonSamplerNode (0.0-1.0)"
    )
    random_nsfw_parser.add_argument(
        "--detail-start", 
        type=float, 
        default=0.5, 
        help="Start percent for DetailDaemonSamplerNode (0.0-1.0)"
    )
    random_nsfw_parser.add_argument(
        "--detail-end", 
        type=float, 
        default=0.8, 
        help="End percent for DetailDaemonSamplerNode (0.0-1.0)"
    )
    random_nsfw_parser.add_argument(
        "--split-sigmas", 
        type=float, 
        help="Value to split sigmas for multi-stage sampling"
    )
    random_nsfw_parser.add_argument(
        "--split-first-cfg", 
        type=float, 
        help="CFG for first stage of split-sigma sampling"
    )
    random_nsfw_parser.add_argument(
        "--split-second-cfg", 
        type=float, 
        help="CFG for second stage of split-sigma sampling"
    )
    random_nsfw_parser.add_argument(
        "--use-deepshrink", 
        action="store_true", 
        help="Use DeepShrink for improved high-frequency details"
    )
    random_nsfw_parser.add_argument(
        "--deepshrink-factor", 
        type=float, 
        default=2.0, 
        help="Downscale factor for DeepShrink"
    )
    random_nsfw_parser.add_argument(
        "--deepshrink-start", 
        type=float, 
        default=0.0, 
        help="Start percent for DeepShrink (0.0-1.0)"
    )
    random_nsfw_parser.add_argument(
        "--deepshrink-end", 
        type=float, 
        default=0.35, 
        help="End percent for DeepShrink (0.0-1.0)"
    )
    random_nsfw_parser.add_argument(
        "--deepshrink-gradual", 
        type=float, 
        default=0.6, 
        help="Gradual percent for DeepShrink (0.0-1.0)"
    )
    random_nsfw_parser.add_argument(
        "--use-zsnr", 
        action="store_true", 
        help="Enable Zero SNR for potentially improved results"
    )
    random_nsfw_parser.add_argument(
        "--use-vpred", 
        action="store_true", 
        help="Use v-prediction sampling"
    )
    
    random_nsfw_parser.add_argument(
        "--comfy-output-dir",
        type=str,
        default="/home/kade/toolkit/diffusion/comfy/ComfyUI/output",
        help="ComfyUI output directory",
    )
    random_nsfw_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated images",
    )
    random_nsfw_parser.add_argument(
        "--increment",
        action="store_true",
        help="Force seed increment by exactly +1 for each image when using --count",
    )

    # Remote ComfyUI options
    random_nsfw_parser.add_argument(
        "--remote",
        action="store_true",
        help="Use SSH to copy images from a remote ComfyUI instance",
    )
    random_nsfw_parser.add_argument(
        "--ssh-host",
        type=str,
        default="otter_den",
        help="SSH hostname or IP address for remote ComfyUI instance",
    )
    random_nsfw_parser.add_argument(
        "--ssh-port",
        type=int,
        default=1487,
        help="SSH port for remote ComfyUI instance",
    )
    random_nsfw_parser.add_argument(
        "--ssh-user",
        type=str,
        default="kade",
        help="SSH username for remote ComfyUI instance",
    )
    random_nsfw_parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key file for remote ComfyUI instance",
    )

    random_nsfw_parser.set_defaults(func=generate_random_nsfw)
    return random_nsfw_parser


def get_artist_name(lora_path):
    """Extract the artist name from a LoRA path
    
    Args:
        lora_path: Path to the LoRA file
        
    Returns:
        The artist name or None if it can't be determined
    """
    filename = os.path.basename(lora_path)
    
    # Special case handling for known artists with multiple LoRAs
    if "bill_watterson" in filename:
        return "bill_watterson"
    
    # Check for common artist LoRA naming patterns
    for pattern in ARTIST_PATTERNS:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    # Try to extract artist name from the directory structure
    parts = lora_path.split('/')
    if len(parts) > 1:
        # Check if the parent directory might be the artist name
        parent_dir = parts[-2]
        if parent_dir != "noob" and parent_dir != "loras":
            return parent_dir
    
    return None


def update_artist_history(selected_artists):
    """Update the history of selected artists
    
    Args:
        selected_artists: Set of artist names that were selected in this run
    """
    global _previously_selected_artists
    
    # Increment the age of all existing entries
    for artist in list(_previously_selected_artists.keys()):
        _previously_selected_artists[artist] += 1
        # Remove artists that are too old
        if _previously_selected_artists[artist] > _MAX_HISTORY:
            del _previously_selected_artists[artist]
    
    # Add new selections with age 1 (freshly used)
    for artist in selected_artists:
        _previously_selected_artists[artist] = 1


def get_random_style_loras(count=3):
    """Get random style LoRAs from the noob/ directory, ensuring variety in artists

    Args:
        count: Number of LoRAs to select

    Returns:
        List of tuples (lora_path, strength)
    """
    # Get the LoRA directory
    lora_dir = get_lora_directory()

    # Check if the noob directory exists
    noob_dir = os.path.join(lora_dir, "noob")
    if not os.path.exists(noob_dir) or not os.path.isdir(noob_dir):
        logger.warning(f"Noob directory not found at {noob_dir}, using main LoRA directory")
        noob_dir = lora_dir

    # Get all .safetensors files in the directory
    lora_files = []
    for root, _, files in os.walk(noob_dir):
        for file in files:
            if file.endswith(".safetensors"):
                # Get the relative path from the lora_dir
                rel_path = os.path.relpath(os.path.join(root, file), lora_dir)
                lora_files.append(rel_path)

    # Filter out non-style LoRAs
    style_loras = []
    
    # Skip excluded LoRAs
    for lora_path in lora_files:
        if lora_path in EXCLUDED_LORAS:
            continue
            
        # For other LoRAs, try to analyze their type
        try:
            # Analyze the LoRA type
            full_path = os.path.join(lora_dir, lora_path)
            analysis = analyze_lora_type(full_path)
            lora_type = analysis.get("type", "unknown")
            
            # Only include style LoRAs
            if lora_type == "style":
                style_loras.append(lora_path)
        except Exception as e:
            logger.warning(f"Error analyzing LoRA {lora_path}: {e}")
            # If analysis fails, check if it matches known style patterns
            if "realistic" in lora_path.lower() or "style" in lora_path.lower():
                style_loras.append(lora_path)

    # If we don't have enough style LoRAs, use whatever is available
    if len(style_loras) < count:
        logger.warning(f"Not enough style LoRAs found (only {len(style_loras)}), including some non-style LoRAs")
        remaining_count = count - len(style_loras)
        non_style_loras = [lora for lora in lora_files if lora not in style_loras and lora not in EXCLUDED_LORAS]
        if non_style_loras:
            additional_loras = random.sample(non_style_loras, min(remaining_count, len(non_style_loras)))
            style_loras.extend(additional_loras)

    # Group LoRAs by artist
    loras_by_artist = {}
    for lora in style_loras:
        artist = get_artist_name(lora)
        if not artist:
            # For LoRAs without identified artists, use the filename as the key
            artist = os.path.basename(lora)
        
        if artist not in loras_by_artist:
            loras_by_artist[artist] = []
        loras_by_artist[artist].append(lora)
    
    # Determine weights for each artist based on usage history
    artists = list(loras_by_artist.keys())
    artist_weights = {}
    
    for artist in artists:
        # Default weight is 1.0 (never used)
        weight = 1.0
        
        # If the artist was recently used, reduce its weight
        if artist in _previously_selected_artists:
            age = _previously_selected_artists[artist]
            # The more recently used, the lower the weight (from 0.1 to 0.9)
            weight = min(0.1 + (0.9 * age / _MAX_HISTORY), 0.9)
        
        artist_weights[artist] = weight
    
    # Print debug info about artist weights - uncomment if needed
    logger.debug("Artist weights:")
    for artist, weight in sorted(artist_weights.items(), key=lambda x: x[1]):
        logger.debug(f"  {artist}: {weight:.2f}")
    
    # Select artists with weighted probability, avoiding duplicates
    selected_loras = []
    selected_artists = set()
    
    # If we have bill_watterson LoRAs, make sure we only select one version
    bill_watterson_loras = []
    for artist in artists:
        if "bill_watterson" in artist:
            bill_watterson_loras.extend(loras_by_artist[artist])
    
    # If we found multiple Bill Watterson LoRAs, choose only one
    if bill_watterson_loras:
        bill_watterson_selected = random.choice(bill_watterson_loras)
        selected_loras.append(bill_watterson_selected)
        selected_artists.add("bill_watterson")
        
        # Remove all bill_watterson artists from consideration
        for artist in list(artist_weights.keys()):
            if "bill_watterson" in artist:
                del artist_weights[artist]
    
    # Now select the remaining LoRAs
    remaining_count = count - len(selected_loras)
    
    while remaining_count > 0 and artist_weights:
        # Convert weights to a list in the same order as the artists
        available_artists = list(artist_weights.keys())
        weights = [artist_weights[a] for a in available_artists]
        
        # Select an artist based on weights
        if not available_artists:  # Safety check
            break
        
        selected_artist = random.choices(available_artists, weights=weights, k=1)[0]
        
        # Select a random LoRA from this artist
        artist_loras = loras_by_artist[selected_artist]
        selected_lora = random.choice(artist_loras)
        selected_loras.append(selected_lora)
        selected_artists.add(selected_artist)
        
        # Remove this artist from consideration for future selections in this run
        del artist_weights[selected_artist]
        
        remaining_count -= 1
    
    # Update the history with the artists we selected
    update_artist_history(selected_artists)
    
    # If we couldn't select enough unique artists, log a warning
    if len(selected_loras) < count:
        logger.warning(f"Not enough unique artist LoRAs found (only {len(selected_loras)}), using fewer LoRAs than requested")
    
    return selected_loras


def weighted_random_strength():
    """Generate a random LoRA strength with preference for the 0.30-0.40 range.
    
    Occasionally goes up to 0.50, rarely to 0.60, and very rarely to 0.70.
    
    Returns:
        A float value between 0.30 and 0.70
    """
    # Select a range based on their probabilities
    selected_range = random.choices(
        LORA_STRENGTH_RANGES, 
        weights=[r[2] for r in LORA_STRENGTH_RANGES],
        k=1
    )[0]
    
    # Generate a random value within the selected range
    return round(random.uniform(selected_range[0], selected_range[1]), 2)


def generate_random_nsfw(args):
    """Generate a random NSFW furry prompt with randomized style LoRAs"""
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
        logger.info(f"Using seed increment mode: Starting with seed {seed} and incrementing by +1 for each image")

    # Set up NSFW intensity
    intensity_levels = {"suggestive": 1, "mild": 2, "explicit": 3, "hardcore": 4}
    explicit_level = intensity_levels.get(args.intensity, 2)

    # Parse kinks
    kinks = []
    if args.kinks:
        kinks = [k.strip() for k in args.kinks.split(",")]
    
    # Check for kink-related terms and recommend specific LoRAs
    for kink_name, kink_data in KINK_LORA_RECOMMENDATIONS.items():
        has_kink = any(kink.lower() in kink_data["terms"] for kink in kinks)
        
        if has_kink and not args.checkpoint:
            logger.info(f"Detected {kink_name}-related kink. Recommending {kink_data['lora']} LoRA for better results.")
            logger.info(f"Example usage: --checkpoint {kink_data['checkpoint']} with {kink_data['lora']} LoRA")
            logger.info("Consider using a direct command instead of random-nsfw for better control: ")
            logger.info(f"cringegen nsfw-furry --lora {kink_data['lora']} --checkpoint {kink_data['checkpoint']}")
    
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
    
    # Set colors and patterns
    if args.colors:
        generator.colors = [c.strip() for c in args.colors.split(",")]
    if args.pattern:
        generator.pattern = args.pattern
    
    # Log the species and gender being used
    logger.info(f"Generating random NSFW prompts with species={generator.species}, gender={generator.gender}")

    # Get random style LoRAs
    style_loras = get_random_style_loras(3)

    # Generate random strengths for each LoRA
    lora_strengths = []
    for _ in range(len(style_loras)):
        if args.min_lora_strength != 0.3 or args.max_lora_strength != 0.7:
            # If user specified custom min/max, use uniform distribution
            strength = random.uniform(args.min_lora_strength, args.max_lora_strength)
            lora_strengths.append(round(strength, 2))
        else:
            # Use weighted distribution favoring lower values
            lora_strengths.append(weighted_random_strength())

    # Debug output for selected LoRAs
    logger.info(f"Selected {len(style_loras)} random style LoRAs:")
    for i, lora_name in enumerate(style_loras):
        logger.info(f"  - {lora_name} (strength: {lora_strengths[i]})")

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
        logger.info(f"Generated random NSFW prompt (seed {curr_seed}):")
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

        # Check for checkpoint
        if not args.checkpoint:
            # Use preferred checkpoint instead of first available
            args.checkpoint = get_preferred_checkpoint(args.comfy_url)
        else:
            args.checkpoint = get_matching_checkpoint(args.checkpoint, args.comfy_url)

        if not args.checkpoint:
            logger.error("No checkpoint specified and none found automatically.")
            return

        for i, (prompt, negative_prompt, curr_seed) in enumerate(prompts):
            # Create workflow for this prompt
            workflow = create_nsfw_furry_workflow(
                prompt=prompt,
                negative_prompt=negative_prompt,
                checkpoint=args.checkpoint,
                lora=None,
                loras=style_loras,
                lora_weights=lora_strengths,
                seed=curr_seed,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg=args.cfg,
                lora_strength=1.0,  # Not used since we're using loras instead
                sampler=args.sampler,
                scheduler=args.scheduler,
                # Add advanced workflow options
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
                                f"random_nsfw_{curr_seed}",
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
                                f"random_nsfw_{curr_seed}",
                            ) is not None
                            
                        if success:
                            logger.info(f"Copied image to {args.output_dir}")
                            # If successful, add to the list of copied images
                            copied_path = os.path.join(args.output_dir, f"random_nsfw_{curr_seed}{os.path.splitext(image_path)[1]}")
                            copied_images.append(copied_path)
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
                                # Add to the list of copied images
                                copied_images.extend(copied)
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
                            # Add to the list of copied images
                            copied_images.extend(copied)
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
                            # Add to the list of copied images
                            copied_images.extend(copied)
                    except Exception as fallback_e:
                        logger.error(f"Fallback copy also failed: {fallback_e}")
                        logger.warning("Could not copy any images")

    # Open images with imv if requested and we have any
    if args.show and copied_images:
        open_images_with_imv(copied_images)
                                
    return prompts
