"""
Furry-specific ComfyUI workflows
"""

import random
from typing import Any, Dict, Optional

from ..utils.comfy_api import (
    get_available_samplers,
    get_available_schedulers,
    is_valid_sampler,
    is_valid_scheduler,
)
from ..workflows.base import ComfyWorkflow
from ..utils.logger import get_logger

# Create logger for this module
logger = get_logger(__name__)


def create_basic_furry_workflow(
    checkpoint: str,
    lora: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int = 20,
    cfg: float = None,  # Default will be set based on model
    lora_strength: float = 0.35,
    loras: Optional[list] = None,  # New parameter for multiple LoRAs
    lora_weights: Optional[list] = None,  # New parameter for LoRA weights
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    use_deepshrink: bool = False,
    use_pag: bool = False,
    use_zsnr: bool = False,
    use_vpred: bool = False,
    split_sigmas: Optional[float] = None,
    # New parameters
    split_first_cfg: Optional[float] = None,
    split_second_cfg: Optional[float] = None,
    pag_scale: float = 3.0,
    pag_sigma_start: float = -1.0,
    pag_sigma_end: float = -1.0,
    deepshrink_factor: float = 2.0,
    deepshrink_start: float = 0.0,
    deepshrink_end: float = 0.35,
    deepshrink_gradual: float = 0.6,
    use_detail_daemon: bool = False,
    detail_amount: float = 0.1,
    detail_start: float = 0.5,
    detail_end: float = 0.8,
    batch_size: int = 1,
    show: bool = False,
) -> Dict[str, Any]:
    """Create a basic workflow for furry image generation

    When using detail_daemon=True, this creates a special sampling path:
    - For normal sampling: Uses SamplerCustomAdvanced with DetailDaemonSamplerNode
      and DisableNoise to enhance details in the generation
    - For split-sigmas: Adds DetailDaemonSamplerNode to the second stage of sampling

    Args:
        checkpoint: Name of the checkpoint to use
        lora: Name of the LoRA to use
        prompt: The positive prompt
        negative_prompt: The negative prompt
        width: The width of the image
        height: The height of the image
        seed: The seed for generation
        steps: The number of sampling steps
        cfg: The CFG scale (defaults to model-specific value)
        lora_strength: The strength of the LoRA
        loras: List of additional LoRAs to use
        lora_weights: List of weights for additional LoRAs
        sampler: The sampler to use
        scheduler: The scheduler to use
        use_deepshrink: Whether to use PatchModelAddDownscale_v2
        use_pag: Whether to use Perturbed-Attention Guidance
        use_zsnr: Whether to enable Zero SNR
        use_vpred: Whether to use v-prediction sampling
        split_sigmas: Value to split sigmas for multi-stage sampling
        split_first_cfg: CFG for first stage of split-sigma sampling
        split_second_cfg: CFG for second stage of split-sigma sampling
        pag_scale: Scale for Perturbed-Attention Guidance
        pag_sigma_start: Start sigma for PAG
        pag_sigma_end: End sigma for PAG
        deepshrink_factor: Downscale factor for DeepShrink
        deepshrink_start: Start percent for DeepShrink
        deepshrink_end: End percent for DeepShrink
        deepshrink_gradual: Gradual percent for DeepShrink
        use_detail_daemon: Whether to use DetailDaemonSamplerNode
        detail_amount: Detail amount for DetailDaemonSamplerNode (0.0-1.0)
        detail_start: Start percent for DetailDaemonSamplerNode (0.0-1.0)
        detail_end: End percent for DetailDaemonSamplerNode (0.0-1.0)
        batch_size: The batch size for the image
        show: Whether to show the image

    Returns:
        A workflow dictionary
    """
    workflow = ComfyWorkflow()

    # Set model-specific default CFG if not provided
    if cfg is None:
        # Default CFG values based on model
        if "noobai" in checkpoint.lower():
            cfg = 3.5  # NoobAI models use lower CFG
        elif "ponydiffusion" in checkpoint.lower():
            cfg = 7.0  # PonyDiffusion models use higher CFG
        else:
            cfg = 7.0  # Default value for other models
        logger.info(f"Using model-specific default CFG: {cfg} for model: {checkpoint}")
    else:
        logger.info(f"Using user-specified CFG: {cfg}")

    # Flag to track whether a LoRA is being used
    using_lora = bool(lora) or (loras and len(loras) > 0)

    # Initialize loras and lora_weights if not provided
    loras = loras or []
    lora_weights = lora_weights or []

    # Load the checkpoint
    logger.info(f"Using checkpoint: {checkpoint}")
    model_node = workflow.add_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint})
    model_out = workflow.get_output(model_node, 0)
    clip_out = workflow.get_output(model_node, 1)
    vae_out = workflow.get_output(model_node, 2)

    # Add primary LoRA if specified
    if lora:
        logger.info(f"Using primary LoRA: {lora} (strength: {lora_strength})")
        lora_node = workflow.add_node(
            "LoraLoader",
            {
                "model": model_out,
                "clip": clip_out,
                "lora_name": lora,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        )
        model_out = workflow.get_output(lora_node, 0)
        clip_out = workflow.get_output(lora_node, 1)

    # Apply multiple LoRAs if specified
    if loras and len(loras) > 0:
        # Use default weights if not provided
        if not lora_weights or len(lora_weights) != len(loras):
            lora_weights = [0.35] * len(
                loras
            )  # Default weight of 0.35 (in the preferred 0.30-0.40 range)

        logger.info(f"Adding {len(loras)} {'additional' if lora else ''} LoRAs to workflow")
        for i, (lora_name, weight) in enumerate(zip(loras, lora_weights)):
            if not lora_name or not lora_name.strip():
                continue

            logger.info(f"  {i+1}. {lora_name} (weight: {weight})")
            lora_node = workflow.add_node(
                "LoraLoader",
                {
                    "model": model_out,
                    "clip": clip_out,
                    "lora_name": lora_name,
                    "strength_model": weight,
                    "strength_clip": weight,
                },
            )
            model_out = workflow.get_output(lora_node, 0)
            clip_out = workflow.get_output(lora_node, 1)

    if not using_lora:
        logger.info("No LoRAs specified, using base model only")

    # Apply ModelSamplingDiscrete if ZSNR or v-prediction is requested
    if use_zsnr or use_vpred:
        logger.info(f"Adding ModelSamplingDiscrete (ZSNR: {use_zsnr}, v-prediction: {use_vpred})")
        sampling_discrete_node = workflow.add_node(
            "ModelSamplingDiscrete",
            {
                "model": model_out,
                "sampling": "v_prediction" if use_vpred else "eps",
                "zsnr": use_zsnr,
            },
        )
        model_out = workflow.get_output(sampling_discrete_node, 0)

    # Apply DeepShrink if requested
    if use_deepshrink:
        logger.info(
            f"Adding DeepShrink (factor: {deepshrink_factor}, start: {deepshrink_start}, end: {deepshrink_end}, gradual: {deepshrink_gradual})"
        )
        deepshrink_node = workflow.add_node(
            "PatchModelAddDownscale_v2",
            {
                "model": model_out,
                "block_number": 3,
                "downscale_factor": deepshrink_factor,
                "start_percent": deepshrink_start,
                "end_percent": deepshrink_end,
                "gradual_percent": deepshrink_gradual,
                "downscale_after_skip": True,
                "downscale_method": "bicubic",
                "upscale_method": "bicubic",
            },
        )
        model_out = workflow.get_output(deepshrink_node, 0)

    # Apply Perturbed-Attention Guidance if requested
    if use_pag:
        logger.info(
            f"Adding Perturbed-Attention Guidance (scale: {pag_scale}, sigma_start: {pag_sigma_start}, sigma_end: {pag_sigma_end})"
        )
        pag_node = workflow.add_node(
            "PerturbedAttention",
            {
                "model": model_out,
                "scale": pag_scale,
                "adaptive_scale": 0,
                "unet_block": "middle",
                "unet_block_id": 0,
                "sigma_start": pag_sigma_start,
                "sigma_end": pag_sigma_end,
                "rescale": 0,
                "rescale_mode": "full",
                "unet_block_list": "",
            },
        )
        model_out = workflow.get_output(pag_node, 0)

    # Encode prompts
    positive = workflow.add_node("CLIPTextEncode", {"clip": clip_out, "text": prompt})
    positive_out = workflow.get_output(positive)

    negative = workflow.add_node("CLIPTextEncode", {"clip": clip_out, "text": negative_prompt})
    negative_out = workflow.get_output(negative)

    # Create empty latent
    latent = workflow.add_node(
        "EmptyLatentImage", {"width": width, "height": height, "batch_size": batch_size}
    )
    latent_out = workflow.get_output(latent)

    # Adjust sampling parameters when no LoRA is used
    if not using_lora:
        # For checkpoint-only generation, we need more steps and different parameters
        steps = max(steps, 40)  # Use at least 40 steps for checkpoint-only
        # Don't override the model-specific CFG value
        # cfg = min(max(cfg, 7.5), 8.5)  # Adjust CFG to 7.5-8.5 range for better results

    # Determine the sampler and scheduler to use
    if sampler:
        # User-specified sampler
        workflow_sampler = sampler
    else:
        # Always use euler as the default sampler
        workflow_sampler = "euler"

    # Determine the scheduler to use
    if scheduler:
        # User-specified scheduler
        workflow_scheduler = scheduler
    else:
        # Always use karras as the default scheduler
        workflow_scheduler = "karras"

    logger.info(f"Using sampler: {workflow_sampler}, scheduler: {workflow_scheduler}")

    # Split-sigma workflow if requested
    if split_sigmas is not None:
        logger.info(f"Using split-sigma workflow with split at {split_sigmas}")

        # Use provided CFGs or default values based on model
        first_cfg = split_first_cfg if split_first_cfg is not None else min(cfg, 3.0)

        # For split_second_cfg, use model-specific defaults if not specified
        if split_second_cfg is not None:
            second_cfg = split_second_cfg
        else:
            # Use the same model-specific logic for the second stage CFG
            if "noobai" in checkpoint.lower():
                second_cfg = 3.5  # NoobAI models
            else:
                second_cfg = cfg  # Other models use the main CFG value

        logger.info(f"Split-sigma CFGs: first stage = {first_cfg}, second stage = {second_cfg}")
        logger.info(
            f"Debug: Model = {'NoobAI' if 'noobai' in checkpoint.lower() else 'PonyDiffusion' if 'ponydiffusion' in checkpoint.lower() else 'Other'}, Main CFG = {cfg}"
        )

        # Create a KarrasScheduler with rho=7 (similar to noobai_api.json)
        karras_scheduler = workflow.add_node(
            "KarrasScheduler", {"steps": steps, "sigma_max": 100, "sigma_min": 0.0291675, "rho": 7}
        )
        sigmas_out = workflow.get_output(karras_scheduler, 0)

        # Split the sigmas at the specified value
        split_node = workflow.add_node(
            "SplitAtSigma", {"sigma": split_sigmas, "sigmas": sigmas_out}
        )
        first_sigmas_out = workflow.get_output(split_node, 0)
        second_sigmas_out = workflow.get_output(split_node, 1)

        # Create SplitSigmasDenoise
        split_sigmas_denoise = workflow.add_node(
            "SplitSigmasDenoise", {"denoise": 1.0, "sigmas": sigmas_out}
        )
        denoise_out = workflow.get_output(split_sigmas_denoise, 1)

        # Create noise
        noise_node = workflow.add_node("RandomNoise", {"noise_seed": seed})
        noise_out = workflow.get_output(noise_node, 0)

        # Create disable noise for second stage
        disable_noise = workflow.add_node("DisableNoise", {})
        disable_noise_out = workflow.get_output(disable_noise, 0)

        # Create first stage CFG guider
        first_cfg_guider = workflow.add_node(
            "CFGGuider",
            {
                "cfg": first_cfg,
                "model": model_out,
                "positive": positive_out,
                "negative": negative_out,
            },
        )
        first_guider_out = workflow.get_output(first_cfg_guider, 0)

        # Create second stage CFG guider
        second_cfg_guider = workflow.add_node(
            "CFGGuider",
            {
                "cfg": second_cfg,
                "model": model_out,
                "positive": positive_out,
                "negative": negative_out,
            },
        )
        second_guider_out = workflow.get_output(second_cfg_guider, 0)

        # Create first stage sampler (euler_ancestral)
        first_sampler_select = workflow.add_node(
            "KSamplerSelect", {"sampler_name": "euler_ancestral"}
        )
        first_sampler_out = workflow.get_output(first_sampler_select, 0)

        # Create second stage sampler (euler_ancestral)
        second_sampler_select = workflow.add_node("SamplerEulerAncestral", {"eta": 1, "s_noise": 1})
        second_sampler_out = workflow.get_output(second_sampler_select, 0)

        # Add DetailDaemonSamplerNode for second stage if requested
        if use_detail_daemon:
            logger.info(
                f"Adding DetailDaemonSampler (amount: {detail_amount}, start: {detail_start}, end: {detail_end})"
            )
            detail_daemon = workflow.add_node(
                "DetailDaemonSamplerNode",
                {
                    "detail_amount": detail_amount,
                    "start": detail_start,
                    "end": detail_end,
                    "bias": 0.5,
                    "exponent": 1,
                    "start_offset": 0,  # Changed to 0 to match the example JSON
                    "end_offset": 0,
                    "fade": 0,
                    "smooth": True,
                    "cfg_scale_override": 0,
                    "sampler": second_sampler_out,
                },
            )
            detail_daemon_out = workflow.get_output(detail_daemon, 0)

            # Add BlehForceSeedSampler with a fixed seed offset to ensure different sampling
            bleh_force_seed = workflow.add_node(
                "BlehForceSeedSampler",
                {
                    "seed_offset": 200,  # Fixed offset as in the example JSON
                    "sampler": detail_daemon_out,
                },
            )
            second_sampler_out = workflow.get_output(bleh_force_seed, 0)
        else:
            # When no detail daemon, just add seed offset to the sampler
            bleh_force_seed = workflow.add_node(
                "BlehForceSeedSampler", {"seed_offset": 200, "sampler": second_sampler_out}
            )
            second_sampler_out = workflow.get_output(bleh_force_seed, 0)

        # Create first sampling stage with SamplerCustomAdvanced
        first_stage = workflow.add_node(
            "SamplerCustomAdvanced",
            {
                "noise": noise_out,
                "guider": first_guider_out,
                "sampler": first_sampler_out,
                "sigmas": first_sigmas_out,
                "latent_image": latent_out,
            },
        )
        first_stage_out = workflow.get_output(first_stage, 0)

        # Create second sampling stage with SamplerCustomAdvanced
        second_stage = workflow.add_node(
            "SamplerCustomAdvanced",
            {
                "noise": disable_noise_out,
                "guider": second_guider_out,
                "sampler": second_sampler_out,
                "sigmas": second_sigmas_out,
                "latent_image": first_stage_out,
            },
        )
        sampler_out = workflow.get_output(second_stage, 0)
    else:
        # Standard single-stage sampling
        if use_detail_daemon:
            logger.info(
                f"Adding DetailDaemonSampler (amount: {detail_amount}, start: {detail_start}, end: {detail_end})"
            )

            # Create noise with the specified seed
            noise_node = workflow.add_node("RandomNoise", {"noise_seed": seed})
            noise_out = workflow.get_output(noise_node, 0)

            # Create a disable noise node for second stage
            disable_noise = workflow.add_node("DisableNoise", {})
            disable_noise_out = workflow.get_output(disable_noise, 0)

            # Create the base sampler (using euler_ancestral for first stage)
            base_sampler = workflow.add_node("KSamplerSelect", {"sampler_name": "euler_ancestral"})
            first_sampler_out = workflow.get_output(base_sampler, 0)

            # Create the second stage sampler (must be euler_ancestral as in the example JSON)
            euler_ancestral = workflow.add_node("SamplerEulerAncestral", {"eta": 1, "s_noise": 1})
            second_sampler_out = workflow.get_output(euler_ancestral, 0)

            # Create the DetailDaemonSamplerNode
            detail_daemon = workflow.add_node(
                "DetailDaemonSamplerNode",
                {
                    "detail_amount": detail_amount,
                    "start": detail_start,
                    "end": detail_end,
                    "bias": 0.5,
                    "exponent": 1,
                    "start_offset": 0,  # Changed to 0 to match example JSON
                    "end_offset": 0,
                    "fade": 0,
                    "smooth": True,
                    "cfg_scale_override": 0,
                    "sampler": second_sampler_out,
                },
            )
            detail_daemon_out = workflow.get_output(detail_daemon, 0)

            # Add seed offset to the detail daemon
            bleh_force_seed = workflow.add_node(
                "BlehForceSeedSampler", {"seed_offset": 200, "sampler": detail_daemon_out}
            )
            final_sampler_out = workflow.get_output(bleh_force_seed, 0)

            # Create CFG Guiders for both stages
            first_cfg_guider = workflow.add_node(
                "CFGGuider",
                {
                    "model": model_out,
                    "positive": positive_out,
                    "negative": negative_out,
                    "cfg": 1.0,  # Low CFG for first stage as in the example JSON
                },
            )
            first_guider_out = workflow.get_output(first_cfg_guider, 0)

            second_cfg_guider = workflow.add_node(
                "CFGGuider",
                {
                    "model": model_out,
                    "positive": positive_out,
                    "negative": negative_out,
                    "cfg": cfg,  # Full CFG for second stage
                },
            )
            second_guider_out = workflow.get_output(second_cfg_guider, 0)

            # Create a KarrasScheduler for sigmas
            karras_scheduler = workflow.add_node(
                "KarrasScheduler",
                {"steps": steps, "sigma_max": 100, "sigma_min": 0.0291675, "rho": 7},
            )
            sigmas_out = workflow.get_output(karras_scheduler, 0)

            # Split sigmas at a reasonable point
            split_sigma = workflow.add_node(
                "SplitAtSigma", {"sigma": 7.0, "sigmas": sigmas_out}  # Value from the example JSON
            )
            first_sigmas_out = workflow.get_output(split_sigma, 0)
            second_sigmas_out = workflow.get_output(split_sigma, 1)

            # Create first stage with SamplerCustomAdvanced
            first_stage = workflow.add_node(
                "SamplerCustomAdvanced",
                {
                    "noise": noise_out,
                    "guider": first_guider_out,
                    "sampler": first_sampler_out,
                    "sigmas": first_sigmas_out,
                    "latent_image": latent_out,
                },
            )
            first_stage_out = workflow.get_output(first_stage, 0)

            # Create second stage with SamplerCustomAdvanced
            second_stage = workflow.add_node(
                "SamplerCustomAdvanced",
                {
                    "noise": disable_noise_out,
                    "guider": second_guider_out,
                    "sampler": final_sampler_out,
                    "sigmas": second_sigmas_out,
                    "latent_image": first_stage_out,
                },
            )
            sampler_out = workflow.get_output(second_stage, 0)
        else:
            # Standard KSampler when not using DetailDaemon
            sampler = workflow.add_node(
                "KSampler",
                {
                    "model": model_out,
                    "positive": positive_out,
                    "negative": negative_out,
                    "latent_image": latent_out,
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": workflow_sampler,
                    "scheduler": workflow_scheduler,
                    "denoise": 1.0,
                },
            )
            sampler_out = workflow.get_output(sampler, 0)

    # Decode the latent
    decode = workflow.add_node("VAEDecode", {"vae": vae_out, "samples": sampler_out})
    decode_out = workflow.get_output(decode)

    # Save the image
    save_image = workflow.add_node(
        "SaveImage",
        {
            "images": decode_out,
            "filename_prefix": "cringegen_furry",
        },
    )

    # Add a PreviewImage node if show is enabled
    if show:
        preview_node = workflow.add_node(
            "PreviewImage",
            {
                "images": decode_out,
                "show_preview": True,
            },
        )

        # Add a shell command node to display the image with imv
        imv_node = workflow.add_node(
            "ExecuteShellCommand",
            {
                "save_output": decode_out,
                "command": 'imv "$PATH"',
                "auto_execute": True,
            },
        )

    return workflow.to_dict()


def create_nsfw_furry_workflow(
    checkpoint: str,
    lora: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int = 20,
    cfg: float = None,
    lora_strength: float = 0.35,
    loras: Optional[list] = None,
    lora_weights: Optional[list] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    use_deepshrink: bool = False,
    use_pag: bool = False,
    use_zsnr: bool = False,
    use_vpred: bool = False,
    split_sigmas: Optional[float] = None,
    split_first_cfg: Optional[float] = None,
    split_second_cfg: Optional[float] = None,
    pag_scale: float = 3.0,
    pag_sigma_start: float = -1.0,
    pag_sigma_end: float = -1.0,
    deepshrink_factor: float = 2.0,
    deepshrink_start: float = 0.0,
    deepshrink_end: float = 0.35,
    deepshrink_gradual: float = 0.6,
    use_detail_daemon: bool = False,
    detail_amount: float = 0.1,
    detail_start: float = 0.5,
    detail_end: float = 0.8,
    batch_size: int = 1,
    show: bool = False,
) -> Dict[str, Any]:
    """Create a workflow for NSFW furry image generation

    Args:
        checkpoint: Name of the checkpoint to use
        lora: Name of the LoRA to use
        prompt: The positive prompt
        negative_prompt: The negative prompt
        width: The width of the image
        height: The height of the image
        seed: The seed for generation
        steps: The number of sampling steps
        cfg: The CFG scale
        lora_strength: The strength of the LoRA
        loras: List of additional LoRAs to use
        lora_weights: List of weights for additional LoRAs
        sampler: The sampler to use
        scheduler: The scheduler to use
        use_deepshrink: Whether to use PatchModelAddDownscale_v2
        use_pag: Whether to use Perturbed-Attention Guidance
        use_zsnr: Whether to enable Zero SNR
        use_vpred: Whether to use v-prediction sampling
        split_sigmas: Value to split sigmas for multi-stage sampling
        split_first_cfg: CFG for first stage of split-sigma sampling
        split_second_cfg: CFG for second stage of split-sigma sampling
        pag_scale: Scale for Perturbed-Attention Guidance
        pag_sigma_start: Start sigma for PAG
        pag_sigma_end: End sigma for PAG
        deepshrink_factor: Downscale factor for DeepShrink
        deepshrink_start: Start percent for DeepShrink
        deepshrink_end: End percent for DeepShrink
        deepshrink_gradual: Gradual percent for DeepShrink
        use_detail_daemon: Whether to use DetailDaemonSamplerNode
        detail_amount: Detail amount for DetailDaemonSamplerNode
        detail_start: Start percent for DetailDaemonSamplerNode
        detail_end: End percent for DetailDaemonSamplerNode
        batch_size: The batch size for the image
        show: Whether to show the image with imv after generating

    Returns:
        A workflow dictionary
    """
    # For NSFW workflows, we just use the basic workflow with a different filename prefix
    workflow_dict = create_basic_furry_workflow(
        checkpoint=checkpoint,
        lora=lora,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        lora_strength=lora_strength,
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
        batch_size=batch_size,
        show=show,
    )

    # Modify the filename prefix to indicate NSFW content
    for node_id, node in workflow_dict.items():
        if node["class_type"] == "SaveImage":
            node["inputs"]["filename_prefix"] = "cringegen_nsfw_furry"

    return workflow_dict
