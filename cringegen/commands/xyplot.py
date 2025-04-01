"""
XY Plot generation commands for cringegen

This module provides functionality for generating XY plot grids that vary parameters
across X and Y axes to visualize their impact on generated images.
"""

import logging
import os
import tempfile
import json
import subprocess
import time
import pprint
from typing import List, Dict, Any, Tuple, Optional, Callable
import re
import pathlib

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
    ensure_dir_exists,
    rsync_image_from_comfyui,
    open_images_with_imv,
    rsync_latest_images_from_comfyui,
    copy_latest_images_from_comfyui,
)
from ..workflows.base import get_workflow_template

logger = logging.getLogger(__name__)

# Dictionary to store parameter handler functions
param_handlers = {}


def register_param_handler(param_type: str):
    """Decorator to register parameter handlers"""

    def decorator(func):
        param_handlers[param_type] = func
        return func

    return decorator


def add_xyplot_command(subparsers, parent_parser):
    """Add the xyplot command to the CLI"""
    xyplot_parser = subparsers.add_parser(
        "xyplot", help="Generate an XY plot varying parameters", parents=[parent_parser]
    )

    # Base parameters
    xyplot_parser.add_argument(
        "--workflow",
        type=str,
        default="furry",
        choices=["furry", "nsfw", "character"],
        help="Workflow type to use for generation",
    )
    xyplot_parser.add_argument("--checkpoint", type=str, help="Base checkpoint model to use")
    xyplot_parser.add_argument(
        "--prompt",
        type=str,
        help="Base prompt to use (if not specified, a random one will be generated)",
    )
    xyplot_parser.add_argument("--negative-prompt", type=str, help="Base negative prompt to use")

    # XY parameters
    xyplot_parser.add_argument(
        "--x-param",
        type=str,
        required=True,
        choices=[
            "checkpoint",
            "lora",
            "loras",
            "sampler",
            "scheduler",
            "cfg",
            "steps",
            "lora_weight",
            "seed",
            "prompt_variation",
            "split_sigmas",
            "detail_daemon",
            "detail_amount",
            "detail_start",
            "detail_end",
            "detail_exponent",
            "pag",
            "pag_scale",
            "pag_sigma_start",
            "pag_sigma_end",
            "deepshrink",
            "deepshrink_factor",
            "deepshrink_start",
            "deepshrink_end",
            "deepshrink_gradual",
        ],
        help="Parameter to vary on the X axis",
    )
    xyplot_parser.add_argument(
        "--y-param",
        type=str,
        required=True,
        choices=[
            "checkpoint",
            "lora",
            "loras",
            "sampler",
            "scheduler",
            "cfg",
            "steps",
            "lora_weight",
            "seed",
            "prompt_variation",
            "split_sigmas",
            "detail_daemon",
            "detail_amount",
            "detail_start",
            "detail_end",
            "detail_exponent",
            "pag",
            "pag_scale",
            "pag_sigma_start",
            "pag_sigma_end",
            "deepshrink",
            "deepshrink_factor",
            "deepshrink_start",
            "deepshrink_end",
            "deepshrink_gradual",
        ],
        help="Parameter to vary on the Y axis",
    )
    xyplot_parser.add_argument(
        "--x-values", type=str, required=True, help="Comma-separated values for X axis parameter"
    )
    xyplot_parser.add_argument(
        "--y-values", type=str, required=True, help="Comma-separated values for Y axis parameter"
    )

    # Plot settings
    xyplot_parser.add_argument(
        "--width", type=int, default=1024, help="Width of each individual image"
    )
    xyplot_parser.add_argument(
        "--height", type=int, default=1024, help="Height of each individual image"
    )
    xyplot_parser.add_argument(
        "--label-alignment",
        type=str,
        default="center",
        choices=["start", "center", "end"],
        help="Alignment for axis labels",
    )
    xyplot_parser.add_argument(
        "--font-size", type=float, default=40.0, help="Font size for labels"
    )
    xyplot_parser.add_argument(
        "--horizontal-spacing", type=int, default=0, help="Horizontal spacing between images (pixels)"
    )
    xyplot_parser.add_argument(
        "--vertical-spacing", type=int, default=0, help="Vertical spacing between images (pixels)"
    )
    xyplot_parser.add_argument(
        "--debug-mode", action="store_true", help="Enable debug mode for layout visualization"
    )

    # Debug options
    xyplot_parser.add_argument(
        "--dump-workflows", action="store_true", help="Dump workflow JSON files for debugging"
    )

    # Output settings
    xyplot_parser.add_argument(
        "--output-name", type=str, default="xyplot", help="Base name for output grid image"
    )
    xyplot_parser.add_argument(
        "--comfy-output-dir",
        type=str,
        default="/home/kade/toolkit/diffusion/comfy/ComfyUI/output",
        help="ComfyUI output directory",
    )
    xyplot_parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory for generated images"
    )
    
    # Generation parameters
    xyplot_parser.add_argument(
        "--seed", type=int, help="Global seed to use for generation (if not varying seed as a parameter)"
    )

    # PAG (Perturbed-Attention Guidance) options
    xyplot_parser.add_argument(
        "--pag", action="store_true", help="Use Perturbed-Attention Guidance for better details"
    )
    xyplot_parser.add_argument("--pag-scale", type=float, default=3.0, help="Scale for PAG")
    xyplot_parser.add_argument(
        "--pag-sigma-start", type=float, default=-1.0, help="Start sigma for PAG (default: auto)"
    )
    xyplot_parser.add_argument(
        "--pag-sigma-end", type=float, default=-1.0, help="End sigma for PAG (default: auto)"
    )

    # DeepShrink options
    xyplot_parser.add_argument(
        "--deepshrink",
        action="store_true",
        help="Use DeepShrink for improved high-frequency details",
    )
    xyplot_parser.add_argument(
        "--deepshrink-factor", type=float, default=2.0, help="Downscale factor for DeepShrink"
    )
    xyplot_parser.add_argument(
        "--deepshrink-start", type=float, default=0.0, help="Start percent for DeepShrink (0.0-1.0)"
    )
    xyplot_parser.add_argument(
        "--deepshrink-end", type=float, default=0.35, help="End percent for DeepShrink (0.0-1.0)"
    )
    xyplot_parser.add_argument(
        "--deepshrink-gradual",
        type=float,
        default=0.6,
        help="Gradual percent for DeepShrink (0.0-1.0)",
    )

    # Split-Sigmas options
    xyplot_parser.add_argument(
        "--split-sigmas", type=float, help="Value to split sigmas for multi-stage sampling"
    )
    xyplot_parser.add_argument(
        "--split-first-cfg", type=float, help="CFG for first stage of split-sigma sampling"
    )
    xyplot_parser.add_argument(
        "--split-second-cfg", type=float, help="CFG for second stage of split-sigma sampling"
    )
    xyplot_parser.add_argument(
        "--split-first-sampler", type=str, help="Sampler for first stage of split-sigma sampling (e.g., euler, euler_ancestral)"
    )
    xyplot_parser.add_argument(
        "--split-second-sampler", type=str, help="Sampler for second stage of split-sigma sampling (e.g., euler, dpm_2_ancestral)"
    )
    xyplot_parser.add_argument(
        "--split-first-scheduler", type=str, help="Scheduler for first stage of split-sigma sampling (e.g., normal, karras)"
    )
    xyplot_parser.add_argument(
        "--split-second-scheduler", type=str, help="Scheduler for second stage of split-sigma sampling (e.g., normal, karras)"
    )
    
    # Detail Daemon options
    xyplot_parser.add_argument(
        "--detail-daemon",
        action="store_true",
        help="Use DetailDaemonSamplerNode for enhanced details",
    )
    xyplot_parser.add_argument(
        "--detail-amount",
        type=float,
        default=0.1,
        help="Detail amount for DetailDaemonSamplerNode (0.0-1.0)",
    )
    xyplot_parser.add_argument(
        "--detail-start",
        type=float,
        default=0.5,
        help="Start percent for DetailDaemonSamplerNode (0.0-1.0)",
    )
    xyplot_parser.add_argument(
        "--detail-end",
        type=float,
        default=0.8,
        help="End percent for DetailDaemonSamplerNode (0.0-1.0)",
    )
    xyplot_parser.add_argument(
        "--detail-exponent", type=float, default=1.5, help="Exponent for detail strength (higher = stronger effect)"
    )

    # Remote ComfyUI options
    xyplot_parser.add_argument(
        "--remote",
        action="store_true",
        help="Use SSH to copy images from a remote ComfyUI instance",
    )
    xyplot_parser.add_argument(
        "--ssh-host",
        type=str,
        default="otter_den",
        help="SSH hostname or IP address for remote ComfyUI instance",
    )
    xyplot_parser.add_argument(
        "--ssh-port",
        type=int,
        default=1487,
        help="SSH port for remote ComfyUI instance",
    )
    xyplot_parser.add_argument(
        "--ssh-user",
        type=str,
        default="kade",
        help="SSH username for remote ComfyUI instance",
    )
    xyplot_parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key file for remote ComfyUI instance",
    )

    # Add a separator between sections
    xyplot_parser.add_argument("--lora", type=str, help="Base LoRA to use")
    xyplot_parser.add_argument(
        "--lora-strength", type=float, default=0.35, help="Strength for the LoRA"
    )
    xyplot_parser.add_argument(
        "--loras", type=str, help="Comma-separated list of additional LoRAs to use"
    )
    xyplot_parser.add_argument(
        "--lora-weights", type=str, help="Comma-separated list of weights for additional LoRAs"
    )

    xyplot_parser.set_defaults(func=generate_xyplot)
    return xyplot_parser


@register_param_handler("checkpoint")
def handle_checkpoint_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle checkpoint parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Find the correct checkpoint node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "CheckpointLoaderSimple":
            node["inputs"]["ckpt_name"] = value
            logger.debug(f"Set checkpoint to {value}")
            break

    return workflow_copy


@register_param_handler("lora")
def handle_lora_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle LoRA parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Find the correct LoRA node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "LoraLoader":
            node["inputs"]["lora_name"] = value
            logger.debug(f"Set LoRA to {value}")
            break

    return workflow_copy


@register_param_handler("sampler")
def handle_sampler_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle sampler parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Find the correct sampler node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "KSampler" or node["class_type"] == "KSamplerAdvanced":
            node["inputs"]["sampler_name"] = value
            logger.debug(f"Set sampler to {value}")
            break

    return workflow_copy


@register_param_handler("scheduler")
def handle_scheduler_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle scheduler parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Find the correct sampler node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "KSampler" or node["class_type"] == "KSamplerAdvanced":
            node["inputs"]["scheduler"] = value
            logger.debug(f"Set scheduler to {value}")
            break

    return workflow_copy


@register_param_handler("cfg")
def handle_cfg_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle CFG scale parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Convert value to float
    cfg_value = float(value)

    # Find the correct sampler node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "KSampler" or node["class_type"] == "KSamplerAdvanced":
            node["inputs"]["cfg"] = cfg_value
            logger.debug(f"Set CFG to {cfg_value}")
            break

    return workflow_copy


@register_param_handler("steps")
def handle_steps_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle steps parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Convert value to int
    steps_value = int(value)

    # Find the correct sampler node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "KSampler" or node["class_type"] == "KSamplerAdvanced":
            node["inputs"]["steps"] = steps_value
            logger.debug(f"Set steps to {steps_value}")
            break

    return workflow_copy


@register_param_handler("lora_weight")
def handle_lora_weight_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle LoRA weight parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Convert value to float
    weight_value = float(value)
    
    # Check if we're varying both loras and lora_weight
    varying_both = args.x_param == "loras" and args.y_param == "lora_weight" or \
                   args.y_param == "loras" and args.x_param == "lora_weight"
    
    # If we're varying both, we need special handling
    if varying_both:
        # The loras parameter handler will handle setting up the nodes and connections
        # Let's store the desired weight in args so that loras handler can use it
        args._current_lora_weight = weight_value
        logger.debug(f"Setting _current_lora_weight to {weight_value} for combined loras+weight variation")
        return workflow_copy
    
    # Normal case: find all LoraLoader nodes and set their weights
    lora_nodes_found = False
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "LoraLoader":
            node["inputs"]["strength_model"] = weight_value
            node["inputs"]["strength_clip"] = weight_value
            logger.debug(f"Set LoRA weight to {weight_value} in node {node_id}")
            lora_nodes_found = True
    
    if not lora_nodes_found:
        logger.warning(f"No LoraLoader nodes found to set weight {weight_value}")
        
    return workflow_copy


@register_param_handler("loras")
def handle_loras_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle multiple LoRAs parameter variation
    
    Value format: "lora1:weight1,lora2:weight2,lora3:weight3"
    Example: "pony/by_wolfy-nail-v3s3000.safetensors:0.35,pony/cotw-v1s400.safetensors:0.4"
    
    If no weight is specified, default to 0.35: 
    "pony/by_wolfy-nail-v3s3000.safetensors,pony/cotw-v1s400.safetensors:0.4"
    """
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Check if we have a common weight to apply from lora_weight parameter
    global_weight = getattr(args, '_current_lora_weight', None)
    varying_both = (args.x_param == "loras" and args.y_param == "lora_weight") or \
                   (args.y_param == "loras" and args.x_param == "lora_weight")
    
    if varying_both and global_weight is not None:
        logger.debug(f"Using global weight {global_weight} from lora_weight parameter")
    
    # Parse the comma-separated lora:weight pairs
    lora_entries = []
    
    # Handle empty case
    if not value or value.lower() == "none":
        loras = []
        lora_weights = []
    else:
        # Split by comma
        pairs = value.split(",")
        loras = []
        lora_weights = []
        
        for pair in pairs:
            # Check if pair contains weight
            if ":" in pair:
                lora_name, weight_str = pair.split(":", 1)
                loras.append(lora_name.strip())
                try:
                    # If we're varying both loras and lora_weight, and this weight
                    # is explicitly set, we'll keep it. Otherwise, use the global weight.
                    if varying_both and global_weight is not None:
                        # Log the choice between explicit and global weight
                        logger.debug(f"Using explicit weight {weight_str} for LoRA {lora_name} over global weight {global_weight}")
                    weight = float(weight_str.strip())
                    lora_weights.append(weight)
                except ValueError:
                    logger.warning(f"Invalid weight '{weight_str}' for LoRA '{lora_name}', using default 0.35")
                    # Use global weight if available, otherwise default
                    weight = global_weight if varying_both and global_weight is not None else 0.35
                    lora_weights.append(weight)
            else:
                # No weight specified, use global weight if available or default 0.35
                loras.append(pair.strip())
                weight = global_weight if varying_both and global_weight is not None else 0.35
                lora_weights.append(weight)
    
    logger.debug(f"Processed LoRAs: {loras} with weights: {lora_weights}")
    
    # No LoRAs to add
    if not loras:
        logger.warning("No LoRAs specified, using base model only")
        return workflow_copy
    
    # Find the checkpoint loader node to get the original model and clip outputs
    checkpoint_node_id = None
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "CheckpointLoaderSimple":
            checkpoint_node_id = node_id
            break
    
    if checkpoint_node_id is None:
        logger.error("Could not find CheckpointLoaderSimple node in workflow")
        return workflow_copy
    
    # Initialize model and clip outputs from the checkpoint
    model_out = [checkpoint_node_id, 0]  # [node_id, output_index]
    clip_out = [checkpoint_node_id, 1]
    
    # Track nodes we've already processed to avoid duplicating changes
    processed_nodes = set()
    
    # First check if any patch nodes like DeepShrink were applied to the model
    # We need to find the latest model modification in the chain
    for node_id, node in workflow_copy.items():
        if node["class_type"] in ["PatchModelAddDownscale_v2", "PerturbedAttention", "ModelSamplingDiscrete"]:
            # Check if this node gets its input directly or indirectly from the checkpoint
            input_source = None
            if "model" in node["inputs"] and isinstance(node["inputs"]["model"], list):
                input_source = node["inputs"]["model"][0]
            
            # If it's connected to the checkpoint or a previously processed node
            if input_source == checkpoint_node_id or input_source in processed_nodes:
                # Update model output to this node
                model_out = [node_id, 0]
                processed_nodes.add(node_id)
    
    logger.debug(f"Starting with model output from node {model_out[0]} (after model patches)")
    
    # Find and remove existing LoRA nodes
    lora_node_ids = []
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "LoraLoader":
            lora_node_ids.append(node_id)
    
    # If there are existing LoRA nodes, remove their references
    if lora_node_ids:
        logger.debug(f"Removing {len(lora_node_ids)} existing LoRA nodes: {lora_node_ids}")
    
    # Create new LoRA nodes
    next_node_id = max([int(nid) for nid in workflow_copy.keys()]) + 1
    lora_nodes_added = []
    
    # Add each LoRA in sequence
    for i, (lora_name, weight) in enumerate(zip(loras, lora_weights)):
        if not lora_name:
            continue
            
        # Create new node ID for this LoRA
        new_node_id = str(next_node_id + i)
        
        # Add LoRA node
        workflow_copy[new_node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_out,
                "clip": clip_out,
                "lora_name": lora_name,
                "strength_model": weight,
                "strength_clip": weight
            }
        }
        
        # Update model and clip outputs for the next node in the chain
        model_out = [new_node_id, 0]
        clip_out = [new_node_id, 1]
        lora_nodes_added.append(new_node_id)
        
        logger.debug(f"Added LoraLoader node {new_node_id} for {lora_name} with weight {weight}")
    
    if lora_nodes_added:
        logger.debug(f"Added {len(lora_nodes_added)} LoRA nodes: {lora_nodes_added}")
        logger.debug(f"Final model output from node {model_out[0]} and clip output from node {clip_out[0]}")
    
    # Now update all node connections to use the new model and clip outputs
    nodes_updated = 0
    
    # First identify the important nodes that need the model or clip connections
    updated_nodes = set()
    for node_id, node in workflow_copy.items():
        # Skip the nodes we just added
        if node_id in lora_nodes_added:
            continue
            
        # Process CLIPTextEncode, KSampler, and CFGGuider nodes
        if node["class_type"] in ["CLIPTextEncode", "KSampler", "KSamplerAdvanced", "CFGGuider"]:
            # Update clip connections for CLIPTextEncode
            if node["class_type"] == "CLIPTextEncode" and "clip" in node["inputs"]:
                # Only update if it was previously using the original clip or a removed LoRA
                if isinstance(node["inputs"]["clip"], list) and (
                    node["inputs"]["clip"][0] == checkpoint_node_id or 
                    node["inputs"]["clip"][0] in lora_node_ids
                ):
                    node["inputs"]["clip"] = clip_out
                    updated_nodes.add(node_id)
                    nodes_updated += 1
                    
            # Update model connections for other nodes
            if "model" in node["inputs"] and isinstance(node["inputs"]["model"], list):
                # Only update if it was previously using the original model or a removed LoRA
                if (node["inputs"]["model"][0] == checkpoint_node_id or 
                    node["inputs"]["model"][0] in lora_node_ids or
                    node["inputs"]["model"][0] in processed_nodes):
                    node["inputs"]["model"] = model_out
                    updated_nodes.add(node_id)
                    nodes_updated += 1
    
    logger.debug(f"Updated {nodes_updated} node connections to use the new LoRA outputs")
    
    # Check if any SamplerCustomAdvanced node is using a CFGGuider that we updated
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "SamplerCustomAdvanced" and "guider" in node["inputs"]:
            guider = node["inputs"]["guider"]
            if isinstance(guider, list) and guider[0] in updated_nodes:
                logger.debug(f"SamplerCustomAdvanced node {node_id} is already using updated CFGGuider {guider[0]}")
    
    return workflow_copy


@register_param_handler("seed")
def handle_seed_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle seed parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Convert value to int
    try:
        seed_value = int(value)
        logger.debug(f"Converting seed value '{value}' to integer: {seed_value}")
    except ValueError:
        logger.error(f"Failed to convert seed value '{value}' to integer, using 0")
        seed_value = 0

    # Find sampler nodes and RandomNoise nodes to set seed
    seed_set = False
    
    # First find and set KSampler/KSamplerAdvanced nodes
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "KSampler" or node["class_type"] == "KSamplerAdvanced":
            node["inputs"]["seed"] = seed_value
            logger.debug(f"Set seed to {seed_value} in KSampler node {node_id}")
            seed_set = True
            # Don't break here - set it for all sampler nodes
            
    # Also find and set RandomNoise nodes
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "RandomNoise":
            node["inputs"]["noise_seed"] = seed_value
            logger.debug(f"Set noise_seed to {seed_value} in RandomNoise node {node_id}")
            seed_set = True
            # Don't break here - set it for all RandomNoise nodes

    if not seed_set:
        logger.warning(f"Could not find any KSampler, KSamplerAdvanced, or RandomNoise nodes to set seed={seed_value}")

    return workflow_copy


@register_param_handler("prompt_variation")
def handle_prompt_variation_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle prompt variation parameter"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Get base prompt
    base_prompt = args.prompt or "a character"

    # Create new prompt with the variation
    new_prompt = f"{base_prompt}, {value}"

    # Find the correct prompt node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "CLIPTextEncode" and "positive" in node_id.lower():
            node["inputs"]["text"] = new_prompt
            logger.debug(f"Set prompt to '{new_prompt}'")
            break

    return workflow_copy


@register_param_handler("split_sigmas")
def handle_split_sigmas_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle split-sigma parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Try to extract numeric value from more complex text
    try:
        # Direct conversion first
        split_value = float(value)
    except ValueError:
        # First look for any number in parentheses
        numeric_match = re.search(r'\((\d+\.?\d*)\)', value)
        if numeric_match:
            split_value = float(numeric_match.group(1))
        else:
            # If not found in parentheses, look for any number in the string
            numeric_match = re.search(r'(\d+\.?\d*)', value)
            if numeric_match:
                split_value = float(numeric_match.group(1))
            else:
                # Default to 0 if no numeric value found
                logger.warning(f"Could not extract numeric value from '{value}', defaulting to 0")
                split_value = 0.0
    
    logger.debug(f"Extracted split-sigma value: {split_value} from '{value}'")
    
    # For split-sigma workflow, we need to set the split-sigmas parameter
    # This will be applied by the workflow creation function, so we just
    # need to update the args
    args_copy = args
    args_copy.split_sigmas = split_value
    logger.debug(f"Set split-sigmas to {split_value}")
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("detail_daemon")
def handle_detail_daemon_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle detail-daemon parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to extract boolean value from more complex text
    if isinstance(value, bool):
        detail_daemon_value = value
    else:
        # Convert various forms of text to boolean
        value_lower = value.lower()
        if any(truth in value_lower for truth in ["true", "yes", "on", "enabled", "1"]):
            detail_daemon_value = True
        elif any(false in value_lower for false in ["false", "no", "off", "disabled", "0"]):
            detail_daemon_value = False
        else:
            # Default to False if can't determine
            logger.warning(f"Could not determine boolean value from '{value}', defaulting to False")
            detail_daemon_value = False
    
    logger.debug(f"Extracted detail-daemon value: {detail_daemon_value} from '{value}'")
    
    # For detail-daemon workflow, we need to set the detail-daemon parameter
    # This will be applied by the workflow creation function, so we just
    # need to update the args
    args_copy = args
    args_copy.detail_daemon = detail_daemon_value
    logger.debug(f"Set detail-daemon to {detail_daemon_value}")
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("detail_amount")
def handle_detail_amount_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle detail amount parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        detail_amount = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(\d+\.?\d*)', value)
        if numeric_match:
            detail_amount = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to 0.1")
            detail_amount = 0.1
    
    logger.debug(f"Extracted detail amount: {detail_amount} from '{value}'")
    
    # Ensure detail_daemon is enabled
    args_copy = args
    args_copy.detail_daemon = True
    args_copy.detail_amount = detail_amount
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("detail_start")
def handle_detail_start_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle detail start parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        detail_start = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(\d+\.?\d*)', value)
        if numeric_match:
            detail_start = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to 0.5")
            detail_start = 0.5
    
    logger.debug(f"Extracted detail start: {detail_start} from '{value}'")
    
    # Ensure detail_daemon is enabled
    args_copy = args
    args_copy.detail_daemon = True
    args_copy.detail_start = detail_start
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("detail_end")
def handle_detail_end_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle detail end parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        detail_end = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(\d+\.?\d*)', value)
        if numeric_match:
            detail_end = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to 0.8")
            detail_end = 0.8
    
    logger.debug(f"Extracted detail end: {detail_end} from '{value}'")
    
    # Ensure detail_daemon is enabled
    args_copy = args
    args_copy.detail_daemon = True
    args_copy.detail_end = detail_end
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("pag")
def handle_pag_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle PAG parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to extract boolean value from more complex text
    if isinstance(value, bool):
        pag_value = value
    else:
        # Convert various forms of text to boolean
        value_lower = value.lower()
        if any(truth in value_lower for truth in ["true", "yes", "on", "enabled", "1"]):
            pag_value = True
        elif any(false in value_lower for false in ["false", "no", "off", "disabled", "0"]):
            pag_value = False
        else:
            # Default to False if can't determine
            logger.warning(f"Could not determine boolean value from '{value}', defaulting to False")
            pag_value = False
    
    logger.debug(f"Extracted PAG value: {pag_value} from '{value}'")
    
    # Update args
    args_copy = args
    args_copy.pag = pag_value
    logger.debug(f"Set pag to {pag_value}")
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("pag_scale")
def handle_pag_scale_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle PAG scale parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        pag_scale = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(\d+\.?\d*)', value)
        if numeric_match:
            pag_scale = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to 3.0")
            pag_scale = 3.0
    
    logger.debug(f"Extracted PAG scale: {pag_scale} from '{value}'")
    
    # Ensure PAG is enabled
    args_copy = args
    args_copy.pag = True
    args_copy.pag_scale = pag_scale
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("pag_sigma_start")
def handle_pag_sigma_start_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle PAG sigma start parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        pag_sigma_start = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(-?\d+\.?\d*)', value)
        if numeric_match:
            pag_sigma_start = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to -1.0 (auto)")
            pag_sigma_start = -1.0
    
    logger.debug(f"Extracted PAG sigma start: {pag_sigma_start} from '{value}'")
    
    # Ensure PAG is enabled
    args_copy = args
    args_copy.pag = True
    args_copy.pag_sigma_start = pag_sigma_start
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("pag_sigma_end")
def handle_pag_sigma_end_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle PAG sigma end parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        pag_sigma_end = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(-?\d+\.?\d*)', value)
        if numeric_match:
            pag_sigma_end = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to -1.0 (auto)")
            pag_sigma_end = -1.0
    
    logger.debug(f"Extracted PAG sigma end: {pag_sigma_end} from '{value}'")
    
    # Ensure PAG is enabled
    args_copy = args
    args_copy.pag = True
    args_copy.pag_sigma_end = pag_sigma_end
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("deepshrink")
def handle_deepshrink_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle DeepShrink parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to extract boolean value from more complex text
    if isinstance(value, bool):
        deepshrink_value = value
    else:
        # Convert various forms of text to boolean
        value_lower = value.lower()
        if any(truth in value_lower for truth in ["true", "yes", "on", "enabled", "1"]):
            deepshrink_value = True
        elif any(false in value_lower for false in ["false", "no", "off", "disabled", "0"]):
            deepshrink_value = False
        else:
            # Default to False if can't determine
            logger.warning(f"Could not determine boolean value from '{value}', defaulting to False")
            deepshrink_value = False
    
    logger.debug(f"Extracted DeepShrink value: {deepshrink_value} from '{value}'")
    
    # Update args
    args_copy = args
    args_copy.deepshrink = deepshrink_value
    logger.debug(f"Set deepshrink to {deepshrink_value}")
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("deepshrink_factor")
def handle_deepshrink_factor_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle DeepShrink factor parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        deepshrink_factor = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(\d+\.?\d*)', value)
        if numeric_match:
            deepshrink_factor = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to 2.0")
            deepshrink_factor = 2.0
    
    logger.debug(f"Extracted DeepShrink factor: {deepshrink_factor} from '{value}'")
    
    # Ensure DeepShrink is enabled
    args_copy = args
    args_copy.deepshrink = True
    args_copy.deepshrink_factor = deepshrink_factor
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("deepshrink_start")
def handle_deepshrink_start_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle DeepShrink start parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        deepshrink_start = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(\d+\.?\d*)', value)
        if numeric_match:
            deepshrink_start = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to 0.0")
            deepshrink_start = 0.0
    
    logger.debug(f"Extracted DeepShrink start: {deepshrink_start} from '{value}'")
    
    # Ensure DeepShrink is enabled
    args_copy = args
    args_copy.deepshrink = True
    args_copy.deepshrink_start = deepshrink_start
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("deepshrink_end")
def handle_deepshrink_end_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle DeepShrink end parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        deepshrink_end = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(\d+\.?\d*)', value)
        if numeric_match:
            deepshrink_end = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to 0.35")
            deepshrink_end = 0.35
    
    logger.debug(f"Extracted DeepShrink end: {deepshrink_end} from '{value}'")
    
    # Ensure DeepShrink is enabled
    args_copy = args
    args_copy.deepshrink = True
    args_copy.deepshrink_end = deepshrink_end
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("deepshrink_gradual")
def handle_deepshrink_gradual_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle DeepShrink gradual parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Try to convert value to float
    try:
        deepshrink_gradual = float(value)
    except ValueError:
        # Extract any numeric value from the string
        numeric_match = re.search(r'(\d+\.?\d*)', value)
        if numeric_match:
            deepshrink_gradual = float(numeric_match.group(1))
        else:
            logger.warning(f"Could not extract numeric value from '{value}', defaulting to 0.6")
            deepshrink_gradual = 0.6
    
    logger.debug(f"Extracted DeepShrink gradual: {deepshrink_gradual} from '{value}'")
    
    # Ensure DeepShrink is enabled
    args_copy = args
    args_copy.deepshrink = True
    args_copy.deepshrink_gradual = deepshrink_gradual
    
    # Re-create the workflow with the updated args
    workflow_creator = get_workflow_template(args.workflow)
    if workflow_creator:
        return workflow_creator(args_copy)
    else:
        return workflow_copy


@register_param_handler("detail_exponent")
def handle_detail_exponent_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle detail_exponent parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Convert value to float
    try:
        detail_exponent = float(value)
        logger.info(f"Setting detail_exponent to {detail_exponent}")
    except ValueError:
        logger.error(f"Failed to convert detail_exponent value '{value}' to float, using 1.5")
        detail_exponent = 1.5
    
    # First make sure detail_daemon is enabled
    # Find DetailDaemonSamplerNode nodes and set their exponent
    daemon_nodes_found = False
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "DetailDaemonSamplerNode":
            node["inputs"]["exponent"] = detail_exponent
            daemon_nodes_found = True
            logger.debug(f"Set exponent to {detail_exponent} in DetailDaemonSamplerNode {node_id}")
    
    if not daemon_nodes_found:
        logger.warning(f"No DetailDaemonSamplerNode found, detail_exponent={detail_exponent} not applied")
    
    return workflow_copy


def dump_workflow_to_file(workflow: Dict[str, Any], filename: str):
    """Dump a workflow to a file for debugging.
    
    Args:
        workflow: The workflow to dump
        filename: The filename to use
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(workflow, f, indent=4)


def generate_single_image(
    workflow: Dict[str, Any], x_param: str, x_value: str, y_param: str, y_value: str, args
) -> str:
    """Generate a single image for the XY plot.

    Args:
        workflow: The base workflow to modify
        x_param: The X parameter being varied
        x_value: The value for the X parameter
        y_param: The Y parameter being varied
        y_value: The value for the Y parameter
        args: The parsed arguments

    Returns:
        Path to the generated image file
    """
    # Sanitize values for file paths
    x_value_safe = sanitize_path(x_value)
    y_value_safe = sanitize_path(y_value)
    
    # Log what we're generating
    logger.info(f"Generating image {args.index_count}/{args.total_count}: {x_param}={x_value}, {y_param}={y_value}")

    # Create a copy of the workflow for modification
    modified_workflow = json.loads(json.dumps(workflow))

    # Apply the X parameter handler
    if x_param in param_handlers:
        modified_workflow = param_handlers[x_param](modified_workflow, x_value, args)
    else:
        logger.warning(f"No handler for X parameter: {x_param}")

    # Apply the Y parameter handler
    if y_param in param_handlers:
        modified_workflow = param_handlers[y_param](modified_workflow, y_value, args)
    else:
        logger.warning(f"No handler for Y parameter: {y_param}")
        
    # Add a unique image name to the SaveImage node to prevent caching/reuse
    unique_prefix = f"xy_{x_param}_{x_value_safe}_{y_param}_{y_value_safe}_{int(time.time())}"
    for node_id, node in modified_workflow.items():
        if node["class_type"] == "SaveImage":
            # Set a unique filename_prefix based on the current parameters
            node["inputs"]["filename_prefix"] = unique_prefix
            logger.debug(f"Set unique filename_prefix to {unique_prefix} in SaveImage node")

    # Dump the workflow if requested
    if args.dump_workflows:
        debug_dir = os.path.join(args.output_dir, "debug")
        debug_filename = os.path.join(
            debug_dir, f"workflow_{x_param}_{x_value_safe}_{y_param}_{y_value_safe}.json"
        )
        dump_workflow_to_file(modified_workflow, debug_filename)

    # Determine the output filename - ensure we don't have duplicate extensions
    output_filename = os.path.join(
        args.output_dir, f"xy_{x_value_safe}_{y_value_safe}.png"
    )

    # Create a blank image in case generation fails
    blank_filename = os.path.join(
        args.output_dir, f"blank_{x_value_safe}_{y_value_safe}.png"
    )

    try:
        # Queue the prompt for generation
        prompt_result = queue_prompt(modified_workflow)
        
        if 'prompt_id' not in prompt_result:
            logger.error(f"Failed to queue prompt, no prompt_id returned: {prompt_result}")
            return blank_filename
            
        prompt_id = prompt_result['prompt_id']
        logger.info(f"Queued prompt with ID: {prompt_id}")
        
        # Track the generation status with a more responsive approach
        start_time = time.time()
        max_wait_time = 300  # 5 minutes max wait
        last_progress = 0
        last_progress_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check the current status
            status_result = check_generation_status(prompt_id)
            
            current_status = status_result.get('status', 'unknown')
            current_progress = status_result.get('progress', 0.0)
            
            # Report progress if it changed significantly (at least 10%)
            if current_progress >= last_progress + 0.1:
                logger.info(f"Progress for {x_param}={x_value}, {y_param}={y_value}: {current_progress:.0%}")
                last_progress = current_progress
                last_progress_time = time.time()
            
            # If completed, we can proceed
            if current_status == 'completed':
                logger.info(f"Generation completed for {x_param}={x_value}, {y_param}={y_value} in {time.time() - start_time:.1f}s")
                break
                
            # If it's taking too long without progress, log a warning but keep waiting
            if time.time() - last_progress_time > 60:  # No progress for 1 minute
                logger.warning(f"No progress update for 1 minute (current: {current_progress:.0%})")
                last_progress_time = time.time()  # Reset to avoid spamming logs
            
            # If there's an error, report it and stop waiting
            if current_status == 'error':
                logger.error(f"Error in generation: {status_result.get('error', 'Unknown error')}")
                return blank_filename
                
            # Short sleep to avoid hammering the server
            time.sleep(1 if current_status == 'pending' else 0.5)
            
        # If we exited the loop due to timeout
        if time.time() - start_time >= max_wait_time:
            logger.warning(f"Timed out waiting for generation after {max_wait_time}s")
            return blank_filename

        # Get the path to the generated image - with a slight delay to ensure the image is available
        time.sleep(1)  # Short delay to ensure ComfyUI has finished writing the image
        image_paths = get_image_path(prompt_id)
        if not image_paths:
            logger.warning(f"Failed to find generated image for {x_param}={x_value}, {y_param}={y_value}")
            # Try to find image by prefix instead (in case get_image_path fails)
            time.sleep(1)  # Give it a little more time
            if args.remote:
                # For remote sessions, use SSH to list files that match our prefix
                ssh_cmd = ["ssh", "-p", str(args.ssh_port), f"{args.ssh_user}@{args.ssh_host}",
                          f"find {args.comfy_output_dir} -name '{unique_prefix}*' -type f -print -quit"]
                try:
                    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        image_paths = [os.path.basename(result.stdout.strip())]
                        logger.info(f"Found image by unique prefix: {image_paths[0]}")
                    else:
                        logger.warning(f"Failed to find image with prefix {unique_prefix}")
                        return blank_filename
                except Exception as e:
                    logger.error(f"Error finding image by prefix: {e}")
                    return blank_filename
            else:
                # For local sessions, try to find the file directly
                try:
                    matching_files = list(pathlib.Path(args.comfy_output_dir).glob(f"{unique_prefix}*"))
                    if matching_files:
                        image_paths = [os.path.basename(str(matching_files[0]))]
                        logger.info(f"Found image by unique prefix: {image_paths[0]}")
                    else:
                        logger.warning(f"Failed to find image with prefix {unique_prefix}")
                        return blank_filename
                except Exception as e:
                    logger.error(f"Error finding image by prefix: {e}")
                    return blank_filename
            
        # Use the first image path returned
        image_path = image_paths[0] if isinstance(image_paths, list) else image_paths

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Get the image from ComfyUI (locally or remotely)
        if args.remote:
            # Extract the basename without extension for cleaner output
            basename, _ = os.path.splitext(os.path.basename(output_filename))
            
            # Use rsync to get the image from the remote server
            dest_path = rsync_image_from_comfyui(
                image_path, 
                args.ssh_host,
                args.comfy_output_dir, 
                os.path.dirname(output_filename),
                basename,  # Use the basename without extension as a prefix
                args.ssh_port, 
                args.ssh_user, 
                getattr(args, 'ssh_key', None)  # Handle missing ssh_key attribute
            )
            
            if not dest_path:
                logger.warning(f"Failed to rsync image from remote server")
                return blank_filename
                
            # Update output_filename to the actual path returned by rsync
            output_filename = dest_path
        else:
            # Copy the image from the local ComfyUI server
            copy_image_from_comfyui(image_path, output_filename, args.comfy_output_dir)

        # Check if the image was successfully copied
        if os.path.exists(output_filename):
            logger.info(f"Successfully saved image to {output_filename}")
            return output_filename
        else:
            logger.warning(f"Failed to save image to {output_filename}")
            return blank_filename

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return blank_filename


def create_xy_grid(
    image_paths: List[str],
    x_values: List[str],
    y_values: List[str],
    output_path: str,
    x_label: str,
    y_label: str,
    label_alignment: str = "center",
    debug_mode: bool = False,
    font_size: float = 40.0,
    horizontal_spacing: int = 0,
    vertical_spacing: int = 0,
) -> str:
    """Create an XY grid using the imx Rust library

    Args:
        image_paths: List of paths to images (row-major order)
        x_values: List of X axis values
        y_values: List of Y axis values
        output_path: Path to save the output grid
        x_label: Label for X axis
        y_label: Label for Y axis
        label_alignment: Alignment for labels (start, center, end)
        debug_mode: Enable debug visualization
        font_size: Font size for labels (ignored in current imx version)
        horizontal_spacing: Horizontal spacing between images in pixels
        vertical_spacing: Vertical spacing between images in pixels

    Returns:
        Path to the output grid image
    """
    # Calculate padding based on label length
    # For longer labels, allocate more space
    max_x_label_len = max(len(str(x)) for x in x_values)
    max_y_label_len = max(len(str(y)) for y in y_values)
    
    top_padding = 160  # Base value
    left_padding = 120  # Base value
    
    # Increase padding for longer labels
    if max_x_label_len > 15:
        top_padding += (max_x_label_len - 15) * 3
    if max_y_label_len > 15:
        left_padding += (max_y_label_len - 15) * 3
    
    # Create a temporary JSON config file for imx
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
        config = {
            "images": image_paths,
            "output": output_path,
            "rows": len(y_values),
            "row_labels": [f"{y_label}: {y}" for y in y_values],
            "column_labels": [f"{x_label}: {x}" for x in x_values],
            "column_label_alignment": label_alignment,
            "row_label_alignment": label_alignment,
            "top_padding": top_padding,
            "left_padding": left_padding,
            "font_size": font_size,  # Now should be used by updated imx library
            "horizontal_spacing": horizontal_spacing,
            "vertical_spacing": vertical_spacing,
            "debug_mode": debug_mode,
        }
        json.dump(config, temp)
        config_path = temp.name

    try:
        # Get the absolute path to imx-plot in the project root
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        imx_plot_path = os.path.join(script_dir, "imx-plot")

        # Check if imx-plot exists
        if not os.path.exists(imx_plot_path):
            logger.warning(
                f"imx-plot script not found at {imx_plot_path}, falling back to ImageMagick"
            )
            return create_grid_with_imagemagick(
                image_paths, x_values, y_values, output_path, x_label, y_label
            )

        # Call the imx binary
        logger.info(f"Running imx-plot: {imx_plot_path} {config_path}")
        cmd = [imx_plot_path, config_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"Failed to create grid with imx-plot: {result.stderr}")
            logger.info("Falling back to ImageMagick...")
            return create_grid_with_imagemagick(
                image_paths, x_values, y_values, output_path, x_label, y_label
            )

        logger.info(f"Successfully created XY grid at {output_path}")
        if debug_mode:
            logger.info(
                f"Debug visualization saved to {os.path.splitext(output_path)[0]}_debug{os.path.splitext(output_path)[1]}"
            )

        # Clean up the temporary file
        os.unlink(config_path)

        return output_path
    except Exception as e:
        logger.error(f"Error creating XY grid with imx-plot: {e}")
        logger.info("Falling back to ImageMagick...")
        try:
            return create_grid_with_imagemagick(
                image_paths, x_values, y_values, output_path, x_label, y_label
            )
        except Exception as e2:
            logger.error(f"Error creating grid with ImageMagick: {e2}")
            return None


def create_grid_with_imagemagick(
    image_paths: List[str],
    x_values: List[str],
    y_values: List[str],
    output_path: str,
    x_label: str,
    y_label: str,
) -> str:
    """Create an XY grid using ImageMagick's montage command as a fallback

    Args:
        image_paths: List of paths to images (row-major order)
        x_values: List of X axis values
        y_values: List of Y axis values
        output_path: Path to save the output grid
        x_label: Label for X axis
        y_label: Label for Y axis

    Returns:
        Path to the output grid image
    """
    logger.info("Creating grid with ImageMagick")

    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Organize images by row
        rows = []
        for y_idx in range(len(y_values)):
            row_images = image_paths[y_idx * len(x_values) : (y_idx + 1) * len(x_values)]
            rows.append(row_images)

        # Create a labeled image for each cell
        labeled_images = []
        for y_idx, row in enumerate(rows):
            for x_idx, img_path in enumerate(row):
                # Create labeled version
                labeled_path = os.path.join(temp_dir, f"labeled_{y_idx}_{x_idx}.png")
                label_text = f"{x_label}: {x_values[x_idx]}, {y_label}: {y_values[y_idx]}"

                # Create labeled image with ImageMagick
                cmd = [
                    "convert",
                    img_path,
                    "-gravity",
                    "North",
                    "-background",
                    "white",
                    "-splice",
                    "0x30",
                    "-pointsize",
                    "14",
                    "-annotate",
                    "+0+5",
                    label_text,
                    labeled_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to add label to image: {result.stderr}")
                    labeled_images.append(img_path)  # Use original if labeling fails
                else:
                    labeled_images.append(labeled_path)

        # Create the grid with montage
        cols = len(x_values)
        cmd = [
            "montage",
            *labeled_images,
            "-tile",
            f"{cols}x",
            "-geometry",
            "+5+5",
            "-background",
            "white",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to create grid with montage: {result.stderr}")
            return None

        logger.info(f"Successfully created XY grid with ImageMagick at {output_path}")
        return output_path


def sanitize_path(path: str) -> str:
    """Sanitize a path for use in filenames.
    
    Args:
        path: The path to sanitize
        
    Returns:
        Sanitized path
    """
    # Replace characters that cause issues in paths
    path = path.replace('/', '_')
    path = path.replace('\\', '_')
    path = path.replace(':', '_')
    path = path.replace(';', '_')
    path = path.replace(',', '_')
    path = path.replace(' ', '_')
    path = path.replace('\t', '_')
    path = path.replace('\n', '_')
    path = path.replace('\r', '_')
    
    # Replace multiple underscores with a single one
    while '__' in path:
        path = path.replace('__', '_')
        
    return path


def generate_xyplot(args):
    """Generate an XY plot varying two parameters.

    Args:
        args: Command line arguments
    """
    # Store the command for metadata
    try:
        command = f"cringegen xyplot"
        # Add all arguments to the command
        for arg_name, arg_value in vars(args).items():
            # Skip function and private attributes
            if arg_name == 'func' or arg_name.startswith('_'):
                continue
            # Skip default comfy_url and comfy_output_dir (typically long and not useful for reproducibility)
            if arg_name in ['comfy_url', 'comfy_output_dir'] and arg_value is not None:
                continue
            
            if arg_value is True:
                command += f" --{arg_name.replace('_', '-')}"
            elif arg_value is False or arg_value is None:
                continue
            else:
                # Quote values with spaces
                if isinstance(arg_value, str) and ' ' in arg_value:
                    command += f" --{arg_name.replace('_', '-')} \"{arg_value}\""
                else:
                    command += f" --{arg_name.replace('_', '-')} {arg_value}"
        
        logger.info(f"Command to be saved in metadata: {command}")
    except Exception as e:
        logger.warning(f"Error reconstructing command for metadata: {e}")
        command = "cringegen xyplot (command reconstruction failed)"

    # Check if ComfyUI server is running
    if not check_comfy_server(args.comfy_url):
        logger.error("ComfyUI server is not running or not accessible")
        return

    # Ensure output directory exists
    ensure_dir_exists(args.output_dir)

    # Extract X and Y values from comma-separated or semicolon-separated strings
    x_values = args.x_values.replace(";", ",").split(",")
    y_values = args.y_values.replace(";", ",").split(",")

    # Clean up values (remove leading/trailing whitespace)
    x_values = [x.strip() for x in x_values]
    y_values = [y.strip() for y in y_values]

    # Remove any empty values
    x_values = [x for x in x_values if x]
    y_values = [y for y in y_values if y]

    logger.info(f"Generating XY plot with {len(x_values)}x{len(y_values)} grid")
    logger.info(f"X axis ({args.x_param}): {x_values}")
    logger.info(f"Y axis ({args.y_param}): {y_values}")

    # Get base workflow based on workflow type
    workflow_creator = get_workflow_template(args.workflow)
    if not workflow_creator:
        logger.error(f"Unknown workflow type: {args.workflow}")
        return

    # Create base workflow
    workflow = workflow_creator(args)

    # Set initial parameters in the workflow if provided
    if args.checkpoint:
        workflow = handle_checkpoint_param(workflow, args.checkpoint, args)

    if args.prompt:
        for node_id, node in workflow.items():
            if node["class_type"] == "CLIPTextEncode" and "positive" in node_id.lower():
                node["inputs"]["text"] = args.prompt
                break

    if args.negative_prompt:
        for node_id, node in workflow.items():
            if node["class_type"] == "CLIPTextEncode" and "negative" in node_id.lower():
                node["inputs"]["text"] = args.negative_prompt
                break

    # Generate images for each parameter combination
    image_paths = []
    
    # Add total count of images to generate for progress tracking
    args.total_count = len(x_values) * len(y_values)
    args.index_count = 0
    
    # Generate images for each position in the grid
    for y_idx, y_value in enumerate(y_values):
        row_images = []
        for x_idx, x_value in enumerate(x_values):
            args.index_count += 1
            
            # Sanitize values for safe filenames
            x_value_safe = sanitize_path(x_value)
            y_value_safe = sanitize_path(y_value)
            
            logger.info(
                f"Generating image {args.index_count}/{args.total_count}: {args.x_param}={x_value}, {args.y_param}={y_value}"
            )

            image_path = generate_single_image(
                workflow, args.x_param, x_value, args.y_param, y_value, args
            )
            if image_path:
                row_images.append(image_path)
            else:
                logger.warning(
                    f"Failed to generate image for {args.x_param}={x_value}, {args.y_param}={y_value}"
                )
                # Add a placeholder or default image
                default_image = os.path.join(
                    os.path.dirname(__file__), "..", "data", "placeholder.png"
                )
                if os.path.exists(default_image):
                    row_images.append(default_image)
                else:
                    # Create a blank image as placeholder
                    blank_path = os.path.join(args.output_dir, f"blank_{x_value_safe}_{y_value_safe}.png")
                    # Create directory if needed
                    os.makedirs(os.path.dirname(blank_path), exist_ok=True)
                    subprocess.run(
                        ["convert", "-size", f"{args.width}x{args.height}", "xc:white", blank_path]
                    )
                    row_images.append(blank_path)

        image_paths.append(row_images)

    # Flatten the image paths list
    flat_image_paths = [path for row in image_paths for path in row]

    # Create the grid
    output_path = os.path.join(args.output_dir, f"{args.output_name}.png")
    
    try:
        # Try to create the grid using imx-plot
        grid_path = create_xy_grid(
            flat_image_paths,
            x_values,
            y_values,
            output_path,
            args.x_param,
            args.y_param,
            args.label_alignment,
            args.debug_mode,
            args.font_size,
            args.horizontal_spacing,
            args.vertical_spacing,
        )
    
        if grid_path:
            logger.info(f"XY plot grid saved to {grid_path}")
            
            # Add command to image metadata
            try:
                # Check if exiftool is available
                exiftool_available = False
                try:
                    result = subprocess.run(["which", "exiftool"], capture_output=True, text=True)
                    exiftool_available = result.returncode == 0
                except Exception:
                    exiftool_available = False
                    
                if exiftool_available:
                    # Add metadata to the image using exiftool
                    metadata_cmd = [
                        "exiftool",
                        "-overwrite_original",
                        f"-Comment={command}",
                        grid_path
                    ]
                    result = subprocess.run(metadata_cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info("Successfully added command to image metadata using exiftool")
                    else:
                        logger.warning(f"Failed to add metadata with exiftool: {result.stderr}")
                        raise Exception("exiftool failed")
                else:
                    # Exiftool not available
                    logger.warning("exiftool not found, falling back to ImageMagick")
                    raise Exception("exiftool not found")
                    
            except Exception as e:
                logger.info(f"Falling back to ImageMagick for metadata: {str(e)}")
                
                # Try using ImageMagick's convert as a fallback
                try:
                    # Check if convert is available
                    convert_available = False
                    try:
                        result = subprocess.run(["which", "convert"], capture_output=True, text=True)
                        convert_available = result.returncode == 0
                    except Exception:
                        convert_available = False
                        
                    if convert_available:
                        # Use ImageMagick to add metadata
                        convert_cmd = [
                            "convert", 
                            grid_path, 
                            "-set", 
                            "comment", 
                            command, 
                            grid_path
                        ]
                        result = subprocess.run(convert_cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            logger.info("Successfully added command to image metadata using ImageMagick")
                        else:
                            logger.warning(f"Failed to add metadata using ImageMagick: {result.stderr}")
                    else:
                        logger.warning("Neither exiftool nor ImageMagick convert are available for adding metadata")
                except Exception as e2:
                    logger.warning(f"Error adding metadata using ImageMagick: {e2}")
            
            # If show option is enabled, open the grid image with imv
            if hasattr(args, "show") and args.show:
                logger.info("Opening XY plot grid with imv")
                open_images_with_imv([grid_path])
            return grid_path
    except Exception as e:
        logger.warning(f"Failed to create grid with imx-plot: {str(e)}")

    # If imx-plot failed, fall back to ImageMagick
    logger.info("Falling back to ImageMagick...")
    try:
        imagemagick_path = create_grid_with_imagemagick(
            flat_image_paths, x_values, y_values, output_path, args.x_param, args.y_param
        )
        if imagemagick_path:
            logger.info(f"Created XY plot grid at {imagemagick_path}")
            return imagemagick_path
    except Exception as e:
        logger.error(f"Failed to create XY plot grid: {str(e)}")
        raise
