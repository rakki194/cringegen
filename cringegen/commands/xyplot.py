"""
XY Plot generation commands for CringeGen

This module provides functionality for generating XY plot grids that vary parameters
across X and Y axes to visualize their impact on generated images.
"""

import logging
import os
import tempfile
import json
import subprocess
import time
from typing import List, Dict, Any, Tuple, Optional, Callable

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
            "sampler",
            "scheduler",
            "cfg",
            "steps",
            "lora_weight",
            "seed",
            "prompt_variation",
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
            "sampler",
            "scheduler",
            "cfg",
            "steps",
            "lora_weight",
            "seed",
            "prompt_variation",
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
        "--width", type=int, default=512, help="Width of each individual image"
    )
    xyplot_parser.add_argument(
        "--height", type=int, default=512, help="Height of each individual image"
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

    # Find the correct LoRA node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "LoraLoader":
            node["inputs"]["strength_model"] = weight_value
            node["inputs"]["strength_clip"] = weight_value
            logger.debug(f"Set LoRA weight to {weight_value}")
            break

    return workflow_copy


@register_param_handler("seed")
def handle_seed_param(workflow: Dict[str, Any], value: str, args) -> Dict[str, Any]:
    """Handle seed parameter variation"""
    # Get a copy of the workflow
    workflow_copy = json.loads(json.dumps(workflow))

    # Convert value to int
    seed_value = int(value)

    # Find the correct sampler node
    for node_id, node in workflow_copy.items():
        if node["class_type"] == "KSampler" or node["class_type"] == "KSamplerAdvanced":
            node["inputs"]["seed"] = seed_value
            logger.debug(f"Set seed to {seed_value}")
            break

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
            logger.debug(f"Set prompt variation to: {new_prompt}")
            break

    return workflow_copy


def generate_single_image(
    workflow: Dict[str, Any], x_param: str, x_value: str, y_param: str, y_value: str, args
) -> str:
    """Generate a single image with specified parameters

    Args:
        workflow: Base workflow template
        x_param: X-axis parameter type
        x_value: X-axis parameter value
        y_param: Y-axis parameter type
        y_value: Y-axis parameter value
        args: Command arguments

    Returns:
        Path to the generated image
    """
    # Apply X parameter
    modified_workflow = param_handlers[x_param](workflow, x_value, args)

    # Apply Y parameter (if different from X)
    if y_param != x_param:
        modified_workflow = param_handlers[y_param](modified_workflow, y_value, args)

    # Set image dimensions
    for node_id, node in modified_workflow.items():
        if node["class_type"] == "EmptyLatentImage":
            node["inputs"]["width"] = args.width
            node["inputs"]["height"] = args.height

    # Queue the workflow
    prompt_id_response = queue_prompt(modified_workflow, args.comfy_url)
    if not prompt_id_response:
        logger.error("Failed to queue prompt for XY plot image")
        return None

    # Extract the actual prompt_id string from the response
    if isinstance(prompt_id_response, dict) and "prompt_id" in prompt_id_response:
        prompt_id = prompt_id_response["prompt_id"]
    else:
        prompt_id = str(prompt_id_response)

    # Poll for generation status with a reasonable timeout
    max_poll_time = 300  # 5 minutes max
    poll_interval = 2  # Check every 2 seconds
    start_time = time.time()

    logger.info(f"Monitoring generation status for prompt {prompt_id}")

    # Poll until completed or timeout
    last_progress = 0
    while time.time() - start_time < max_poll_time:
        status = check_generation_status(prompt_id, args.comfy_url)

        # Print progress updates only when progress changes significantly
        if status["status"] == "pending":
            if time.time() - start_time > 10:  # Only log after 10 seconds of waiting
                logger.info(f"Generation pending... ({int(time.time() - start_time)}s elapsed)")
        elif status["status"] == "processing":
            current_progress = int(status["progress"] * 100)
            if current_progress >= last_progress + 10 or (time.time() - start_time) % 10 < 2:
                logger.info(
                    f"Generation in progress: {current_progress}% ({int(time.time() - start_time)}s elapsed)"
                )
                last_progress = current_progress
        elif status["status"] == "completed":
            logger.info(f"Generation completed in {int(time.time() - start_time)}s!")
            break
        elif status["status"] == "error":
            logger.error(f"Generation error: {status['error']}")
            return None

        # Wait before polling again
        time.sleep(poll_interval)

    # Check if we timed out
    if time.time() - start_time >= max_poll_time:
        logger.warning(f"Generation timed out after {max_poll_time}s")
        return None

    # Get the image filename
    if status and status["status"] == "completed" and status["images"]:
        image_filename = status["images"][0]["filename"]
        logger.info(f"Found image: {image_filename}")
    else:
        logger.error("No images were generated")
        return None

    # Copy the image
    if args.remote:
        local_path = rsync_image_from_comfyui(
            image_filename,
            args.ssh_host,
            args.comfy_output_dir,
            args.output_dir,
            f"xy_{x_value}_{y_value}",
            args.ssh_port,
            args.ssh_user,
            args.ssh_key,
        )
    else:
        local_path = copy_image_from_comfyui(
            image_filename, args.comfy_output_dir, args.output_dir, f"xy_{x_value}_{y_value}"
        )

    if not local_path:
        logger.warning("Failed to copy image, trying to copy the latest image instead...")

        # Try with latest images as fallback
        copied = []

        if args.remote:
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
            logger.info(f"Copied most recent image as fallback")
            local_path = copied[0]

    return local_path


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
            "top_padding": 60,
            "left_padding": 80,
            # font_size is ignored in current imx version
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


def generate_xyplot(args):
    """Generate an XY plot with varying parameters

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

    # Split parameter values
    x_values = [x.strip() for x in args.x_values.split(",")]
    y_values = [y.strip() for y in args.y_values.split(",")]

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
    total_images = len(x_values) * len(y_values)
    current_image = 0

    for y_idx, y_value in enumerate(y_values):
        row_images = []
        for x_idx, x_value in enumerate(x_values):
            current_image += 1
            logger.info(
                f"Generating image {current_image}/{total_images}: {args.x_param}={x_value}, {args.y_param}={y_value}"
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
                    blank_path = os.path.join(args.output_dir, f"blank_{x_value}_{y_value}.png")
                    subprocess.run(
                        ["convert", "-size", f"{args.width}x{args.height}", "xc:white", blank_path]
                    )
                    row_images.append(blank_path)

        image_paths.extend(row_images)

    # Create the grid
    output_path = os.path.join(args.output_dir, f"{args.output_name}.png")
    grid_path = create_xy_grid(
        image_paths,
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
            # Add metadata to the image using exiftool
            metadata_cmd = [
                "exiftool",
                "-overwrite_original",
                f"-Comment={command}",
                grid_path
            ]
            result = subprocess.run(metadata_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Successfully added command to image metadata")
            else:
                logger.warning(f"Failed to add metadata to image: {result.stderr}")
                
                # Try using ImageMagick's convert as a fallback
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
        except Exception as e:
            logger.warning(f"Error adding metadata to image: {e}")
        
        # If show option is enabled, open the grid image with imv
        if hasattr(args, "show") and args.show:
            logger.info("Opening XY plot grid with imv")
            open_images_with_imv([grid_path])
    else:
        logger.error("Failed to create XY plot grid")
