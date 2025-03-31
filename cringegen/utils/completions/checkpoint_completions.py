"""
Checkpoint completion utilities for shell auto-completion.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from ...utils.logger import get_logger

logger = get_logger(__name__)


def get_checkpoint_completions(
    partial_name: str = "",
    include_info: bool = True,
    checkpoint_dir: Optional[str] = None,
    max_results: int = 50,
) -> List[Tuple[str, str]]:
    """
    Get checkpoint completions with optional model information.

    Args:
        partial_name: Partial checkpoint name to filter by
        include_info: Whether to include model information
        checkpoint_dir: Directory containing checkpoint files (defaults to ComfyUI checkpoint dir)
        max_results: Maximum number of results to return

    Returns:
        List of tuples (checkpoint_name, description) for shell completion
    """
    # Get the checkpoint directory
    checkpoint_directory = checkpoint_dir or get_checkpoint_directory()

    if not checkpoint_directory or not os.path.isdir(checkpoint_directory):
        logger.warning(f"Checkpoint directory not found: {checkpoint_directory}")
        return []

    completions = []

    try:
        # Get all checkpoint files
        checkpoint_files = []
        for root, _, files in os.walk(checkpoint_directory):
            for file in files:
                # Common checkpoint extensions
                if any(
                    file.endswith(ext) for ext in (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
                ):
                    if partial_name.lower() in file.lower():
                        checkpoint_path = os.path.join(root, file)
                        # Strip file extension for completion
                        checkpoint_name = os.path.splitext(file)[0]
                        checkpoint_files.append((checkpoint_name, checkpoint_path))

        # Sort by name and limit results
        checkpoint_files = sorted(checkpoint_files, key=lambda x: x[0])[:max_results]

        # Add model information if requested
        if include_info:
            checkpoint_info = get_checkpoint_info()
            for checkpoint_name, _ in checkpoint_files:
                info = checkpoint_info.get(checkpoint_name, "Checkpoint")
                completions.append((checkpoint_name, info))
        else:
            completions = [(name, "Checkpoint") for name, _ in checkpoint_files]

    except Exception as e:
        logger.error(f"Error generating checkpoint completions: {e}")

    return completions


def get_checkpoint_directory() -> Optional[str]:
    """
    Get the checkpoint directory from ComfyUI configuration.

    Returns:
        Path to the checkpoint directory or None if not found
    """
    try:
        # Try to get from comfy_api
        try:
            from ...utils.comfy_api import get_comfyui_directory

            comfyui_dir = get_comfyui_directory()
        except ImportError:
            # Fallback to common locations
            common_locations = [
                "~/comfyui",
                "~/stable-diffusion/ComfyUI",
                "~/comfy-ui",
            ]
            for loc in common_locations:
                path = os.path.expanduser(loc)
                if os.path.exists(path):
                    comfyui_dir = path
                    break
            else:
                return None

        # Default ComfyUI checkpoint directory
        checkpoint_dir = os.path.join(comfyui_dir, "models", "checkpoints")

        if os.path.isdir(checkpoint_dir):
            return checkpoint_dir

        # Alternative locations
        alt_locations = [
            os.path.join(comfyui_dir, "models"),
            os.path.join(os.path.dirname(comfyui_dir), "models", "checkpoints"),
        ]

        for location in alt_locations:
            if os.path.isdir(location):
                return location

        return None

    except Exception as e:
        logger.error(f"Error getting checkpoint directory: {e}")
        return None


def get_checkpoint_info(cache_file: Optional[str] = None) -> Dict[str, str]:
    """
    Get information about checkpoints, either from cache or by analyzing files.

    Args:
        cache_file: Path to cache file (defaults to ~/.cache/cringegen/checkpoint_info.json)

    Returns:
        Dictionary mapping checkpoint names to descriptions
    """
    # Default cache location
    if cache_file is None:
        cache_dir = os.path.expanduser("~/.cache/cringegen")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "checkpoint_info.json")

    # Try to load from cache
    checkpoint_info = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                checkpoint_info = json.load(f)
                logger.debug(f"Loaded checkpoint info from cache: {len(checkpoint_info)} entries")
        except Exception as e:
            logger.error(f"Error loading checkpoint info cache: {e}")

    # If cache is empty, generate some basic information
    if not checkpoint_info:
        checkpoint_dir = get_checkpoint_directory()
        if checkpoint_dir:
            checkpoint_info = analyze_checkpoints(checkpoint_dir)

            # Save to cache
            try:
                with open(cache_file, "w") as f:
                    json.dump(checkpoint_info, f)
                    logger.debug(f"Saved checkpoint info to cache: {len(checkpoint_info)} entries")
            except Exception as e:
                logger.error(f"Error saving checkpoint info cache: {e}")

    return checkpoint_info


def analyze_checkpoints(checkpoint_dir: str) -> Dict[str, str]:
    """
    Analyze checkpoint files to extract basic information.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Dictionary mapping checkpoint names to descriptions
    """
    checkpoint_info = {}

    try:
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                if any(
                    file.endswith(ext) for ext in (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
                ):
                    checkpoint_path = os.path.join(root, file)
                    checkpoint_name = os.path.splitext(file)[0]

                    # Extract information from filename pattern
                    description = "Checkpoint"

                    # Look for common patterns in filenames
                    if "xl" in file.lower():
                        description = "SDXL Checkpoint"
                    elif "sd15" in file.lower() or "sd1.5" in file.lower():
                        description = "SD1.5 Checkpoint"
                    elif "sdxl-refiner" in file.lower():
                        description = "SDXL Refiner"
                    elif "inpainting" in file.lower():
                        description = "Inpainting Checkpoint"

                    # Check file size to guess model type
                    try:
                        file_size_gb = os.path.getsize(checkpoint_path) / (1024 * 1024 * 1024)
                        if file_size_gb > 5.0:
                            description = "SDXL Checkpoint"
                        elif 1.5 <= file_size_gb <= 2.5:
                            description = "SD1.5 Checkpoint"
                    except Exception:
                        pass

                    checkpoint_info[checkpoint_name] = description

    except Exception as e:
        logger.error(f"Error analyzing checkpoints: {e}")

    return checkpoint_info
