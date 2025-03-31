"""
Utilities for file operations in the CringeGen package.
"""

import glob
import os
import shutil
from typing import List, Optional

from .logger import get_logger

# Create logger
logger = get_logger(__name__)


def ensure_dir_exists(path: str) -> None:
    """Ensure that a directory exists, creating it if necessary.

    Args:
        path: Path to the directory to ensure exists
    """
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


def copy_image_from_comfyui(
    image_name: str, source_dir: str, dest_dir: str, output_prefix: str = None
) -> Optional[str]:
    """Copy an image from the ComfyUI output directory to the cringegen output directory.

    Args:
        image_name: Name of the image file or full ComfyUI path
        source_dir: Source directory containing the image
        dest_dir: Destination directory
        output_prefix: Optional prefix for output filename

    Returns:
        Path to the copied image if successful, None otherwise
    """
    # Ensure destination directory exists
    ensure_dir_exists(dest_dir)
    
    # ComfyUI sometimes returns paths in format 'subfolder/filename'
    # Extract subfolder if present
    subfolder = ""
    if '/' in image_name:
        subfolder, image_name = image_name.rsplit('/', 1)
    
    # Build source and destination paths
    source_path = os.path.join(source_dir, subfolder, image_name)
    
    # Create output filename with optional prefix
    if output_prefix:
        filename, ext = os.path.splitext(image_name)
        output_filename = f"{output_prefix}{ext}"
    else:
        output_filename = image_name
    
    dest_path = os.path.join(dest_dir, output_filename)

    # Check if source file exists
    if not os.path.exists(source_path):
        # Try checking if the file is in the root source directory
        alt_source_path = os.path.join(source_dir, image_name)
        if os.path.exists(alt_source_path):
            source_path = alt_source_path
            logger.info(f"Found image at alternate path: {source_path}")
        else:
            logger.error(f"Source file {source_path} does not exist")
            # List available files in source directory
            logger.info(f"Files in source directory {source_dir}:")
            try:
                files = os.listdir(source_dir)
                for file in files[:10]:  # Show only first 10 files
                    logger.info(f"  - {file}")
                if len(files) > 10:
                    logger.info(f"  ... and {len(files) - 10} more files")
            except Exception as e:
                logger.error(f"Error listing source directory: {e}")
            return None

    # Copy the file
    try:
        shutil.copy2(source_path, dest_path)
        logger.info(f"Successfully copied {image_name} to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Error copying file {source_path} to {dest_path}: {e}")
        return None


def copy_latest_images_from_comfyui(
    source_dir: str, dest_dir: str, pattern: str = "*", limit: int = 5
) -> List[str]:
    """Copy the latest images from the ComfyUI output directory to the cringegen output directory.

    Args:
        source_dir: Source directory containing the images
        dest_dir: Destination directory
        pattern: Glob pattern to match filenames
        limit: Maximum number of images to copy

    Returns:
        List of paths to the copied images
    """
    # Ensure directories exist
    ensure_dir_exists(dest_dir)

    # Find image files matching the pattern
    search_pattern = os.path.join(source_dir, pattern)
    files = glob.glob(search_pattern)
    logger.debug(f"Found {len(files)} files matching pattern {pattern} in {source_dir}")

    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)

    # Limit the number of files
    files = files[:limit]

    # Copy the files
    copied_files = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)
        try:
            shutil.copy2(file_path, dest_path)
            logger.info(f"Successfully copied {file_name} to {dest_path}")
            copied_files.append(dest_path)
        except Exception as e:
            logger.error(f"Error copying file {file_path} to {dest_path}: {e}")

    # Return the list of copied files
    if copied_files:
        logger.info(f"Copied {len(copied_files)} images to {dest_dir}")
    else:
        logger.warning(f"No images were copied to {dest_dir}")

    return copied_files
