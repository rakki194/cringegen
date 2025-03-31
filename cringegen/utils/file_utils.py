"""
Utilities for file operations in the CringeGen package.
"""

import glob
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

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


def rsync_image_from_comfyui(
    image_name: str, 
    ssh_host: str, 
    remote_dir: str, 
    dest_dir: str, 
    output_prefix: str = None,
    ssh_port: int = 1487,
    ssh_user: str = None,
    ssh_key: str = None
) -> Optional[str]:
    """Copy an image from a remote ComfyUI output directory using rsync over SSH.

    Args:
        image_name: Name of the image file or full ComfyUI path
        ssh_host: SSH hostname or IP
        remote_dir: Remote directory containing the image
        dest_dir: Local destination directory
        output_prefix: Optional prefix for output filename
        ssh_port: SSH port (default: 1487)
        ssh_user: SSH username (default: current user)
        ssh_key: Path to SSH private key file (default: use system default)

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
    
    # Build remote path
    remote_path = os.path.join(remote_dir, subfolder, image_name)
    
    # Create output filename with optional prefix
    if output_prefix:
        filename, ext = os.path.splitext(image_name)
        output_filename = f"{output_prefix}{ext}"
    else:
        output_filename = image_name
    
    dest_path = os.path.join(dest_dir, output_filename)

    # Prepare SSH host string
    if ssh_user:
        ssh_host_str = f"{ssh_user}@{ssh_host}"
    else:
        ssh_host_str = ssh_host

    # Prepare rsync command
    rsync_cmd = ["rsync", "-avz", "--progress"]
    
    # Add SSH options
    ssh_options = f"ssh -p {ssh_port}"
    if ssh_key:
        ssh_options += f" -i {ssh_key}"
    
    rsync_cmd.extend(["-e", ssh_options])
    
    # Add source and destination
    rsync_cmd.append(f"{ssh_host_str}:{remote_path}")
    rsync_cmd.append(dest_path)
    
    logger.info(f"Running rsync command: {' '.join(rsync_cmd)}")
    
    try:
        # Execute rsync command
        result = subprocess.run(
            rsync_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        logger.debug(f"Rsync output: {result.stdout}")
        logger.info(f"Successfully rsynced {image_name} to {dest_path}")
        return dest_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Rsync error: {e.stderr}")
        
        # Try alternative path (without subfolder)
        if subfolder:
            logger.info("Trying alternative path without subfolder...")
            alt_remote_path = os.path.join(remote_dir, image_name)
            rsync_cmd[-2] = f"{ssh_host_str}:{alt_remote_path}"
            
            try:
                result = subprocess.run(
                    rsync_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                logger.debug(f"Rsync output: {result.stdout}")
                logger.info(f"Successfully rsynced {image_name} to {dest_path}")
                return dest_path
            except subprocess.CalledProcessError as e2:
                logger.error(f"Alternative rsync also failed: {e2.stderr}")
        
        return None
    except Exception as e:
        logger.error(f"Error during rsync: {e}")
        return None


def rsync_latest_images_from_comfyui(
    ssh_host: str, 
    remote_dir: str, 
    dest_dir: str, 
    limit: int = 5,
    ssh_port: int = 1487,
    ssh_user: str = None,
    ssh_key: str = None
) -> List[str]:
    """Copy the latest images from a remote ComfyUI output directory using rsync over SSH.

    Args:
        ssh_host: SSH hostname or IP
        remote_dir: Remote directory containing the images
        dest_dir: Local destination directory
        limit: Maximum number of images to copy
        ssh_port: SSH port (default: 1487)
        ssh_user: SSH username (default: current user)
        ssh_key: Path to SSH private key file (default: use system default)

    Returns:
        List of paths to the copied images
    """
    # Ensure destination directory exists
    ensure_dir_exists(dest_dir)
    
    # Prepare SSH host string
    if ssh_user:
        ssh_host_str = f"{ssh_user}@{ssh_host}"
    else:
        ssh_host_str = ssh_host
    
    # First, get a list of files by running ls -t (sort by modification time)
    ssh_cmd = ["ssh"]
    
    # Add SSH options
    if ssh_port != 1487:
        ssh_cmd.extend(["-p", str(ssh_port)])
    
    if ssh_key:
        ssh_cmd.extend(["-i", ssh_key])
    
    ssh_cmd.append(ssh_host_str)
    
    # Use ls -t to get files sorted by modification time, newest first
    ls_cmd = f"find '{remote_dir}' -type f -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' | xargs ls -t 2>/dev/null | head -n {limit}"
    ssh_cmd.append(ls_cmd)
    
    logger.info(f"Running SSH command to list files: {' '.join(ssh_cmd)}")
    
    try:
        # Execute SSH command
        result = subprocess.run(
            ssh_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        if result.returncode != 0:
            logger.error(f"SSH error: {result.stderr}")
            return []
        
        remote_files = result.stdout.strip().split('\n')
        remote_files = [f for f in remote_files if f]  # Filter empty lines
        
        if not remote_files:
            logger.warning(f"No files found in remote directory {remote_dir}")
            return []
        
        logger.info(f"Found {len(remote_files)} files to copy")
        
        # Now rsync each file
        copied_files = []
        for remote_file in remote_files:
            if not remote_file:
                continue
                
            filename = os.path.basename(remote_file)
            dest_path = os.path.join(dest_dir, filename)
            
            # Prepare rsync command
            rsync_cmd = ["rsync", "-avz", "--progress"]
            
            # Add SSH options
            ssh_options = f"ssh -p {ssh_port}"
            if ssh_key:
                ssh_options += f" -i {ssh_key}"
            
            rsync_cmd.extend(["-e", ssh_options])
            
            # Add source and destination
            rsync_cmd.append(f"{ssh_host_str}:{remote_file}")
            rsync_cmd.append(dest_path)
            
            logger.debug(f"Running rsync command: {' '.join(rsync_cmd)}")
            
            try:
                # Execute rsync command
                rsync_result = subprocess.run(
                    rsync_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                logger.debug(f"Rsync output: {rsync_result.stdout}")
                logger.info(f"Successfully rsynced {filename} to {dest_path}")
                copied_files.append(dest_path)
            except subprocess.CalledProcessError as e:
                logger.error(f"Rsync error for {filename}: {e.stderr}")
            except Exception as e:
                logger.error(f"Error during rsync for {filename}: {e}")
        
        # Return the list of copied files
        if copied_files:
            logger.info(f"Copied {len(copied_files)} images to {dest_dir}")
        else:
            logger.warning(f"No images were copied to {dest_dir}")
        
        return copied_files
    except Exception as e:
        logger.error(f"Error getting file list from remote server: {e}")
        return []


def open_images_with_imv(image_paths: List[str]) -> bool:
    """Open images with imv image viewer.

    Args:
        image_paths: List of paths to images to open

    Returns:
        True if successful, False otherwise
    """
    if not image_paths:
        logger.warning("No images to open with imv")
        return False
    
    try:
        # Check if imv is installed
        which_result = subprocess.run(
            ["which", "imv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if which_result.returncode != 0:
            logger.error("imv is not installed. Please install imv to use the --show option.")
            return False
        
        # Build command to open images with imv
        cmd = ["imv"]
        cmd.extend(image_paths)
        
        logger.info(f"Opening {len(image_paths)} images with imv")
        
        # Run in background
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        return True
    except Exception as e:
        logger.error(f"Error opening images with imv: {e}")
        return False
