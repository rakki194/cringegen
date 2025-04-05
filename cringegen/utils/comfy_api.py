"""
Utilities for interacting with the ComfyUI API
"""

import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .logger import get_logger

# Create logger
logger = get_logger(__name__)

# Cache for samplers, schedulers, LoRAs and checkpoints
_SAMPLER_CACHE = None
_SCHEDULER_CACHE = None
_LORA_CACHE = None
_CHECKPOINT_CACHE = None

# Global ComfyAPI instance cache for each URL
_COMFY_API_CACHE = {}

# Checkpoint to LoRA directory mappings
# Maps checkpoint names to LoRA directory prefixes
CHECKPOINT_LORA_MAPPINGS = {
    "noobaiXLVpredv10.safetensors": ["noob/"],
    "ponyDiffusionV6XL_v6StartWithThisOne.safetensors": ["pony/"],
}

# LoRA directory prefix to checkpoint mappings (reverse of above)
LORA_CHECKPOINT_MAPPINGS = {
    "noob/": [
        "noobaiXLVpredv10.safetensors",
    ],
    "pony/": ["ponyDiffusionV6XL_v6StartWithThisOne.safetensors"],
}


class ComfyAPIError(Exception):
    """Exception raised for errors in the ComfyUI API."""

    pass


def resolve_model_path(model_path: str, model_type: str = "lora") -> str:
    """Resolve a model path for use with ComfyUI

    If a full path is provided, this function extracts just the filename that ComfyUI expects.
    If a relative path is provided, it is returned as-is.

    Args:
        model_path: Path to the model (absolute or relative)
        model_type: Type of model (lora, checkpoint, vae, etc.)

    Returns:
        The filename of the model as expected by ComfyUI
    """
    # If the path is already relative (contains no path separator), return it as is
    if "/" not in model_path and "\\" not in model_path:
        return model_path

    # Extract just the filename from the path
    filename = os.path.basename(model_path)
    logger.info(f"Resolved model path: {model_path} -> {filename}")
    return filename


class ComfyAPI:
    """Class for interacting with the ComfyUI API"""

    def __init__(self, base_url: str = None):
        """Initialize the ComfyUI API client

        Args:
            base_url: The base URL of the ComfyUI server
        """
        if base_url is None:
            base_url = get_comfy_api_url()

        self.base_url = base_url
        logger.info(f"Initializing ComfyAPI with base URL: {base_url}")
        self.client_id = self._get_client_id()
        logger.info(f"Client ID: {self.client_id}")

    def _get_client_id(self) -> str:
        """Get a client ID from the ComfyUI server"""
        try:
            logger.debug(f"Getting client ID from {self.base_url}/prompt")
            r = requests.get(f"{self.base_url}/prompt")
            logger.debug(f"Response status: {r.status_code}")
            json_data = r.json()
            logger.debug(f"Response JSON: {json_data}")
            client_id = json_data.get("client_id", "")
            logger.debug(f"Extracted client ID: {client_id}")

            # If client_id is empty, generate a random one
            if not client_id:
                import uuid

                client_id = str(uuid.uuid4())
                logger.debug(f"Generated random client ID: {client_id}")

            return client_id
        except Exception as e:
            logger.error(f"Error in _get_client_id: {e}")
            # Return a random client ID on error
            import uuid

            client_id = str(uuid.uuid4())
            logger.warning(f"Generated random client ID on error: {client_id}")
            return client_id

    def queue_prompt(self, workflow: Dict[str, Any], max_retries: int = 5, initial_retry_delay: float = 1.0) -> Dict[str, Any]:
        """Queue a workflow prompt for execution

        Args:
            workflow: The workflow to queue
            max_retries: Maximum number of retries on failure (default: 5)
            initial_retry_delay: Initial delay between retries in seconds (default: 1.0)

        Returns:
            The response from the server
        """
        retry_count = 0
        retry_delay = initial_retry_delay
        last_error = None

        while retry_count <= max_retries:
            try:
                prompt_data = {"prompt": workflow, "client_id": self.client_id}
                logger.debug(f"Queuing prompt with client ID: {self.client_id}")
                logger.debug(f"Sending request to {self.base_url}/prompt")
                r = requests.post(f"{self.base_url}/prompt", json=prompt_data)
                logger.debug(f"Response status: {r.status_code}")
                json_data = r.json()
                logger.debug(f"Response JSON: {json_data}")
                return json_data
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"Connection error when queueing prompt: {e}")
                    logger.info(f"Retrying ({retry_count}/{max_retries}) in {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    # Exponential backoff with jitter
                    retry_delay = min(retry_delay * 2, 30.0) * (0.9 + 0.2 * random.random())
                else:
                    logger.error(f"Max retries exceeded when queueing prompt: {e}")
                    break
            except Exception as e:
                last_error = e
                logger.error(f"Error in queue_prompt: {e}")
                break
        
        # If we've exhausted retries or encountered a non-connection error
        raise ComfyAPIError(f"Failed to queue prompt: {last_error}")

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Get an image from the ComfyUI server

        Args:
            filename: The filename of the image
            subfolder: The subfolder containing the image
            folder_type: The type of folder (input or output)

        Returns:
            The image data
        """
        try:
            r = requests.get(
                f"{self.base_url}/view",
                params={"filename": filename, "subfolder": subfolder, "type": folder_type},
            )
            if r.status_code == 200:
                return r.content
            else:
                logger.error(f"Failed to get image: HTTP {r.status_code}")
                raise ComfyAPIError(f"Failed to get image: {r.status_code}")
        except Exception as e:
            logger.error(f"Error getting image: {e}")
            raise ComfyAPIError(f"Error getting image: {e}")

    def wait_for_image(self, prompt_id: str, timeout: int = 300, max_retries: int = 10, initial_retry_delay: float = 1.0) -> List[Dict[str, Any]]:
        """Wait for an image to be generated

        Args:
            prompt_id: The ID of the prompt
            timeout: Timeout in seconds (default: 300)
            max_retries: Maximum consecutive connection errors before failing (default: 10)
            initial_retry_delay: Initial delay between retries in seconds (default: 1.0)

        Returns:
            A list of image information dictionaries
        """
        start_time = time.time()
        logger.debug(f"Waiting for image with prompt ID {prompt_id}, timeout {timeout}s")

        # Keep track of consecutive errors
        consecutive_errors = 0
        retry_delay = initial_retry_delay

        while time.time() - start_time < timeout:
            try:
                r = requests.get(f"{self.base_url}/history/{prompt_id}")

                # Special handling for 404 - this usually means the prompt is still processing
                # and hasn't been added to history yet, so it's not a real error
                if r.status_code == 404:
                    logger.debug(
                        f"Prompt {prompt_id} not in history yet, waiting... ({int(time.time() - start_time)}s elapsed)"
                    )
                    time.sleep(2)  # Wait a bit longer for processing
                    continue  # This is not counted as an error

                # Check if we got a valid response with content
                if r.status_code != 200 or not r.text.strip():
                    logger.warning(
                        f"Invalid or empty response from server (status {r.status_code})"
                    )
                    consecutive_errors += 1

                    # If we've had too many consecutive errors, fail
                    if consecutive_errors >= max_retries:
                        logger.error(f"Too many consecutive errors ({max_retries})")
                        raise ComfyAPIError(f"Too many consecutive errors while waiting for image")

                    # Wait with exponential backoff before retrying
                    time.sleep(retry_delay)
                    # Exponential backoff with jitter
                    retry_delay = min(retry_delay * 1.5, 10.0) * (0.9 + 0.2 * random.random())
                    continue

                # Reset error counter and delay on successful response
                consecutive_errors = 0
                retry_delay = initial_retry_delay

                # Try to decode JSON
                try:
                    data = r.json()
                except ValueError as e:
                    logger.warning(f"Invalid JSON response: {r.text[:100]}...")
                    time.sleep(1)
                    continue

                # Check if we have the expected data structure
                if not isinstance(data, dict):
                    logger.warning(f"Unexpected response format (not a dict): {type(data)}")
                    time.sleep(1)
                    continue

                # Check if we have output data for our prompt
                if data.get(prompt_id, {}).get("outputs"):
                    logger.debug(f"Image generated after {time.time() - start_time:.2f}s")
                    return self._extract_image_info(data[prompt_id])

                # If we get here, the prompt is in history but not yet complete
                logger.debug(
                    f"Prompt {prompt_id} in history but processing not complete... ({int(time.time() - start_time)}s elapsed)"
                )

                # Not ready yet, wait and retry
                time.sleep(2)

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error when checking status: {e}")
                consecutive_errors += 1

                # If we've had too many consecutive errors, fail
                if consecutive_errors >= max_retries:
                    logger.error(f"Too many consecutive connection errors ({max_retries})")
                    raise ComfyAPIError(f"Connection error while waiting for image: {e}")

                # Wait with exponential backoff before retrying
                logger.info(f"Retrying in {retry_delay:.1f}s... ({consecutive_errors}/{max_retries})")
                time.sleep(retry_delay)
                # Exponential backoff with jitter
                retry_delay = min(retry_delay * 1.5, 10.0) * (0.9 + 0.2 * random.random())
                continue

            except Exception as e:
                logger.error(f"Unexpected error while waiting for image: {e}")
                consecutive_errors += 1

                # If we've had too many consecutive errors, fail
                if consecutive_errors >= max_retries:
                    logger.error(f"Too many consecutive errors ({max_retries})")
                    raise ComfyAPIError(f"Error while waiting for image: {e}")

                # Wait with exponential backoff before retrying
                time.sleep(retry_delay)
                # Exponential backoff with jitter
                retry_delay = min(retry_delay * 1.5, 10.0) * (0.9 + 0.2 * random.random())

        # If we get here, we've timed out
        logger.warning(f"Timed out after {timeout}s waiting for image")
        raise ComfyAPIError(f"Timed out after {timeout}s waiting for image")

    def _extract_image_info(self, history_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract image information from a history item

        Args:
            history_item: The history item from which to extract image information

        Returns:
            A list of image information dictionaries
        """
        outputs = history_item.get("outputs", {})
        images = []

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img in node_output["images"]:
                    images.append(
                        {
                            "filename": img.get("filename"),
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        }
                    )

        logger.debug(f"Extracted {len(images)} images from history item")
        return images

    def check_generation_status(self, prompt_id: str, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
        """Check the status of a generation without waiting for timeout

        Args:
            prompt_id: The ID of the prompt
            max_retries: Maximum number of retries on connection failure (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)

        Returns:
            A dictionary with the status information:
            {
                "status": "pending"|"processing"|"completed"|"error",
                "progress": float (0-1),
                "images": list of image dicts (empty if not completed),
                "error": error message (if status is "error")
            }
        """
        for attempt in range(1, max_retries + 1):
            try:
                # Check if prompt exists in history (meaning it's been processed or is processing)
                r = requests.get(f"{self.base_url}/history/{prompt_id}")

                # If not in history yet, it's still pending
                if r.status_code == 404:
                    logger.debug(f"Prompt {prompt_id} not found in history yet, likely still pending")
                    return {"status": "pending", "progress": 0.0, "images": [], "error": None}

                # Check for other errors
                if r.status_code != 200:
                    logger.warning(f"API returned error code {r.status_code} for prompt {prompt_id}")
                    if attempt < max_retries:
                        logger.info(f"Retrying status check ({attempt}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return {
                        "status": "error",
                        "progress": 0.0,
                        "images": [],
                        "error": f"API returned error code {r.status_code}",
                    }

                # Parse JSON response
                try:
                    data = r.json()
                except ValueError as e:
                    logger.warning(f"Invalid JSON response for prompt {prompt_id}: {e}")
                    if attempt < max_retries:
                        logger.info(f"Retrying after JSON parse error ({attempt}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return {
                        "status": "error",
                        "progress": 0.0,
                        "images": [],
                        "error": "Invalid JSON response",
                    }

                # Check if this is a valid history item
                if not isinstance(data, dict):
                    logger.warning(f"Unexpected response format for prompt {prompt_id}: {type(data)}")
                    if attempt < max_retries:
                        logger.info(f"Retrying after format error ({attempt}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return {
                        "status": "error",
                        "progress": 0.0,
                        "images": [],
                        "error": "Unexpected response format",
                    }

                # Check if prompt ID is in the response
                if prompt_id not in data:
                    # This could be because the prompt is still pending or because it's invalid
                    # Try to check if there are any recent outputs we can use instead
                    logger.debug(
                        f"Prompt {prompt_id} not found in history data. Checking for recent outputs."
                    )

                    # If history has any entries at all, see if we can find the most recent one
                    if data:
                        # Get most recent entry
                        most_recent = None
                        most_recent_num = -1

                        for key, entry in data.items():
                            if isinstance(entry, dict) and "outputs" in entry:
                                entry_num = entry.get("number", -1)
                                if entry_num > most_recent_num:
                                    most_recent = key
                                    most_recent_num = entry_num

                        if most_recent and "outputs" in data[most_recent]:
                            # We found a recent output, use that instead
                            logger.info(
                                f"Using most recent output from prompt {most_recent} instead of {prompt_id}"
                            )
                            images = self._extract_image_info(data[most_recent])
                            if images:
                                return {
                                    "status": "completed",
                                    "progress": 1.0,
                                    "images": images,
                                    "error": f"Using output from prompt {most_recent} as {prompt_id} was not found",
                                }

                    # If we got here, we didn't find any suitable replacement
                    return {
                        "status": "pending",  # Assume pending rather than error
                        "progress": 0.0,
                        "images": [],
                        "error": "Prompt ID not found in history data",
                    }

                # Get the history item for this prompt
                history_item = data[prompt_id]

                # Check if completed (has outputs)
                if "outputs" in history_item and history_item["outputs"]:
                    # Extract images
                    images = self._extract_image_info(history_item)
                    return {"status": "completed", "progress": 1.0, "images": images, "error": None}

                # Still processing - check progress if available
                progress = 0.0
                if "progress" in history_item:
                    progress = float(history_item["progress"])

                return {"status": "processing", "progress": progress, "images": [], "error": None}

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error checking generation status: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying after connection error ({attempt}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                return {"status": "error", "progress": 0.0, "images": [], "error": f"Connection error: {str(e)}"}
            except Exception as e:
                logger.warning(f"Error checking generation status: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying after unexpected error ({attempt}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                return {"status": "error", "progress": 0.0, "images": [], "error": str(e)}


# Helper functions for easier use


def get_comfy_api_url() -> str:
    """Get the ComfyUI API URL from environment variable or use default

    Returns:
        URL of the ComfyUI API
    """
    return os.environ.get("COMFY_API_URL", "http://127.0.0.1:8188")


def queue_prompt(workflow: Dict[str, Any], api_url: str = None, max_retries: int = 5, initial_retry_delay: float = 1.0) -> Dict[str, Any]:
    """Queue a workflow prompt for execution

    Args:
        workflow: The workflow to queue
        api_url: URL of the ComfyUI server
        max_retries: Maximum number of retries on failure (default: 5)
        initial_retry_delay: Initial delay between retries in seconds (default: 1.0)

    Returns:
        The response from the server
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    api = get_comfy_api_client(api_url)
    return api.queue_prompt(workflow, max_retries=max_retries, initial_retry_delay=initial_retry_delay)


def get_image_path(prompt_id: str, api_url: str = None, timeout: int = 300, max_retries: int = 10, initial_retry_delay: float = 1.0) -> List[str]:
    """Get the path to generated images

    Args:
        prompt_id: ID of the prompt
        api_url: URL of the ComfyUI server
        timeout: Timeout in seconds (default: 300)
        max_retries: Maximum consecutive connection errors before failing (default: 10)
        initial_retry_delay: Initial delay between retries in seconds (default: 1.0)

    Returns:
        List of image paths
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    api = get_comfy_api_client(api_url)

    try:
        # First, check if we can directly get the history without waiting
        try:
            logger.info(f"Checking if image is already generated for prompt {prompt_id}")
            r = requests.get(f"{api_url}/history/{prompt_id}")
            if r.status_code == 200:
                try:
                    data = r.json()
                    if prompt_id in data and "outputs" in data[prompt_id]:
                        # Extract image paths directly
                        history_data = data[prompt_id]
                        images = api._extract_image_info(history_data)

                        if images:
                            filenames = [img["filename"] for img in images]
                            logger.info(f"Image already generated: {', '.join(filenames)}")
                            return filenames
                        else:
                            logger.info("Prompt found in history but no images yet, waiting...")
                except Exception as e:
                    logger.warning(f"Failed to parse history data: {e}")
        except Exception as e:
            logger.warning(f"Failed to check history directly: {e}")

        # If direct check didn't work, wait for the image normally
        logger.info(f"Waiting for image generation (timeout: {timeout}s)")
        images = api.wait_for_image(prompt_id, timeout, max_retries=max_retries, initial_retry_delay=initial_retry_delay)

        if not images:
            logger.warning("No images were generated")
            return []

        filenames = [img["filename"] for img in images]
        logger.info(f"Image generated: {', '.join(filenames)}")
        return filenames

    except ComfyAPIError as e:
        logger.error(f"Failed to get image path: {e}")

        # If we timed out, check if we can see any completed images
        if "Timed out" in str(e):
            logger.info("Timed out but checking for partial results...")
            try:
                r = requests.get(f"{api_url}/history/{prompt_id}")
                if r.status_code == 200:
                    data = r.json()
                    if prompt_id in data and "outputs" in data[prompt_id]:
                        # Extract image paths directly
                        history_data = data[prompt_id]
                        images = api._extract_image_info(history_data)

                        if images:
                            filenames = [img["filename"] for img in images]
                            logger.info(f"Found images after timeout: {', '.join(filenames)}")
                            return filenames
            except Exception as nested_e:
                logger.error(f"Failed to check for partial results: {nested_e}")

        return []

    except Exception as e:
        logger.error(f"Unexpected error in get_image_path: {e}")
        return []


def get_available_samplers(api_url: str = None, force_refresh: bool = False) -> List[str]:
    """Get a list of available samplers

    Args:
        api_url: URL of the ComfyUI server
        force_refresh: Whether to force refresh the cache

    Returns:
        List of available samplers
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    global _SAMPLER_CACHE

    # If samplers are cached and we don't need to refresh, return the cache
    if _SAMPLER_CACHE is not None and not force_refresh:
        return _SAMPLER_CACHE

    # Default fallback list if we can't get samplers from API
    fallback_samplers = [
        "euler",
        "euler_ancestral",
        "heun",
        "dpm_2",
        "dpm_2_ancestral",
        "lms",
        "dpm_fast",
        "dpm_adaptive",
        "dpmpp_2s_ancestral",
        "dpmpp_sde",
        "dpmpp_2m",
        "ddim",
        "uni_pc",
    ]

    # Otherwise, fetch the samplers from the server
    try:
        logger.debug(f"Fetching available samplers from {api_url}/object_info")
        r = requests.get(f"{api_url}/object_info")
        data = r.json()

        # Look for KSampler node to get available samplers
        samplers = []

        # First try standard KSampler
        if "KSampler" in data and "input" in data["KSampler"]:
            sampler_info = data["KSampler"]["input"].get("required", {}).get("sampler_name", {})
            if isinstance(sampler_info, list) and len(sampler_info) > 0:
                # In some ComfyUI versions, the first element is the list of samplers
                if isinstance(sampler_info[0], list):
                    samplers = sampler_info[0]
                else:
                    samplers = sampler_info

        # If no samplers found, try other KSampler nodes
        if not samplers:
            for node_type, node_info in data.items():
                if "KSampler" in node_type and "input" in node_info:
                    sampler_info = node_info["input"].get("required", {}).get("sampler_name", {})
                    if isinstance(sampler_info, list) and len(sampler_info) > 0:
                        # In some ComfyUI versions, the first element is the list of samplers
                        if isinstance(sampler_info[0], list):
                            samplers = sampler_info[0]
                            break
                        else:
                            samplers = sampler_info
                            break

        # If we still have no samplers, use the fallback list
        if not samplers:
            logger.warning("Could not find samplers in ComfyUI API response, using fallback list")
            samplers = fallback_samplers

        # Cache the samplers
        _SAMPLER_CACHE = samplers
        logger.debug(f"Found {len(samplers)} samplers: {', '.join(samplers)}")
        return samplers
    except Exception as e:
        logger.error(f"Error fetching samplers: {e}")
        logger.warning("Using fallback sampler list due to error")
        _SAMPLER_CACHE = fallback_samplers
        return fallback_samplers


def get_available_schedulers(api_url: str = None, force_refresh: bool = False) -> List[str]:
    """Get a list of available schedulers

    Args:
        api_url: URL of the ComfyUI server
        force_refresh: Whether to force refresh the cache

    Returns:
        List of available schedulers
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    global _SCHEDULER_CACHE

    # If schedulers are cached and we don't need to refresh, return the cache
    if _SCHEDULER_CACHE is not None and not force_refresh:
        return _SCHEDULER_CACHE

    # Default fallback list if we can't get schedulers from API
    fallback_schedulers = [
        "normal",
        "karras",
        "exponential",
        "sgm_uniform",
        "simple",
        "ddim_uniform",
        "lcm",
        "turbo",
    ]

    # Otherwise, fetch the schedulers from the server
    try:
        logger.debug(f"Fetching available schedulers from {api_url}/object_info")
        r = requests.get(f"{api_url}/object_info")
        data = r.json()

        # Look for KSampler node to get available schedulers
        schedulers = []

        # First try standard KSampler
        if "KSampler" in data and "input" in data["KSampler"]:
            scheduler_info = data["KSampler"]["input"].get("required", {}).get("scheduler", {})
            if isinstance(scheduler_info, list) and len(scheduler_info) > 0:
                # In some ComfyUI versions, the first element is the list of schedulers
                if isinstance(scheduler_info[0], list):
                    schedulers = scheduler_info[0]
                else:
                    schedulers = scheduler_info

        # If no schedulers found, try other KSampler nodes
        if not schedulers:
            for node_type, node_info in data.items():
                if "KSampler" in node_type and "input" in node_info:
                    scheduler_info = node_info["input"].get("required", {}).get("scheduler", {})
                    if isinstance(scheduler_info, list) and len(scheduler_info) > 0:
                        # In some ComfyUI versions, the first element is the list of schedulers
                        if isinstance(scheduler_info[0], list):
                            schedulers = scheduler_info[0]
                            break
                        else:
                            schedulers = scheduler_info
                            break

        # If we still have no schedulers, use the fallback list
        if not schedulers:
            logger.warning("Could not find schedulers in ComfyUI API response, using fallback list")
            schedulers = fallback_schedulers

        # Cache the schedulers
        _SCHEDULER_CACHE = schedulers
        logger.debug(f"Found {len(schedulers)} schedulers: {', '.join(schedulers)}")
        return schedulers
    except Exception as e:
        logger.error(f"Error fetching schedulers: {e}")
        logger.warning("Using fallback scheduler list due to error")
        _SCHEDULER_CACHE = fallback_schedulers
        return fallback_schedulers


def is_valid_sampler(sampler: str, api_url: str = None) -> bool:
    """Check if a sampler is valid

    Args:
        sampler: Name of the sampler
        api_url: URL of the ComfyUI server

    Returns:
        True if the sampler is valid, False otherwise
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    samplers = get_available_samplers(api_url)
    is_valid = sampler in samplers
    if not is_valid:
        logger.warning(f"Invalid sampler: {sampler}. Available samplers: {', '.join(samplers)}")
    return is_valid


def is_valid_scheduler(scheduler: str, api_url: str = None) -> bool:
    """Check if a scheduler is valid

    Args:
        scheduler: Name of the scheduler
        api_url: URL of the ComfyUI server

    Returns:
        True if the scheduler is valid, False otherwise
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    schedulers = get_available_schedulers(api_url)
    is_valid = scheduler in schedulers
    if not is_valid:
        logger.warning(
            f"Invalid scheduler: {scheduler}. Available schedulers: {', '.join(schedulers)}"
        )
    return is_valid


def get_available_checkpoints(api_url: str = None, force_refresh: bool = False) -> List[str]:
    """Get a list of available checkpoints

    Args:
        api_url: URL of the ComfyUI server
        force_refresh: Whether to force refresh the cache

    Returns:
        List of available checkpoints
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    global _CHECKPOINT_CACHE

    # If checkpoints are cached and we don't need to refresh, return the cache
    if _CHECKPOINT_CACHE is not None and not force_refresh:
        return _CHECKPOINT_CACHE

    # Otherwise, fetch the checkpoints from the server
    try:
        logger.info(f"Fetching available checkpoints from {api_url}/object_info")
        r = requests.get(f"{api_url}/object_info")
        if r.status_code != 200:
            logger.error(f"Failed to get object info: HTTP {r.status_code}")
            return []

        data = r.json()
        logger.debug(f"Received API response")

        # Extract available checkpoints from the response
        checkpoints = []

        # First try CheckpointLoaderSimple
        if "CheckpointLoaderSimple" in data:
            checkpoint_info = data["CheckpointLoaderSimple"]["input"]["required"].get(
                "ckpt_name", None
            )
            logger.debug(f"Checkpoint info type: {type(checkpoint_info)}")
            if isinstance(checkpoint_info, list) and len(checkpoint_info) > 0:
                # In some ComfyUI versions, the first element is the list of checkpoints
                if isinstance(checkpoint_info[0], list):
                    checkpoints = checkpoint_info[0]
                else:
                    checkpoints = checkpoint_info

        # If no checkpoints found via CheckpointLoaderSimple, try other checkpoint loaders
        if not checkpoints:
            for node_type, node_info in data.items():
                if "checkpoint" in node_type.lower() or "ckpt" in node_type.lower():
                    for param_name, param_info in (
                        node_info.get("input", {}).get("required", {}).items()
                    ):
                        if param_name.lower() in ["ckpt_name", "checkpoint", "model_name"]:
                            if isinstance(param_info, list) and len(param_info) > 0:
                                if isinstance(param_info[0], list):
                                    checkpoints = param_info[0]
                                else:
                                    checkpoints = param_info
                                break
                    if checkpoints:
                        break

        # Cache the checkpoints
        _CHECKPOINT_CACHE = checkpoints
        logger.info(f"Found {len(checkpoints)} checkpoints")
        return checkpoints
    except Exception as e:
        logger.error(f"Error getting checkpoints: {e}")
        return []


# Alias for backward compatibility
get_checkpoint_list = get_available_checkpoints


def is_valid_checkpoint(checkpoint: str, api_url: str = None) -> bool:
    """Check if a checkpoint is valid and available in ComfyUI

    Args:
        checkpoint: The checkpoint name to check (can be a filename or path)
        api_url: The URL of the ComfyUI API

    Returns:
        True if the checkpoint is valid, False otherwise
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    available_checkpoints = get_available_checkpoints(api_url)

    # Direct match
    if checkpoint in available_checkpoints:
        return True

    # Check if it's a filename that appears in any of the available paths
    checkpoint_basename = os.path.basename(checkpoint)
    for available_checkpoint in available_checkpoints:
        if (
            available_checkpoint.endswith("/" + checkpoint_basename)
            or available_checkpoint == checkpoint_basename
        ):
            logger.debug(f"Found matching checkpoint: '{available_checkpoint}' for '{checkpoint}'")
            return True

    return False


def get_compatible_loras(checkpoint: str, api_url: str = None) -> List[str]:
    """Get a list of LoRAs compatible with a specific checkpoint

    Args:
        checkpoint: The checkpoint to find compatible LoRAs for
        api_url: The URL of the ComfyUI API

    Returns:
        A list of compatible LoRAs
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    # Normalize checkpoint name to just the filename
    checkpoint_basename = os.path.basename(checkpoint)

    # Get all available LoRAs
    all_loras = get_available_loras(api_url)

    # If the checkpoint is in our mappings, filter by compatible directories
    if checkpoint_basename in CHECKPOINT_LORA_MAPPINGS:
        compatible_prefixes = CHECKPOINT_LORA_MAPPINGS[checkpoint_basename]
        return [
            lora
            for lora in all_loras
            if any(lora.startswith(prefix) for prefix in compatible_prefixes)
        ]
    else:
        # If no mapping exists, return an empty list
        logger.debug(f"No LoRA compatibility mapping found for checkpoint: {checkpoint_basename}")
        return []


def get_compatible_checkpoints(lora: str, api_url: str = None) -> List[str]:
    """Get a list of checkpoints compatible with a specific LoRA

    Args:
        lora: The LoRA to find compatible checkpoints for
        api_url: The URL of the ComfyUI API

    Returns:
        A list of compatible checkpoints
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    # Get all available checkpoints
    all_checkpoints = get_available_checkpoints(api_url)

    # Find the directory prefix of the LoRA
    lora_prefix = None
    for prefix in LORA_CHECKPOINT_MAPPINGS.keys():
        if lora.startswith(prefix):
            lora_prefix = prefix
            break

    # If we found a prefix, return compatible checkpoints
    if lora_prefix:
        compatible_checkpoints = LORA_CHECKPOINT_MAPPINGS[lora_prefix]
        return [
            cp
            for cp in all_checkpoints
            if any(cp.endswith(compat_cp) for compat_cp in compatible_checkpoints)
        ]
    else:
        # If no mapping exists, return an empty list
        logger.debug(f"No checkpoint compatibility mapping found for LoRA: {lora}")
        return []


def get_lora_directory_prefix(lora: str) -> Optional[str]:
    """Get the directory prefix of a LoRA

    Args:
        lora: The LoRA path

    Returns:
        The directory prefix, or None if not found
    """
    for prefix in LORA_CHECKPOINT_MAPPINGS.keys():
        if lora.startswith(prefix):
            return prefix
    return None


def get_checkpoint_suggestion(lora: str, api_url: str = None) -> Optional[str]:
    """Get a suggested checkpoint for a specific LoRA

    Args:
        lora: The LoRA to find a compatible checkpoint for
        api_url: The URL of the ComfyUI API

    Returns:
        A suggested checkpoint, or None if none found
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    compatible_checkpoints = get_compatible_checkpoints(lora, api_url)
    return compatible_checkpoints[0] if compatible_checkpoints else None


def get_lora_suggestions(checkpoint: str, api_url: str = None, limit: int = 5) -> List[str]:
    """Get a list of suggested LoRAs for a specific checkpoint

    Args:
        checkpoint: The checkpoint to find compatible LoRAs for
        api_url: The URL of the ComfyUI API
        limit: Maximum number of suggestions to return

    Returns:
        A list of suggested LoRAs
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    compatible_loras = get_compatible_loras(checkpoint, api_url)
    # Return a random selection of compatible LoRAs
    import random

    if len(compatible_loras) <= limit:
        return compatible_loras
    else:
        return random.sample(compatible_loras, limit)


def clear_cache() -> None:
    """Clear the cache of samplers, schedulers, LoRAs and checkpoints"""
    global _SAMPLER_CACHE, _SCHEDULER_CACHE, _LORA_CACHE, _CHECKPOINT_CACHE, _COMFY_API_CACHE
    _SAMPLER_CACHE = None
    _SCHEDULER_CACHE = None
    _LORA_CACHE = None
    _CHECKPOINT_CACHE = None
    _COMFY_API_CACHE = {}


def get_matching_lora(lora: str, api_url: str = None) -> Optional[str]:
    """Get the full path of a matching LoRA from the available LoRAs

    Args:
        lora: The LoRA name to match (can be a filename or path)
        api_url: The URL of the ComfyUI API

    Returns:
        The full path of the matching LoRA, or None if no match is found
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    available_loras = get_available_loras(api_url)

    # Direct match
    if lora in available_loras:
        return lora

    # Check if it's a filename that appears in any of the available paths
    lora_basename = os.path.basename(lora)
    for available_lora in available_loras:
        if available_lora.endswith("/" + lora_basename) or available_lora == lora_basename:
            logger.debug(f"Found matching LoRA: '{available_lora}' for '{lora}'")
            return available_lora

    return None


def get_matching_checkpoint(checkpoint: str, api_url: str = None) -> Optional[str]:
    """Get the full path of a matching checkpoint from the available checkpoints

    Args:
        checkpoint: The checkpoint name to match (can be a filename or path)
        api_url: The URL of the ComfyUI API

    Returns:
        The full path of the matching checkpoint, or None if no match is found
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    available_checkpoints = get_available_checkpoints(api_url)

    # Direct match
    if checkpoint in available_checkpoints:
        return checkpoint

    # Check if it's a filename that appears in any of the available paths
    checkpoint_basename = os.path.basename(checkpoint)
    for available_checkpoint in available_checkpoints:
        if (
            available_checkpoint.endswith("/" + checkpoint_basename)
            or available_checkpoint == checkpoint_basename
        ):
            logger.debug(f"Found matching checkpoint: '{available_checkpoint}' for '{checkpoint}'")
            return available_checkpoint

    return None


def is_valid_lora(lora: str, api_url: str = None) -> bool:
    """Check if a LoRA is valid and available in ComfyUI

    Args:
        lora: The LoRA name to check (can be a filename or path)
        api_url: URL of the ComfyUI server

    Returns:
        True if the LoRA is valid, False otherwise
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    available_loras = get_available_loras(api_url)

    # Direct match
    if lora in available_loras:
        return True

    # Check if it's a filename that appears in any of the available paths
    lora_basename = os.path.basename(lora)
    for available_lora in available_loras:
        if available_lora.endswith("/" + lora_basename) or available_lora == lora_basename:
            logger.debug(f"Found matching LoRA: '{available_lora}' for '{lora}'")
            return True

    logger.warning(f"LoRA '{lora}' not found in available LoRAs")
    return False


def get_lora_directory() -> str:
    """Get the LoRA directory from ComfyUI

    Returns:
        Path to the LoRA directory
    """
    import os
    from pathlib import Path

    # Get logger
    logger = get_logger(__name__)

    # Check environment variable first
    comfy_dir = os.environ.get("COMFY_DIR", "/home/kade/toolkit/diffusion/comfy")
    logger.info(f"Using ComfyUI directory: {comfy_dir}")

    # Possible LoRA directory names
    lora_dir_names = [
        "models/loras",
        "models/Lora",
        "models/lora",
        "models/LoRAs",
        # Add more potential paths
        "models/LORA",
    ]

    # Check each possible location
    for dir_name in lora_dir_names:
        full_path = os.path.join(comfy_dir, dir_name)
        logger.info(f"Checking LoRA path: {full_path}")
        if os.path.exists(full_path):
            logger.info(f"Found LoRA directory: {full_path}")
            return full_path

    # If we get here, we haven't found any valid directories
    # Try looking more broadly for any directory with lora in the name
    models_dir = os.path.join(comfy_dir, "models")
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            if "lora" in item.lower() and os.path.isdir(os.path.join(models_dir, item)):
                full_path = os.path.join(models_dir, item)
                logger.info(f"Found alternative LoRA directory: {full_path}")
                return full_path

    # Fall back to default, even if it doesn't exist
    default_path = os.path.join(comfy_dir, "models/loras")
    logger.warning(f"No LoRA directory found. Defaulting to: {default_path}")
    return default_path


def get_lora_path(lora_name: str) -> Optional[str]:
    """Get the full path to a LoRA

    Args:
        lora_name: The name of the LoRA

    Returns:
        The full path to the LoRA, or None if not found
    """
    # Get available LoRAs
    loras = get_available_loras()

    # Try direct match
    if lora_name in loras:
        return lora_name

    # Try to match by basename
    for lora in loras:
        if os.path.basename(lora) == lora_name:
            return lora

    # Try to match by name without extension
    lora_base = os.path.splitext(lora_name)[0]
    for lora in loras:
        lora_file = os.path.basename(lora)
        if os.path.splitext(lora_file)[0] == lora_base:
            return lora

    # No match found
    return None


def get_available_loras(api_url: str = None, force_refresh: bool = False) -> List[str]:
    """Get a list of available LoRAs

    Args:
        api_url: URL of the ComfyUI server
        force_refresh: Whether to force refresh the cache

    Returns:
        List of available LoRAs
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    global _LORA_CACHE

    # If LoRAs are cached and we don't need to refresh, return the cache
    if _LORA_CACHE is not None and not force_refresh:
        return _LORA_CACHE

    # Otherwise, fetch the LoRAs from the server
    try:
        logger.info(f"Fetching available LoRAs from {api_url}/object_info")
        r = requests.get(f"{api_url}/object_info")
        if r.status_code != 200:
            logger.error(f"Failed to get object info: HTTP {r.status_code}")
            return []

        data = r.json()
        logger.debug(f"Received API response")

        # Extract available LoRAs from the response
        loras = []

        # First try LoraLoader
        if "LoraLoader" in data:
            lora_info = data["LoraLoader"]["input"]["required"].get("lora_name", None)
            logger.debug(f"LoRA info type: {type(lora_info)}")
            if isinstance(lora_info, list) and len(lora_info) > 0:
                # In some ComfyUI versions, the first element is the list of loras
                if isinstance(lora_info[0], list):
                    loras = lora_info[0]
                else:
                    loras = lora_info

        # If no loras found via LoraLoader, try other nodes with "LoRA" in name
        if not loras:
            for node_type, node_info in data.items():
                if "LoRA" in node_type or "Lora" in node_type:
                    for param_name, param_info in (
                        node_info.get("input", {}).get("required", {}).items()
                    ):
                        if param_name.lower() in ["lora_name", "lora"]:
                            if isinstance(param_info, list) and len(param_info) > 0:
                                if isinstance(param_info[0], list):
                                    loras = param_info[0]
                                else:
                                    loras = param_info
                                break
                    if loras:
                        break

        # Cache the LoRAs
        _LORA_CACHE = loras
        logger.info(f"Found {len(loras)} LoRAs")
        return loras
    except Exception as e:
        logger.error(f"Error fetching LoRAs: {e}")
        return []


def check_comfy_server(api_url: str = None, max_retries: int = 3, retry_delay: float = 1.0) -> Tuple[bool, str]:
    """Check if ComfyUI server is available

    Args:
        api_url: URL of the ComfyUI server
        max_retries: Maximum number of connection attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)

    Returns:
        Tuple of (is_available, message)
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    logger.info(f"Checking ComfyUI server at {api_url}")
    
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(f"{api_url}/", timeout=5)
            if r.status_code == 200:
                return True, "ComfyUI server is available"
            else:
                if attempt < max_retries:
                    logger.warning(f"ComfyUI server returned unexpected status code: {r.status_code}, retrying ({attempt}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                return False, f"ComfyUI server returned unexpected status code: {r.status_code}"
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                logger.warning(f"Connection to ComfyUI server failed (attempt {attempt}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return (
                False,
                f"ComfyUI server not running at {api_url}. Please start ComfyUI before generating images.",
            )
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(f"Connection to ComfyUI server timed out (attempt {attempt}/{max_retries})")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return (
                False,
                f"Connection to ComfyUI server at {api_url} timed out. Server may be busy or not responding.",
            )
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Error connecting to ComfyUI server (attempt {attempt}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return False, f"Error connecting to ComfyUI server: {e}"


def get_preferred_checkpoint(api_url: str = None) -> Optional[str]:
    """Get the preferred checkpoint from available checkpoints

    This function prioritizes certain models in this order:
    1. noobaiXLVpredv10.safetensors (first priority)
    2. Any noobai model
    3. Any ponyDiffusion model
    4. The first available checkpoint

    Args:
        api_url: URL of the ComfyUI server

    Returns:
        Name of the preferred checkpoint, or None if no checkpoints are available
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    available_checkpoints = get_available_checkpoints(api_url)
    logger.info(f"Found {len(available_checkpoints)} available checkpoints")

    if not available_checkpoints:
        logger.warning("No checkpoints available")
        return None

    # Log all available checkpoints to help with debugging
    logger.info(f"Available checkpoints: {', '.join(available_checkpoints)}")

    # First priority: noobaiXLVpredv10.safetensors
    for checkpoint in available_checkpoints:
        if checkpoint.lower() == "noobaixlvpredv10.safetensors":
            logger.info(f"Selected preferred checkpoint (exact match): {checkpoint}")
            return checkpoint

    # Second priority: any noobai model
    for checkpoint in available_checkpoints:
        if "noobai" in checkpoint.lower():
            logger.info(f"Selected noobai checkpoint: {checkpoint}")
            return checkpoint

    # Third priority: any ponyDiffusion model
    for checkpoint in available_checkpoints:
        if "ponydiffusion" in checkpoint.lower():
            logger.info(f"Selected ponyDiffusion checkpoint: {checkpoint}")
            return checkpoint

    # Fallback: first available checkpoint
    logger.info(f"Using first available checkpoint: {available_checkpoints[0]}")
    return available_checkpoints[0]


def get_comfy_api_client(api_url: str = None) -> ComfyAPI:
    """Get or create a ComfyAPI client for the given URL
    
    This function returns a cached ComfyAPI instance for the given URL,
    or creates a new one if none exists yet.
    
    Args:
        api_url: The ComfyUI API URL (default: None, will use get_comfy_api_url())
        
    Returns:
        A ComfyAPI instance
    """
    global _COMFY_API_CACHE
    
    if api_url is None:
        api_url = get_comfy_api_url()
    
    # Check if we already have a client for this URL
    if api_url in _COMFY_API_CACHE:
        logger.debug(f"Using cached ComfyAPI client for {api_url}")
        return _COMFY_API_CACHE[api_url]
    
    # Create a new client and cache it
    logger.debug(f"Creating new ComfyAPI client for {api_url}")
    client = ComfyAPI(api_url)
    _COMFY_API_CACHE[api_url] = client
    return client


def check_generation_status(prompt_id: str, api_url: str = None, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """Check the status of a generation without waiting for timeout

    Args:
        prompt_id: The ID of the prompt
        api_url: URL of the ComfyUI server
        max_retries: Maximum consecutive connection errors before failing (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)

    Returns:
        A dictionary with the status information:
        {
            "status": "pending"|"processing"|"completed"|"error",
            "progress": float (0-1),
            "images": list of image dicts (empty if not completed),
            "error": error message (if status is "error")
        }
    """
    if api_url is None:
        api_url = get_comfy_api_url()

    # Check if ComfyUI's queue_prompt endpoint is showing this as an active prompt
    # This is important because sometimes prompts don't show in history right away
    try:
        logger.debug(f"Checking prompt queue for prompt {prompt_id}")
        r = requests.get(f"{api_url}/prompt")

        if r.status_code == 200:
            try:
                queue_data = r.json()
                if "queue_running" in queue_data and queue_data["queue_running"]:
                    executing = queue_data.get("executing", {})
                    queue = queue_data.get("queue", [])

                    # Check if our prompt is currently executing
                    if executing and executing.get("prompt_id") == prompt_id:
                        logger.debug(f"Prompt {prompt_id} is currently executing")
                        progress = executing.get("progress", 0.0)
                        return {
                            "status": "processing",
                            "progress": progress,
                            "images": [],
                            "error": None,
                        }

                    # Check if our prompt is in the queue
                    if queue:
                        for item in queue:
                            if item.get("prompt_id") == prompt_id:
                                logger.debug(
                                    f"Prompt {prompt_id} is in queue but not executing yet"
                                )
                                return {
                                    "status": "pending",
                                    "progress": 0.0,
                                    "images": [],
                                    "error": None,
                                }
            except Exception as e:
                logger.warning(f"Error parsing queue data: {e}")
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Connection error checking queue: {e}")
        # Don't fail here, try to check history instead
    except Exception as e:
        logger.warning(f"Error checking prompt queue: {e}")

    # Now check the history - use the cached client instead of creating a new one
    api = get_comfy_api_client(api_url)
    return api.check_generation_status(prompt_id, max_retries=max_retries, retry_delay=retry_delay)
