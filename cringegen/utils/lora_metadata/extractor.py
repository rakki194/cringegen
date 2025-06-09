"""
Functions for extracting metadata from LoRA files
"""

import glob
import json
import logging
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Get logger
from ...utils.logger import get_logger

logger = get_logger(__name__)

# Database path for caching metadata
DB_DIR = os.path.expanduser("~/.cache/cringegen")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "lora_metadata.db")


def ensure_cache_dir():
    """Ensure the cache directory exists"""
    os.makedirs(DB_DIR, exist_ok=True)


def initialize_db():
    """Initialize the SQLite database for LoRA metadata, auto-repairing if tables are missing or corrupt."""
    ensure_cache_dir()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if 'loras' table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='loras'")
    loras_exists = cursor.fetchone() is not None

    if not loras_exists:
        # Drop all known tables to ensure a clean state
        for table in [
            "lora_captions", "captions", "lora_tags", "tags", "loras"
        ]:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        # Log the reset
        logger.warning("Database schema missing or corrupt. Resetting LoRA metadata database.")

    # Create tables (idempotent)
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS loras (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        path TEXT UNIQUE,
        activation_text TEXT,
        trigger_phrases TEXT,
        metadata TEXT,
        captions TEXT,
        training_dataset TEXT,
        model_info TEXT,
        tag_frequencies TEXT,
        last_updated INTEGER
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tag TEXT UNIQUE,
        count INTEGER DEFAULT 1
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS lora_tags (
        lora_id INTEGER,
        tag_id INTEGER,
        count INTEGER DEFAULT 1,
        PRIMARY KEY (lora_id, tag_id),
        FOREIGN KEY (lora_id) REFERENCES loras(id),
        FOREIGN KEY (tag_id) REFERENCES tags(id)
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS captions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        caption TEXT UNIQUE,
        count INTEGER DEFAULT 1
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS lora_captions (
        lora_id INTEGER,
        caption_id INTEGER,
        PRIMARY KEY (lora_id, caption_id),
        FOREIGN KEY (lora_id) REFERENCES loras(id),
        FOREIGN KEY (caption_id) REFERENCES captions(id)
    )
    """
    )

    conn.commit()
    conn.close()


def extract_lora_metadata(lora_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a LoRA file

    Args:
        lora_path: Path to the LoRA file

    Returns:
        Dictionary containing metadata, activation text, and other information
    """
    # Initialize empty result
    result = {
        "name": os.path.basename(lora_path),
        "path": lora_path,
        "activation_text": [],
        "trigger_phrases": [],
        "metadata": {},
        "tags": [],
        "tag_frequencies": {},  # Store the full tag frequency data
        "captions": [],  # Store actual captions used in training
        "training_dataset": {},  # Store dataset information
        "model_info": {},  # Store model information
    }

    try:
        # For safetensors files, the metadata is in the header
        if lora_path.endswith(".safetensors"):
            try:
                # Try using safetensors library first
                import safetensors
                from safetensors.torch import load_file

                logger.debug(f"Reading safetensors file: {lora_path}")

                # Try to load header only
                try:
                    metadata = safetensors.safe_open(lora_path, framework="pt").metadata()
                    if metadata:
                        logger.debug(
                            f"Successfully read safetensors metadata: {len(metadata)} entries"
                        )
                        result["metadata"] = metadata

                        # Process model information
                        model_info = {}
                        for key in metadata:
                            if key.startswith("modelspec.") or key in [
                                "ss_base_model_version",
                                "ss_sd_model_name",
                                "ss_output_name",
                                "ss_network_dim",
                                "ss_network_alpha",
                                "ss_network_module",
                            ]:
                                model_info[key] = metadata[key]
                        result["model_info"] = model_info

                        # Extract activation text if available
                        for key in [
                            "ss_activation_text",
                            "activation_text",
                            "activation text",
                            "ss_prompt",
                        ]:
                            if key in metadata and metadata[key]:
                                result["activation_text"] = metadata[key].strip().split(",")
                                logger.debug(f"Found activation text with key {key}")
                                break

                        # Extract trigger phrases if available
                        for key in [
                            "ss_trigger_phrases",
                            "trigger_phrases",
                            "trigger words",
                            "ss_trigger",
                            "triggerwords",
                        ]:
                            if key in metadata and metadata[key]:
                                result["trigger_phrases"] = metadata[key].strip().split(",")
                                logger.debug(f"Found trigger phrases with key {key}")
                                break

                        # Process dataset information
                        if "ss_dataset_dirs" in metadata:
                            try:
                                result["training_dataset"] = json.loads(metadata["ss_dataset_dirs"])
                                logger.debug(
                                    f"Found training dataset info: {result['training_dataset']}"
                                )
                            except:
                                logger.debug(
                                    f"Failed to parse dataset directories: {metadata['ss_dataset_dirs']}"
                                )

                        # Extract tag frequencies - this contains the actual captions
                        for freq_key in [
                            "ss_tag_frequency",
                            "ss_tags",
                            "tag_frequency",
                            "ss_datasets",
                        ]:
                            if freq_key in metadata and metadata[freq_key]:
                                try:
                                    # Try to parse as JSON
                                    tag_freq = json.loads(metadata[freq_key])
                                    result["tag_frequencies"] = tag_freq
                                    logger.debug(f"Found tag frequencies with key {freq_key}")

                                    # Extract tags and captions from tag frequencies
                                    all_tags, all_captions = (
                                        extract_tags_and_captions_from_frequencies(tag_freq)
                                    )
                                    result["tags"] = all_tags
                                    result["captions"] = all_captions

                                    logger.debug(
                                        f"Extracted {len(all_tags)} tags and {len(all_captions)} captions"
                                    )
                                    break
                                except Exception as e:
                                    logger.debug(f"Failed to parse tag frequencies as JSON: {e}")

                        # If we couldn't extract tags from tag_frequencies, try other fields
                        if not result["tags"]:
                            all_tags = set()
                            for field in [
                                "ss_tag_frequency",
                                "ss_tags",
                                "tags",
                                "tag_frequency",
                                "network_args",
                                "networks",
                            ]:
                                if field in metadata and metadata[field]:
                                    tags = parse_tags(metadata[field])
                                    all_tags.update(tags)
                                    logger.debug(f"Found {len(tags)} tags with key {field}")

                            result["tags"] = sorted(list(all_tags))
                    else:
                        logger.warning(f"No metadata found in safetensors file: {lora_path}")
                except Exception as e:
                    logger.warning(f"Error reading safetensors metadata: {e}")
                    # Try loading the full file as fallback
                    try:
                        logger.debug("Attempting to load full file as fallback")
                        tensors = load_file(lora_path, device="cpu")
                        if tensors.metadata:
                            result["metadata"] = tensors.metadata
                    except:
                        logger.warning(f"Failed to load safetensors file as fallback: {lora_path}")

            except ImportError:
                logger.warning("safetensors library not available, skipping safetensors extraction")

        # For .pt files, we need to load the whole file
        elif lora_path.endswith(".pt"):
            try:
                import torch

                logger.debug(f"Reading PyTorch file: {lora_path}")
                checkpoint = torch.load(lora_path, map_location="cpu")

                # Check various metadata locations
                for key in ["ss_metadata", "metadata", "info", "network_info"]:
                    if key in checkpoint:
                        metadata = checkpoint[key]
                        result["metadata"] = metadata
                        logger.debug(f"Found metadata with key {key}")

                        # Extract tags
                        if isinstance(metadata, dict):
                            for tag_key in ["ss_tag_frequency", "tags", "tag_frequency"]:
                                if tag_key in metadata:
                                    result["tags"] = parse_tags(metadata[tag_key])
                                    logger.debug(f"Found tags with key {tag_key}")
                                    break

                            # Extract activation text
                            for act_key in ["ss_activation_text", "activation_text", "prompt"]:
                                if act_key in metadata:
                                    result["activation_text"] = metadata[act_key].strip().split(",")
                                    logger.debug(f"Found activation text with key {act_key}")
                                    break

                            # Extract trigger phrases
                            for trig_key in ["ss_trigger_phrases", "trigger_phrases", "trigger"]:
                                if trig_key in metadata:
                                    result["trigger_phrases"] = (
                                        metadata[trig_key].strip().split(",")
                                    )
                                    logger.debug(f"Found trigger phrases with key {trig_key}")
                                    break
                        break
            except Exception as e:
                logger.warning(f"Error reading PyTorch checkpoint metadata: {e}")

        # Handle other formats like .ckpt
        elif lora_path.endswith(".ckpt") or lora_path.endswith(".bin"):
            try:
                # Try to interpret as a PyTorch checkpoint
                import torch

                logger.debug(f"Reading checkpoint file: {lora_path}")
                checkpoint = torch.load(lora_path, map_location="cpu")

                # Check various metadata locations similar to .pt files
                for key in ["ss_metadata", "metadata", "info", "network_info"]:
                    if key in checkpoint:
                        metadata = checkpoint[key]
                        result["metadata"] = metadata
                        logger.debug(f"Found metadata with key {key}")

                        # Extract tags
                        if isinstance(metadata, dict):
                            for tag_key in ["ss_tag_frequency", "tags", "tag_frequency"]:
                                if tag_key in metadata:
                                    result["tags"] = parse_tags(metadata[tag_key])
                                    logger.debug(f"Found tags with key {tag_key}")
                                    break

                            # Extract activation text
                            for act_key in ["ss_activation_text", "activation_text", "prompt"]:
                                if act_key in metadata:
                                    result["activation_text"] = metadata[act_key].strip().split(",")
                                    logger.debug(f"Found activation text with key {act_key}")
                                    break

                            # Extract trigger phrases
                            for trig_key in ["ss_trigger_phrases", "trigger_phrases", "trigger"]:
                                if trig_key in metadata:
                                    result["trigger_phrases"] = (
                                        metadata[trig_key].strip().split(",")
                                    )
                                    logger.debug(f"Found trigger phrases with key {trig_key}")
                                    break
                        break
            except Exception as e:
                logger.warning(f"Error reading checkpoint metadata: {e}")

        # If we have a .civitai.info file, read that as well
        info_path = f"{lora_path}.civitai.info"
        if os.path.exists(info_path):
            try:
                logger.debug(f"Reading civitai info file: {info_path}")
                with open(info_path, "r", encoding="utf-8") as f:
                    info_data = json.load(f)

                    # Extract tags from civitai info
                    if "tags" in info_data:
                        result["tags"].extend(info_data["tags"])
                        logger.debug(f"Found {len(info_data['tags'])} tags in civitai info")

                    # Extract trigger words
                    for key in ["trainedWords", "triggerWords", "triggers", "prompts"]:
                        if key in info_data and info_data[key]:
                            if isinstance(info_data[key], list):
                                result["trigger_phrases"].extend(info_data[key])
                                logger.debug(
                                    f"Found {len(info_data[key])} trigger words with key {key}"
                                )
                            elif isinstance(info_data[key], str):
                                result["trigger_phrases"].extend(info_data[key].split(","))
                                logger.debug(f"Found trigger words string with key {key}")

                    # Store model info from civitai
                    model_info = {}
                    for key in ["baseModel", "model", "type", "description"]:
                        if key in info_data:
                            model_info[key] = info_data[key]
                    result["model_info"].update(model_info)

                    # Add entire civitai info to metadata
                    result["metadata"]["civitai_info"] = info_data
            except Exception as e:
                logger.warning(f"Error reading civitai info file: {e}")

    except Exception as e:
        logger.error(f"Error extracting metadata from {lora_path}: {e}")

    # Remove duplicates and clean up
    result["tags"] = sorted(list(set(result["tags"])))
    result["captions"] = sorted(list(set(result["captions"])))
    result["activation_text"] = sorted(
        list(set([t.strip() for t in result["activation_text"] if t and t.strip()]))
    )
    result["trigger_phrases"] = sorted(
        list(set([t.strip() for t in result["trigger_phrases"] if t and t.strip()]))
    )

    logger.debug(
        f"Extracted {len(result['tags'])} tags, {len(result['captions'])} captions, {len(result['activation_text'])} activation texts, and {len(result['trigger_phrases'])} trigger phrases"
    )
    return result


def parse_tags(tag_data: str) -> List[str]:
    """
    Parse tags from metadata field

    Args:
        tag_data: String containing tag data

    Returns:
        List of tags
    """
    tags = []

    # Handle special case if tag_data is already a list
    if isinstance(tag_data, list):
        # Flatten any nested lists
        for item in tag_data:
            if isinstance(item, list):
                tags.extend(item)
            elif isinstance(item, str):
                tags.append(item)
            else:
                # Convert non-string items to strings
                tags.append(str(item))
        return [t.strip() for t in tags if t and isinstance(t, str) and t.strip()]

    # Handle special case if tag_data is a dictionary
    if isinstance(tag_data, dict):
        # Convert all keys to strings
        return [str(k).strip() for k in tag_data.keys() if k and str(k).strip()]

    # Handle if tag_data is not a string
    if not isinstance(tag_data, str):
        logger.warning(f"Unexpected tag data type: {type(tag_data)}")
        return []

    # Now handle string formats
    try:
        # Try parsing as JSON dictionary
        if tag_data.strip().startswith("{") and tag_data.strip().endswith("}"):
            try:
                tag_dict = json.loads(tag_data)
                if isinstance(tag_dict, dict):
                    # Return dictionary keys as tags
                    tags = [str(k).strip() for k in tag_dict.keys() if k and str(k).strip()]
                    logger.debug(f"Parsed {len(tags)} tags from JSON dict")
                    return tags
            except json.JSONDecodeError:
                logger.debug("Failed to parse tag data as JSON dict")
                # Fall through to other parsing methods

        # Try parsing as JSON array
        if tag_data.strip().startswith("[") and tag_data.strip().endswith("]"):
            try:
                tag_list = json.loads(tag_data)
                if isinstance(tag_list, list):
                    # Use list items as tags
                    tags = [str(item) for item in tag_list if item]
                    logger.debug(f"Parsed {len(tags)} tags from JSON array")
                    return [t.strip() for t in tags if t and t.strip()]
            except json.JSONDecodeError:
                logger.debug("Failed to parse tag data as JSON array")
                # Fall through to other parsing methods

        # Try parsing as comma-separated values
        if "," in tag_data:
            tags = [t.strip() for t in tag_data.split(",")]
            logger.debug(f"Parsed {len(tags)} tags from comma-separated values")
            return [t for t in tags if t]

        # Try parsing as space-separated values
        if " " in tag_data:
            tags = [t.strip() for t in tag_data.split()]
            logger.debug(f"Parsed {len(tags)} tags from space-separated values")
            return [t for t in tags if t]

        # If we got here and still don't have tags, use the whole string as one tag
        if tag_data.strip():
            return [tag_data.strip()]

        return []
    except Exception as e:
        logger.warning(f"Error parsing tags: {e}")
        return []


def extract_tags_and_captions_from_frequencies(tag_freq: Dict) -> Tuple[List[str], List[str]]:
    """
    Extract tags and captions from tag frequency data

    Args:
        tag_freq: Tag frequency data

    Returns:
        Tuple of (tags, captions)
    """
    all_tags = set()
    all_captions = set()

    # Process various formats of tag frequency data

    # Format 1: {"dataset_name": {"tag1": count, "tag2": count}}
    if isinstance(tag_freq, dict):
        for dataset, content in tag_freq.items():
            if isinstance(content, dict):
                # Extract tags from dict
                for tag, count in content.items():
                    # Skip very long tags as they are likely captions
                    if len(tag) < 100:
                        all_tags.add(tag)
                    else:
                        # Long values are likely captions
                        all_captions.add(tag)

            # Process arrays within datasets
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "tag_frequency" in item:
                        tag_data = item["tag_frequency"]
                        if isinstance(tag_data, dict):
                            for ds_name, tags_dict in tag_data.items():
                                if isinstance(tags_dict, dict):
                                    for tag, count in tags_dict.items():
                                        # Skip very long tags as they are likely captions
                                        if len(tag) < 100:
                                            all_tags.add(tag)
                                        else:
                                            # Long values are likely captions
                                            all_captions.add(tag)

    # Format 2: [{...}, {...}] (list of dataset objects)
    elif isinstance(tag_freq, list):
        for item in tag_freq:
            if isinstance(item, dict) and "tag_frequency" in item:
                tag_data = item["tag_frequency"]
                if isinstance(tag_data, dict):
                    for ds_name, tags_dict in tag_data.items():
                        if isinstance(tags_dict, dict):
                            for tag, count in tags_dict.items():
                                # Skip very long tags as they are likely captions
                                if len(tag) < 100:
                                    all_tags.add(tag)
                                else:
                                    # Long values are likely captions
                                    all_captions.add(tag)

    return list(all_tags), list(all_captions)


def extract_all_lora_metadata(
    lora_dir: str, force_refresh: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Extract metadata from all LoRA files in a directory

    Args:
        lora_dir: Directory containing LoRA files
        force_refresh: Whether to force refresh the cache

    Returns:
        Dictionary mapping LoRA name to metadata
    """
    # Initialize database if it doesn't exist
    initialize_db()

    # Check if directory exists
    if not os.path.exists(lora_dir):
        logger.error(f"LoRA directory does not exist: {lora_dir}")
        return {}

    logger.info(f"Scanning directory: {lora_dir}")

    # Look for the loras subdirectory if we're given a models directory
    if os.path.exists(os.path.join(lora_dir, "loras")):
        lora_dir = os.path.join(lora_dir, "loras")
        logger.info(f"Found loras subdirectory, using: {lora_dir}")

    # Get all safetensors and pt files with various extensions
    lora_extensions = [
        "**/*.safetensors",
        "**/*.pt",
        "**/*.ckpt",  # Some LoRAs might use .ckpt extension
        "**/*.bin",  # Some LoRAs might use .bin extension
    ]

    # Get all files with supported extensions
    lora_paths = []
    for extension in lora_extensions:
        pattern = os.path.join(lora_dir, extension)
        logger.debug(f"Searching with pattern: {pattern}")
        found = glob.glob(pattern, recursive=True)
        logger.debug(f"Found {len(found)} files with pattern {extension}")
        lora_paths.extend(found)

    # Filter out likely non-LoRA files (checkpoints, VAE models, etc.)
    filtered_paths = []
    checkpoint_keywords = [
        "checkpoint",
        "model",
        "vae",
        "diffusion",
        "unet",
        "text_encoder",
        "clip",
        "controlnet",
        "xl_base",
        "xl1_5",
        "sd1_5",
        "sd15",
        "sd21",
        "sd2_1",
    ]

    for path in lora_paths:
        base_name = os.path.basename(path).lower()
        parent_dir = os.path.basename(os.path.dirname(path)).lower()

        # Skip if in a directory that suggests it's not a LoRA
        if parent_dir in [
            "checkpoints",
            "vae",
            "clip",
            "controlnet",
            "diffusion_models",
            "text_encoders",
            "unet",
        ]:
            logger.debug(f"Skipping file in non-LoRA directory: {path}")
            continue

        # Skip if filename contains obvious checkpoint keywords
        if any(keyword in base_name for keyword in checkpoint_keywords):
            if not "lora" in base_name.lower():  # Unless it specifically has "lora" in the name
                logger.debug(f"Skipping likely non-LoRA file: {path}")
                continue

        filtered_paths.append(path)

    lora_paths = filtered_paths

    # Log the total number of files found
    logger.info(f"Found {len(lora_paths)} potential LoRA files after filtering")

    # If we didn't find any files but there are subdirectories, scan them
    if not lora_paths and os.path.exists(lora_dir):
        for subdir in os.listdir(lora_dir):
            full_subdir = os.path.join(lora_dir, subdir)
            if os.path.isdir(full_subdir):
                logger.info(f"Scanning subdirectory: {full_subdir}")
                for root, dirs, files in os.walk(full_subdir):
                    for file in files:
                        if file.endswith((".safetensors", ".pt", ".ckpt", ".bin")):
                            # Apply same filtering logic
                            if (
                                any(keyword in file.lower() for keyword in checkpoint_keywords)
                                and not "lora" in file.lower()
                            ):
                                continue
                            full_path = os.path.join(root, file)
                            logger.debug(f"Found file via direct scan: {full_path}")
                            lora_paths.append(full_path)
        logger.info(f"Found {len(lora_paths)} files via subdirectory scan")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create a dictionary to store results
    results = {}

    # Process each LoRA
    count = 0
    for lora_path in lora_paths:
        try:
            lora_name = os.path.basename(lora_path)
            logger.debug(f"Processing file {count+1}/{len(lora_paths)}: {lora_name}")

            # Skip some obvious non-LoRAs by size (LoRAs are usually much smaller than checkpoints)
            file_size = os.path.getsize(lora_path)
            if file_size > 2 * 1024 * 1024 * 1024:  # 2GB
                logger.debug(f"Skipping file larger than 2GB, likely not a LoRA: {lora_path}")
                continue

            # Check if we already have this LoRA in the database
            cursor.execute("SELECT id, last_updated FROM loras WHERE path = ?", (lora_path,))
            existing = cursor.fetchone()

            # Get file modification time
            mtime = int(os.path.getmtime(lora_path))

            # Extract metadata if not in database, outdated, or force_refresh
            if not existing or existing[1] < mtime or force_refresh:
                logger.info(f"Extracting metadata for {lora_name}")
                metadata = extract_lora_metadata(lora_path)

                # Skip files that don't appear to be LoRAs (no metadata found)
                if (
                    not metadata["metadata"]
                    and not metadata["tags"]
                    and not metadata["activation_text"]
                    and not metadata["trigger_phrases"]
                ):
                    logger.info(f"Skipping file with no metadata, likely not a LoRA: {lora_path}")
                    continue

                # Convert metadata to JSON for storage
                metadata_json = json.dumps(metadata["metadata"])
                activation_text_json = json.dumps(metadata["activation_text"])
                trigger_phrases_json = json.dumps(metadata["trigger_phrases"])
                captions_json = json.dumps(metadata["captions"])
                training_dataset_json = json.dumps(metadata["training_dataset"])
                model_info_json = json.dumps(metadata["model_info"])
                tag_frequencies_json = json.dumps(metadata["tag_frequencies"])

                # Insert or update in database
                if not existing:
                    cursor.execute(
                        "INSERT INTO loras (name, path, activation_text, trigger_phrases, metadata, captions, training_dataset, model_info, tag_frequencies, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            lora_name,
                            lora_path,
                            activation_text_json,
                            trigger_phrases_json,
                            metadata_json,
                            captions_json,
                            training_dataset_json,
                            model_info_json,
                            tag_frequencies_json,
                            mtime,
                        ),
                    )
                    lora_id = cursor.lastrowid
                else:
                    lora_id = existing[0]
                    cursor.execute(
                        "UPDATE loras SET activation_text = ?, trigger_phrases = ?, metadata = ?, captions = ?, training_dataset = ?, model_info = ?, tag_frequencies = ?, last_updated = ? WHERE id = ?",
                        (
                            activation_text_json,
                            trigger_phrases_json,
                            metadata_json,
                            captions_json,
                            training_dataset_json,
                            model_info_json,
                            tag_frequencies_json,
                            mtime,
                            lora_id,
                        ),
                    )

                    # Remove old tag associations
                    cursor.execute("DELETE FROM lora_tags WHERE lora_id = ?", (lora_id,))
                    cursor.execute("DELETE FROM lora_captions WHERE lora_id = ?", (lora_id,))

                # Process tags
                tag_counter = Counter(metadata["tags"])
                for tag, count in tag_counter.items():
                    # Insert or get tag ID
                    cursor.execute("INSERT OR IGNORE INTO tags (tag) VALUES (?)", (tag,))
                    cursor.execute("SELECT id FROM tags WHERE tag = ?", (tag,))
                    tag_id = cursor.fetchone()[0]

                    # Update tag count
                    cursor.execute("UPDATE tags SET count = count + 1 WHERE id = ?", (tag_id,))

                    # Add association
                    cursor.execute(
                        "INSERT INTO lora_tags (lora_id, tag_id, count) VALUES (?, ?, ?)",
                        (lora_id, tag_id, count),
                    )

                # Process captions
                for caption in metadata["captions"]:
                    if caption and len(caption) > 5:  # Skip very short captions
                        # Insert or get caption ID
                        cursor.execute(
                            "INSERT OR IGNORE INTO captions (caption) VALUES (?)", (caption,)
                        )
                        cursor.execute("SELECT id FROM captions WHERE caption = ?", (caption,))
                        caption_id = cursor.fetchone()[0]

                        # Update caption count
                        cursor.execute(
                            "UPDATE captions SET count = count + 1 WHERE id = ?", (caption_id,)
                        )

                        # Add association
                        cursor.execute(
                            "INSERT INTO lora_captions (lora_id, caption_id) VALUES (?, ?)",
                            (lora_id, caption_id),
                        )

                # Add to results
                results[lora_name] = metadata
            else:
                # Load from database
                lora_id = existing[0]
                cursor.execute(
                    "SELECT name, path, activation_text, trigger_phrases, metadata, captions, training_dataset, model_info, tag_frequencies FROM loras WHERE id = ?",
                    (lora_id,),
                )
                row = cursor.fetchone()

                # Get tags for this LoRA
                cursor.execute(
                    "SELECT t.tag FROM tags t JOIN lora_tags lt ON t.id = lt.tag_id WHERE lt.lora_id = ?",
                    (lora_id,),
                )
                tags = [r[0] for r in cursor.fetchall()]

                # Get captions for this LoRA
                cursor.execute(
                    "SELECT c.caption FROM captions c JOIN lora_captions lc ON c.id = lc.caption_id WHERE lc.lora_id = ?",
                    (lora_id,),
                )
                captions = [r[0] for r in cursor.fetchall()]

                # Create metadata object
                metadata = {
                    "name": row[0],
                    "path": row[1],
                    "activation_text": json.loads(row[2]),
                    "trigger_phrases": json.loads(row[3]),
                    "metadata": json.loads(row[4]),
                    "captions": captions if captions else json.loads(row[5]),
                    "training_dataset": json.loads(row[6]),
                    "model_info": json.loads(row[7]),
                    "tag_frequencies": json.loads(row[8]),
                    "tags": tags,
                }

                # Add to results
                results[lora_name] = metadata

            count += 1
            if count % 10 == 0:
                # Commit every 10 files to avoid long transactions
                conn.commit()

        except Exception as e:
            logger.error(f"Error processing LoRA file {lora_path}: {e}")

    # Commit changes and close
    conn.commit()
    conn.close()

    logger.info(f"Successfully processed {count} LoRA files")
    return results


def get_lora_tag_frequencies(min_count: int = 1) -> Dict[str, int]:
    """
    Get tag frequencies from all LoRAs

    Args:
        min_count: Minimum count for a tag to be included

    Returns:
        Dictionary mapping tag to frequency
    """
    # Initialize database if it doesn't exist
    initialize_db()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get tag frequencies
    cursor.execute("SELECT tag, count FROM tags WHERE count >= ? ORDER BY count DESC", (min_count,))
    tag_counts = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    return tag_counts


def get_lora_path(name: str) -> Optional[str]:
    """
    Get the full path to a LoRA by name

    Args:
        name: Name of the LoRA

    Returns:
        Full path to the LoRA, or None if not found
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Try exact match
    cursor.execute("SELECT path FROM loras WHERE name = ?", (name,))
    result = cursor.fetchone()

    # Try case-insensitive match
    if not result:
        cursor.execute("SELECT path FROM loras WHERE lower(name) = lower(?)", (name,))
        result = cursor.fetchone()

    conn.close()

    return result[0] if result else None


def get_loras_by_tag(tag: str, limit: int = 0) -> List[str]:
    """
    Get LoRAs that have a specific tag

    Args:
        tag: Tag to search for
        limit: Maximum number of results (0 for all)

    Returns:
        List of LoRA names
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Query for LoRAs with this tag
    cursor.execute(
        """
        SELECT l.name 
        FROM loras l
        JOIN lora_tags lt ON l.id = lt.lora_id
        JOIN tags t ON lt.tag_id = t.id
        WHERE t.tag = ?
        ORDER BY lt.count DESC
    """,
        (tag,),
    )

    if limit > 0:
        results = [row[0] for row in cursor.fetchmany(limit)]
    else:
        results = [row[0] for row in cursor.fetchall()]

    conn.close()

    return results
