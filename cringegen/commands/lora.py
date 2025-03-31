"""
LoRA-related commands for CringeGen
"""

import json
import logging
import os
import sqlite3

from ..utils.lora_metadata import (
    analyze_lora_type,
    analyze_multiple_loras,
    extract_all_lora_metadata,
    extract_lora_metadata,
    get_activation_text,
    get_lora_path,
    get_lora_tag_frequencies,
    get_loras_by_tag,
    get_loras_by_type,
    get_tag_suggestions,
    match_tags,
    suggest_lora_combinations,
)
from ..utils.lora_metadata.extractor import DB_PATH

logger = logging.getLogger(__name__)


def add_lora_commands(subparsers, parent_parser):
    """Add LoRA-related commands to the CLI"""
    # Add extract-metadata command
    extract_metadata_parser = subparsers.add_parser(
        "extract-metadata", help="Extract metadata from a LoRA file", parents=[parent_parser]
    )
    extract_metadata_parser.add_argument("lora", type=str, help="Name or path of the LoRA file")
    extract_metadata_parser.add_argument("--raw", action="store_true", help="Display raw metadata")
    extract_metadata_parser.add_argument(
        "--save", action="store_true", help="Save metadata to database"
    )
    extract_metadata_parser.set_defaults(func=extract_lora_metadata_cmd)

    # Add extract-all-metadata command
    extract_all_metadata_parser = subparsers.add_parser(
        "extract-all-metadata", help="Extract metadata from all LoRA files", parents=[parent_parser]
    )
    extract_all_metadata_parser.add_argument(
        "--force", action="store_true", help="Force extraction even if metadata exists"
    )
    extract_all_metadata_parser.set_defaults(func=extract_all_lora_metadata_cmd)

    # Add search-tags command
    search_tags_parser = subparsers.add_parser(
        "search-tags", help="Search LoRAs by tags", parents=[parent_parser]
    )
    search_tags_parser.add_argument(
        "tags", type=str, help="Comma-separated list of tags to search for"
    )
    search_tags_parser.add_argument(
        "--all", action="store_true", help="Require all tags to match (AND search)"
    )
    search_tags_parser.set_defaults(func=search_lora_tags_cmd)

    # Add tag-suggestions command
    tag_suggestions_parser = subparsers.add_parser(
        "tag-suggestions", help="Get tag suggestions for a LoRA", parents=[parent_parser]
    )
    tag_suggestions_parser.add_argument("lora", type=str, help="Name or path of the LoRA file")
    tag_suggestions_parser.add_argument(
        "--count", type=int, default=10, help="Number of tag suggestions to return"
    )
    tag_suggestions_parser.set_defaults(func=get_lora_tag_suggestions_cmd)

    # Add activation-text command
    activation_text_parser = subparsers.add_parser(
        "activation-text", help="Get activation text for a LoRA", parents=[parent_parser]
    )
    activation_text_parser.add_argument("lora", type=str, help="Name or path of the LoRA file")
    activation_text_parser.add_argument(
        "--count", type=int, default=3, help="Number of activation text suggestions"
    )
    activation_text_parser.set_defaults(func=get_lora_activation_text_cmd)

    # Add search-captions command
    search_captions_parser = subparsers.add_parser(
        "search-captions", help="Search LoRAs by caption text", parents=[parent_parser]
    )
    search_captions_parser.add_argument(
        "text", type=str, help="Text to search for in LoRA captions"
    )
    search_captions_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results to return"
    )
    search_captions_parser.set_defaults(func=search_lora_captions_cmd)

    # Add analyze-lora command
    analyze_lora_parser = subparsers.add_parser(
        "analyze-lora", help="Analyze a LoRA type", parents=[parent_parser]
    )
    analyze_lora_parser.add_argument("lora", type=str, help="Name or path of the LoRA file")
    analyze_lora_parser.set_defaults(func=analyze_lora_cmd)

    # Add analyze-loras command
    analyze_loras_parser = subparsers.add_parser(
        "analyze-loras", help="Analyze multiple LoRAs", parents=[parent_parser]
    )
    analyze_loras_parser.add_argument(
        "loras", type=str, nargs="+", help="Names or paths of LoRA files"
    )
    analyze_loras_parser.set_defaults(func=analyze_loras_cmd)

    # Add suggest-combinations command
    suggest_combinations_parser = subparsers.add_parser(
        "suggest-combinations", help="Suggest LoRA combinations", parents=[parent_parser]
    )
    suggest_combinations_parser.add_argument(
        "--count", type=int, default=5, help="Number of combinations to suggest"
    )
    suggest_combinations_parser.add_argument(
        "--lora-type", type=str, help="Filter by LoRA type (character, style, concept, etc.)"
    )
    suggest_combinations_parser.set_defaults(func=suggest_lora_combinations_cmd)

    return subparsers


def extract_lora_metadata_cmd(args):
    """Extract metadata from a LoRA file"""
    if not os.path.exists(DB_PATH):
        parent_dir = os.path.dirname(DB_PATH)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS lora_metadata ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "name TEXT,"
            "path TEXT,"
            "tags TEXT,"
            "trigger_phrases TEXT,"
            "metadata TEXT,"
            "raw_metadata TEXT,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        conn.commit()
        conn.close()

    try:
        lora_path = get_lora_path(args.lora)
        if not lora_path:
            logger.error(f"LoRA file not found: {args.lora}")
            return

        metadata = extract_lora_metadata(lora_path, save_to_db=args.save)
        if not metadata:
            logger.error(f"Failed to extract metadata from {lora_path}")
            return

        logger.info(f"Metadata for {os.path.basename(lora_path)}:")

        if args.raw:
            # Display raw metadata
            if "raw" in metadata and metadata["raw"]:
                logger.info("Raw metadata:")
                logger.info(json.dumps(metadata["raw"], indent=2))
            else:
                logger.info("No raw metadata available")
        else:
            # Display processed metadata
            if "name" in metadata and metadata["name"]:
                logger.info(f"Name: {metadata['name']}")

            if "tags" in metadata and metadata["tags"]:
                logger.info("Tags:")
                for tag in metadata["tags"]:
                    logger.info(f"  - {tag}")

            if "trigger_phrases" in metadata and metadata["trigger_phrases"]:
                logger.info("Trigger phrases:")
                for phrase in metadata["trigger_phrases"]:
                    logger.info(f"  - {phrase}")

            # Display tag frequencies if available
            if "tag_frequencies" in metadata and metadata["tag_frequencies"]:
                logger.info("Tag frequencies:")
                for tag, freq in sorted(
                    metadata["tag_frequencies"].items(), key=lambda x: x[1], reverse=True
                )[:10]:
                    logger.info(f"  - {tag}: {freq}")

            # Display activation text if available
            if "activation_text" in metadata and metadata["activation_text"]:
                logger.info("Suggested activation text:")
                for text in metadata["activation_text"]:
                    logger.info(f"  - {text}")

        if args.save:
            logger.info(f"Metadata saved to database for {os.path.basename(lora_path)}")

    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")


def extract_all_lora_metadata_cmd(args):
    """Extract metadata from all LoRA files"""
    try:
        results = extract_all_lora_metadata(force=args.force)
        logger.info(f"Extracted metadata from {len(results)} LoRA files:")
        for lora_name, status in results.items():
            if status:
                logger.info(f"  ✓ {lora_name}")
            else:
                logger.info(f"  ✗ {lora_name} (failed)")
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")


def search_lora_tags_cmd(args):
    """Search LoRAs by tags"""
    tags = [tag.strip() for tag in args.tags.split(",")]
    results = get_loras_by_tag(tags, require_all=args.all)

    if results:
        logger.info(f"Found {len(results)} LoRAs matching tags: {', '.join(tags)}")
        for i, (lora_name, match_score) in enumerate(results):
            logger.info(f"{i+1}. {lora_name} (score: {match_score:.2f})")
    else:
        logger.info(f"No LoRAs found matching the tags: {', '.join(tags)}")


def get_lora_tag_suggestions_cmd(args):
    """Get tag suggestions for a LoRA"""
    try:
        lora_path = get_lora_path(args.lora)
        if not lora_path:
            logger.error(f"LoRA file not found: {args.lora}")
            return

        suggestions = get_tag_suggestions(lora_path, args.count)
        if suggestions:
            logger.info(f"Tag suggestions for {os.path.basename(lora_path)}:")
            for tag, score in suggestions:
                logger.info(f"  - {tag} (score: {score:.2f})")
        else:
            logger.info(f"No tag suggestions found for {os.path.basename(lora_path)}")
    except Exception as e:
        logger.error(f"Error getting tag suggestions: {str(e)}")


def get_lora_activation_text_cmd(args):
    """Get activation text for a LoRA"""
    try:
        lora_path = get_lora_path(args.lora)
        if not lora_path:
            logger.error(f"LoRA file not found: {args.lora}")
            return

        activation_texts = get_activation_text(lora_path, args.count)
        if activation_texts:
            logger.info(f"Activation text suggestions for {os.path.basename(lora_path)}:")
            for i, text in enumerate(activation_texts):
                logger.info(f"{i+1}. {text}")
        else:
            logger.info(f"No activation text found for {os.path.basename(lora_path)}")
    except Exception as e:
        logger.error(f"Error getting activation text: {str(e)}")


def search_lora_captions_cmd(args):
    """Search LoRAs by caption text"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check if the lora_metadata table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lora_metadata'")
        if not cursor.fetchone():
            logger.error("Metadata database not initialized. Run extract-metadata command first.")
            return

        # Search for the text in the metadata or raw_metadata columns
        search_term = f"%{args.text}%"
        cursor.execute(
            "SELECT name, path FROM lora_metadata WHERE "
            "metadata LIKE ? OR raw_metadata LIKE ? "
            "LIMIT ?",
            (search_term, search_term, args.limit),
        )

        results = cursor.fetchall()
        conn.close()

        if results:
            logger.info(f"Found {len(results)} LoRAs matching: '{args.text}'")
            for i, (name, path) in enumerate(results):
                logger.info(f"{i+1}. {name} ({path})")
        else:
            logger.info(f"No LoRAs found matching: '{args.text}'")

    except Exception as e:
        logger.error(f"Error searching captions: {str(e)}")


def analyze_lora_cmd(args):
    """Analyze a LoRA type"""
    try:
        lora_path = get_lora_path(args.lora)
        if not lora_path:
            logger.error(f"LoRA file not found: {args.lora}")
            return

        analysis = analyze_lora_type(lora_path)
        if analysis:
            lora_name = os.path.basename(lora_path)
            logger.info(f"Analysis for {lora_name}:")
            logger.info(f"  Type: {analysis['type']}")
            logger.info(f"  Confidence: {analysis['confidence']:.2f}")
            logger.info("  Type probabilities:")
            for lora_type, prob in sorted(
                analysis["type_probabilities"].items(), key=lambda x: x[1], reverse=True
            ):
                logger.info(f"    - {lora_type}: {prob:.2f}")
        else:
            logger.info(f"Could not analyze {os.path.basename(lora_path)}")
    except Exception as e:
        logger.error(f"Error analyzing LoRA: {str(e)}")


def analyze_loras_cmd(args):
    """Analyze multiple LoRAs"""
    try:
        lora_paths = []
        for lora_name in args.loras:
            lora_path = get_lora_path(lora_name)
            if lora_path:
                lora_paths.append(lora_path)
            else:
                logger.warning(f"LoRA file not found: {lora_name}")

        if not lora_paths:
            logger.error("No valid LoRA files found")
            return

        analyses = analyze_multiple_loras(lora_paths)
        if analyses:
            logger.info(f"Analysis for {len(analyses)} LoRAs:")
            for lora_path, analysis in analyses.items():
                lora_name = os.path.basename(lora_path)
                logger.info(f"- {lora_name}:")
                logger.info(f"  Type: {analysis['type']}")
                logger.info(f"  Confidence: {analysis['confidence']:.2f}")
        else:
            logger.info("Could not analyze the specified LoRAs")
    except Exception as e:
        logger.error(f"Error analyzing LoRAs: {str(e)}")


def suggest_lora_combinations_cmd(args):
    """Suggest LoRA combinations"""
    try:
        combinations = suggest_lora_combinations(count=args.count, lora_type=args.lora_type)
        if combinations:
            logger.info(f"Suggested {len(combinations)} LoRA combinations:")
            for i, combo in enumerate(combinations):
                lora_names = [os.path.basename(path) for path in combo["loras"]]
                logger.info(f"{i+1}. {', '.join(lora_names)}")
                logger.info(f"   Score: {combo['score']:.2f}")
                if "types" in combo:
                    logger.info(f"   Types: {', '.join(combo['types'])}")
        else:
            logger.info("Could not generate LoRA combinations")
    except Exception as e:
        logger.error(f"Error suggesting combinations: {str(e)}")
