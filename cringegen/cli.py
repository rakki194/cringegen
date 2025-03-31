"""
Command-line interface for CringeGen
"""

import argparse
import logging
import sys

from .commands.furry import add_furry_command
from .commands.info import add_info_commands
from .commands.llm import add_llm_commands
from .commands.lora import add_lora_commands
from .commands.nsfw import add_nsfw_command
from .commands.random_nsfw import add_random_nsfw_command
from .commands.utils import add_utils_commands
from .commands.character import add_character_command
from .commands.xyplot import add_xyplot_command
from .utils.logger import configure_cli_logging, get_logger

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    """Main CLI entry point"""
    # Create parent parser for common options
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parent_parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parent_parser.add_argument("--log-file", type=str, help="Path to log file")
    parent_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parent_parser.add_argument(
        "--comfy-url",
        type=str,
        default="http://127.0.0.1:8188",
        help="URL of the ComfyUI server (default: http://127.0.0.1:8188)",
    )
    parent_parser.add_argument(
        "--show",
        action="store_true",
        help="Open generated images with imv after copying",
    )

    # Create parser
    parser = argparse.ArgumentParser(description="CringeGen CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add commands
    add_furry_command(subparsers, parent_parser)
    add_nsfw_command(subparsers, parent_parser)
    add_random_nsfw_command(subparsers, parent_parser)
    add_info_commands(subparsers, parent_parser)
    add_lora_commands(subparsers, parent_parser)
    add_llm_commands(subparsers, parent_parser)
    add_utils_commands(subparsers, parent_parser)
    add_character_command(subparsers, parent_parser)
    add_xyplot_command(subparsers, parent_parser)

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    configure_cli_logging(args)

    # Export the ComfyUI server URL to an environment variable
    import os

    if hasattr(args, "comfy_url") and args.comfy_url:
        os.environ["COMFY_API_URL"] = args.comfy_url

    # Handle command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    main()
