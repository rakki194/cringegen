#!/usr/bin/env python3
"""
Command-line interface for cringegen shell completions.
This provides the CLI endpoint that is called by the shell completion scripts.
"""

import argparse
import os
import sys
from typing import List, Tuple

from ..utils.completions import (
    generate_bash_completion,
    generate_zsh_completion,
    get_activation_completions,
    get_checkpoint_completions,
    get_lora_completions,
    get_prompt_completions,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


def format_completions_for_shell(
    completions: List[Tuple[str, str]], shell_type: str = "bash"
) -> str:
    """
    Format completions for the shell.

    Args:
        completions: List of tuples (term, description)
        shell_type: Type of shell (bash or zsh)

    Returns:
        String formatted for the shell
    """
    if shell_type == "zsh":
        # Format for zsh: "term:description" (for _alternative)
        return " ".join([f'"{term}:{desc}"' for term, desc in completions])
    else:
        # Format for bash: just terms
        return " ".join([term for term, _ in completions])


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(description="cringegen shell completion utility")
    subparsers = parser.add_subparsers(dest="command", help="Completion command")

    # Generate completions
    gen_parser = subparsers.add_parser("generate", help="Generate completion scripts")
    gen_parser.add_argument(
        "--output-dir",
        type=str,
        default="./completions",
        help="Output directory for completion scripts",
    )
    gen_parser.add_argument(
        "--type",
        type=str,
        choices=["bash", "zsh", "all"],
        default="all",
        help="Type of completion to generate",
    )

    # LoRA completion
    lora_parser = subparsers.add_parser("loras", help="Get LoRA completions")
    lora_parser.add_argument(
        "partial_name", type=str, nargs="?", default="", help="Partial LoRA name"
    )
    lora_parser.add_argument(
        "--lora-dir", type=str, default=None, help="Directory containing LoRAs"
    )
    lora_parser.add_argument(
        "--min-confidence", type=float, default=0.0, help="Minimum confidence for type detection"
    )
    lora_parser.add_argument(
        "--type",
        type=str,
        choices=["style", "character", "concept", "all"],
        default="all",
        help="Filter by LoRA type",
    )
    lora_parser.add_argument(
        "--shell",
        type=str,
        choices=["bash", "zsh"],
        default="bash",
        help="Shell type for formatting",
    )

    # Checkpoint completion
    checkpoint_parser = subparsers.add_parser("checkpoints", help="Get checkpoint completions")
    checkpoint_parser.add_argument(
        "partial_name", type=str, nargs="?", default="", help="Partial checkpoint name"
    )
    checkpoint_parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Directory containing checkpoints"
    )
    checkpoint_parser.add_argument(
        "--shell",
        type=str,
        choices=["bash", "zsh"],
        default="bash",
        help="Shell type for formatting",
    )

    # Prompt completion
    prompt_parser = subparsers.add_parser("prompts", help="Get prompt completions")
    prompt_parser.add_argument(
        "partial_term", type=str, nargs="?", default="", help="Partial prompt term"
    )
    prompt_parser.add_argument(
        "--lora", type=str, default=None, help="LoRA name for context-aware completions"
    )
    prompt_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint name for context"
    )
    prompt_parser.add_argument(
        "--shell",
        type=str,
        choices=["bash", "zsh"],
        default="bash",
        help="Shell type for formatting",
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        if not args.command:
            parser.print_help()
            return 1

        if args.command == "generate":
            # Generate completion scripts
            output_dir = os.path.abspath(args.output_dir)
            os.makedirs(output_dir, exist_ok=True)

            script_paths = {}

            if args.type in ["bash", "all"]:
                bash_path = os.path.join(output_dir, "cringegen.bash")
                with open(bash_path, "w") as f:
                    f.write(generate_bash_completion())
                script_paths["bash"] = bash_path

            if args.type in ["zsh", "all"]:
                zsh_path = os.path.join(output_dir, "_cringegen")
                with open(zsh_path, "w") as f:
                    f.write(generate_zsh_completion())
                script_paths["zsh"] = zsh_path

            # Print installation instructions
            bash_path = script_paths.get("bash")
            zsh_path = script_paths.get("zsh")

            logger.info("\nCompletion scripts generated. Install with:\n")

            if bash_path:
                logger.info(f"Bash completion:")
                logger.info(f"  mkdir -p ~/.local/share/bash-completion/completions")
                logger.info(f"  cp {bash_path} ~/.local/share/bash-completion/completions/")
                logger.info(f"  # Or system-wide:")
                logger.info(f"  # sudo cp {bash_path} /etc/bash_completion.d/\n")

            if zsh_path:
                logger.info(f"Zsh completion:")
                logger.info(f"  mkdir -p ~/.zsh/completions")
                logger.info(f"  cp {zsh_path} ~/.zsh/completions/")
                logger.info(f"  # Make sure this directory is in your fpath:")
                logger.info(f"  # Add to ~/.zshrc: fpath=(~/.zsh/completions $fpath)")
                logger.info(f"  # Then run: compinit\n")

        elif args.command == "loras":
            # Get LoRA completions
            LoRA_types = None
            if args.type != "all":
                LoRA_types = [args.type]

            completions = get_lora_completions(
                partial_name=args.partial_name,
                lora_dir=args.lora_dir,
                confidence_threshold=args.min_confidence,
                lora_types=LoRA_types,
                include_type_info=True,
            )

            sys.stdout.write(format_completions_for_shell(completions, args.shell))

        elif args.command == "checkpoints":
            # Get checkpoint completions
            completions = get_checkpoint_completions(
                partial_name=args.partial_name, checkpoint_dir=args.checkpoint_dir
            )

            sys.stdout.write(format_completions_for_shell(completions, args.shell))

        elif args.command == "prompts":
            # Get prompt completions
            lora_names = [args.lora] if args.lora else None

            completions = get_prompt_completions(
                partial_term=args.partial_term,
                lora_names=lora_names,
                checkpoint_name=args.checkpoint,
            )

            sys.stdout.write(format_completions_for_shell(completions, args.shell))

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
