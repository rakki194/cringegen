"""
Shell completion utilities for cringegen command-line tools.

This module provides functionality for generating shell completions for:
- LoRA names
- Checkpoint names
- Prompt terms based on LoRA keywords
- Common parameters and flags
"""

from .checkpoint_completions import get_checkpoint_completions
from .lora_completions import get_lora_completions, get_lora_types
from .prompt_completions import get_activation_completions, get_prompt_completions
from .shell_generators import generate_bash_completion, generate_zsh_completion

__all__ = [
    "get_lora_completions",
    "get_lora_types",
    "get_checkpoint_completions",
    "get_prompt_completions",
    "get_activation_completions",
    "generate_bash_completion",
    "generate_zsh_completion",
]
