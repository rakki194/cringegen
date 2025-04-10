"""
Shell completion script generation for bash and zsh.
"""

import os
from typing import Dict, List, Set

from ...utils.logger import get_logger

logger = get_logger(__name__)

BASH_COMPLETION_TEMPLATE = """
# Bash completion script for cringegen command-line tools
# Generated by cringegen completion utility

_cringegen_completions()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Define all options that take arguments
    local opts_with_args="--lora --additional-loras --checkpoint --prompt --output-dir --steps --cfg-scale --width --height"
    
    # Define all flags (options that don't take arguments)
    local flags="--help --version --debug --tiling --hires-fix --save-metadata"
    
    # Handle specific option completions
    case "${prev}" in
        --lora|--additional-loras)
            # Complete LoRA names - implement this via API
            local loras_output=$(cringegen-completions loras "${cur}" 2>/dev/null)
            COMPREPLY=($(compgen -W "${loras_output}" -- "${cur}"))
            return 0
            ;;
        --checkpoint)
            # Complete checkpoint names - implement this via API
            local checkpoints_output=$(cringegen-completions checkpoints "${cur}" 2>/dev/null)
            COMPREPLY=($(compgen -W "${checkpoints_output}" -- "${cur}"))
            return 0
            ;;
        --prompt)
            # Complete prompt terms - implement this via API
            # Extract the LoRA if specified in the command line
            local lora=""
            for (( i=0; i<${#COMP_WORDS[@]}; i++ )); do
                if [[ "${COMP_WORDS[i]}" == "--lora" && -n "${COMP_WORDS[i+1]}" ]]; then
                    lora="${COMP_WORDS[i+1]}"
                    break
                fi
            done
            
            local prompt_output=$(cringegen-completions prompts "${cur}" --lora "${lora}" 2>/dev/null)
            COMPREPLY=($(compgen -W "${prompt_output}" -- "${cur}"))
            return 0
            ;;
        --output-dir)
            # Use directory completion for output-dir
            COMPREPLY=($(compgen -d -- "${cur}"))
            return 0
            ;;
        *)
            ;;
    esac
    
    # If we're completing an option (starts with -)
    if [[ ${cur} == -* ]]; then
        COMPREPLY=($(compgen -W "${opts_with_args} ${flags}" -- "${cur}"))
        return 0
    fi
    
    # Default to file completion
    COMPREPLY=($(compgen -f -- "${cur}"))
    return 0
}

complete -F _cringegen_completions cringegen
complete -F _cringegen_completions cringegen-generate
complete -F _cringegen_completions cringegen-analyze
"""

ZSH_COMPLETION_TEMPLATE = """
#compdef cringegen cringegen-generate cringegen-analyze

# Zsh completion script for cringegen command-line tools
# Generated by cringegen completion utility

_cringegen_completions() {
    local curcontext="$curcontext" state line
    typeset -A opt_args
    
    _arguments -C \\
        '--lora[Specify primary LoRA]:lora:->loras' \\
        '--additional-loras[Specify additional LoRAs]:additional loras:->loras' \\
        '--checkpoint[Specify checkpoint model]:checkpoint:->checkpoints' \\
        '--prompt[Specify generation prompt]:prompt:->prompts' \\
        '--output-dir[Specify output directory]:directory:_files -/' \\
        '--steps[Number of steps for generation]:steps:' \\
        '--cfg-scale[CFG scale for generation]:cfg scale:' \\
        '--width[Image width]:width:' \\
        '--height[Image height]:height:' \\
        '--tiling[Enable tiling]' \\
        '--hires-fix[Enable hires fix]' \\
        '--save-metadata[Save metadata with generated images]' \\
        '--help[Show help message]' \\
        '--version[Show version]' \\
        '--debug[Enable debug logging]' \\
        '*:filename:_files'
        
    case $state in
        loras)
            local loras_output
            loras_output=$(cringegen-completions loras "$words[$CURRENT]" 2>/dev/null)
            _alternative 'loras:lora:(('"$loras_output"'))'
            ;;
        checkpoints)
            local checkpoints_output
            checkpoints_output=$(cringegen-completions checkpoints "$words[$CURRENT]" 2>/dev/null)
            _alternative 'checkpoints:checkpoint:(('"$checkpoints_output"'))'
            ;;
        prompts)
            local lora=""
            # Look for a --lora option in the command line
            for ((i=1; i<$CURRENT; i++)); do
                if [[ "$words[$i]" == "--lora" && -n "$words[$i+1]" ]]; then
                    lora="$words[$i+1]"
                    break
                fi
            done
            
            local prompt_output
            prompt_output=$(cringegen-completions prompts "$words[$CURRENT]" --lora "$lora" 2>/dev/null)
            _alternative 'prompts:prompt:(('"$prompt_output"'))'
            ;;
    esac
}

_cringegen_completions "$@"
"""


def generate_bash_completion() -> str:
    """
    Generate a bash completion script for cringegen commands.

    Returns:
        Bash completion script as a string
    """
    return BASH_COMPLETION_TEMPLATE.strip()


def generate_zsh_completion() -> str:
    """
    Generate a zsh completion script for cringegen commands.

    Returns:
        Zsh completion script as a string
    """
    return ZSH_COMPLETION_TEMPLATE.strip()


def write_completion_scripts(output_dir: str) -> Dict[str, str]:
    """
    Write completion scripts to output directory.

    Args:
        output_dir: Directory to write scripts to

    Returns:
        Dictionary of script paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Write bash completion
    bash_path = os.path.join(output_dir, "cringegen.bash")
    with open(bash_path, "w") as f:
        f.write(generate_bash_completion())

    # Write zsh completion
    zsh_path = os.path.join(output_dir, "_cringegen")
    with open(zsh_path, "w") as f:
        f.write(generate_zsh_completion())

    return {"bash": bash_path, "zsh": zsh_path}


def print_installation_instructions(script_paths: Dict[str, str]) -> None:
    """
    Print installation instructions for completion scripts.

    Args:
        script_paths: Dictionary of script paths
    """
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
