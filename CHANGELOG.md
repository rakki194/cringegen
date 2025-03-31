# Changelog

All notable changes to the CringeGen project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New `__main__.py` file to allow direct execution as a module (`python -m cringegen`)
- LLM-based caption generation using Ollama
- LLM-based brainstorming functionality
- Random NSFW mode with style LoRA randomization
- Utility commands for model path resolution and LoRA trigger phrase extraction

### Changed

- Refactored CLI architecture to use modular command files
  - Split commands into dedicated files in `cringegen/commands/` directory:
    - `furry.py` for furry generation commands
    - `nsfw.py` for NSFW generation commands
    - `random_nsfw.py` for the new random NSFW mode
    - `info.py` for information and listing commands
    - `utils.py` for utility commands
    - `lora.py` for LoRA-related commands
    - `llm.py` for LLM-related commands
  - Improved maintainability and extensibility of code
  - Added command-specific help messages
- Updated `cli.py` to import and register commands from modular files
- Enhanced logging configuration for better debugging

### Fixed

- Fixed issues with command execution and argument parsing
- Improved error handling in LLM-related functions
- Fixed path resolution for models and LoRAs

## [0.1.0] - 2025-03-30

### Added

- Initial release of CringeGen
- Basic prompt generation for furry and NSFW content
- ComfyUI workflow integration
- Tag/text conversion utilities
- Entity detection and NLP capabilities
