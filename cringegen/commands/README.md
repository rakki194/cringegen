# cringegen Commands

This directory contains the implementation of all cringegen CLI commands, organized into modular files for better maintainability and extensibility.

## Command Modules

- **furry.py**: Commands for generating furry prompts and images
- **nsfw.py**: Commands for generating NSFW furry prompts and images
- **random_nsfw.py**: Commands for generating random NSFW content with style LoRA randomization
- **info.py**: Commands for listing models, samplers, schedulers, and other information
- **lora.py**: Commands for LoRA-related operations
- **llm.py**: Commands for generating content using language models
- **utils.py**: Utility commands such as path resolution and trigger phrase extraction
- **xyplot.py**: Commands for generating XY plot grids that vary parameters across X and Y axes to visualize their impact on generated images

## Command Structure

Each command module follows a common structure:

1. A main function (`add_*_commands`) that adds the module's commands to the CLI
2. Helper functions to add specific subcommands to the parser
3. Command processing functions that handle the actual execution of commands

## Adding New Commands

To add a new command:

1. Create a new module in this directory or extend an existing one
2. Define functions to register your command with the CLI parser
3. Implement the command processing logic
4. Import and register your command in `cli.py`

## Example

```python
def add_example_commands(subparsers):
    """Add example commands to the CLI."""
    add_example_subcommand(subparsers)

def add_example_subcommand(subparsers):
    """Add 'example' subcommand to the CLI."""
    parser = subparsers.add_parser(
        "example",
        help="Example command"
    )
    
    # Add arguments
    parser.add_argument(
        "--example-arg",
        help="An example argument",
        type=str
    )
    
    # Set the function to call when this command is invoked
    parser.set_defaults(func=process_example_command)

def process_example_command(args):
    """Process the 'example' command."""
    # Command implementation here
    pass
```
