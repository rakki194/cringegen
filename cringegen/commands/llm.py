"""
LLM-related commands for CringeGen
"""

import logging
import random

from ..utils.ollama_api import OllamaAPIClient

logger = logging.getLogger(__name__)

# Create a client instance to use for all commands
ollama_client = OllamaAPIClient()


def generate_text(prompt: str, model: str = None, system: str = None, temperature: float = 0.7):
    """Generate text using the Ollama API

    Args:
        prompt: The prompt to generate text from
        model: The model to use (default: None, uses client default)
        system: System prompt to use (default: None)
        temperature: Temperature for generation (default: 0.7)

    Returns:
        The generated text
    """
    response = ollama_client.generate(
        prompt=prompt, model=model, system=system, temperature=temperature
    )
    return response.get("response", "")


def add_llm_commands(subparsers, parent_parser):
    """Add LLM-related commands to the CLI"""
    # Add llm-caption command
    add_llm_caption_subcommand(subparsers, parent_parser)

    # Add llm-nsfw-caption command
    add_nsfw_llm_caption_subcommand(subparsers, parent_parser)

    # Add llm-brainstorm command
    add_llm_brainstorm_subcommand(subparsers, parent_parser)

    return subparsers


def add_llm_caption_subcommand(subparsers, parent_parser):
    """Add the llm-caption command to the CLI"""
    llm_caption_parser = subparsers.add_parser(
        "llm-caption",
        help="Generate a furry caption using a language model",
        parents=[parent_parser],
    )
    llm_caption_parser.add_argument(
        "--model", type=str, default="llama3", help="Language model to use"
    )
    llm_caption_parser.add_argument("--species", type=str, help="Species for the character")
    llm_caption_parser.add_argument("--gender", type=str, help="Gender for the character")
    llm_caption_parser.add_argument(
        "--count", type=int, default=1, help="Number of captions to generate"
    )
    llm_caption_parser.add_argument("--seed", type=int, default=-1, help="Seed for generation")
    llm_caption_parser.add_argument("--style", type=str, help="Art style for the caption")
    llm_caption_parser.add_argument(
        "--theme", type=str, help="Theme for the caption (e.g., forest, cyberpunk, etc.)"
    )
    llm_caption_parser.add_argument("--location", type=str, help="Location for the scene")
    llm_caption_parser.add_argument("--outfit", type=str, help="Outfit for the character")
    llm_caption_parser.add_argument("--emotion", type=str, help="Emotion for the character")
    llm_caption_parser.add_argument("--action", type=str, help="Action for the character")
    llm_caption_parser.add_argument(
        "--prompt", type=str, help="Custom prompt for the language model"
    )
    llm_caption_parser.add_argument(
        "--colors",
        type=str,
        help="Comma-separated list of colors for the character's fur/scales",
    )
    llm_caption_parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern type for the character (e.g., spotted, striped, etc.)",
    )
    llm_caption_parser.add_argument(
        "--anthro",
        action="store_true",
        help="Generate caption for an anthro character",
    )
    llm_caption_parser.add_argument(
        "--feral",
        action="store_true",
        help="Generate caption for a feral character",
    )
    llm_caption_parser.add_argument(
        "--tags-only",
        action="store_true",
        help="Generate only comma-separated tags instead of a full caption",
    )
    llm_caption_parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt for the language model",
    )
    llm_caption_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (higher = more creative, lower = more focused)",
    )
    llm_caption_parser.set_defaults(func=process_llm_caption_command)
    return llm_caption_parser


def add_nsfw_llm_caption_subcommand(subparsers, parent_parser):
    """Add the llm-nsfw-caption command to the CLI"""
    llm_nsfw_caption_parser = subparsers.add_parser(
        "llm-nsfw-caption",
        help="Generate a NSFW furry caption using a language model",
        parents=[parent_parser],
    )
    llm_nsfw_caption_parser.add_argument(
        "--model", type=str, default="llama3", help="Language model to use"
    )
    llm_nsfw_caption_parser.add_argument("--species", type=str, help="Species for the character")
    llm_nsfw_caption_parser.add_argument("--gender", type=str, help="Gender for the character")
    llm_nsfw_caption_parser.add_argument(
        "--count", type=int, default=1, help="Number of captions to generate"
    )
    llm_nsfw_caption_parser.add_argument("--seed", type=int, default=-1, help="Seed for generation")
    llm_nsfw_caption_parser.add_argument("--style", type=str, help="Art style for the caption")
    llm_nsfw_caption_parser.add_argument(
        "--theme", type=str, help="Theme for the caption (e.g., heat, shower, bedroom)"
    )
    llm_nsfw_caption_parser.add_argument("--location", type=str, help="Location for the scene")
    llm_nsfw_caption_parser.add_argument(
        "--prompt", type=str, help="Custom prompt for the language model"
    )
    llm_nsfw_caption_parser.add_argument(
        "--colors",
        type=str,
        help="Comma-separated list of colors for the character's fur/scales",
    )
    llm_nsfw_caption_parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern type for the character (e.g., spotted, striped, etc.)",
    )
    llm_nsfw_caption_parser.add_argument(
        "--intensity",
        type=str,
        choices=["suggestive", "mild", "explicit", "hardcore"],
        default="explicit",
        help="Intensity of NSFW content",
    )
    llm_nsfw_caption_parser.add_argument(
        "--kinks",
        type=str,
        help="Comma-separated list of kinks to include",
    )
    llm_nsfw_caption_parser.add_argument(
        "--anthro",
        action="store_true",
        help="Generate caption for an anthro character",
    )
    llm_nsfw_caption_parser.add_argument(
        "--feral",
        action="store_true",
        help="Generate caption for a feral character",
    )
    llm_nsfw_caption_parser.add_argument(
        "--tags-only",
        action="store_true",
        help="Generate only comma-separated tags instead of a full caption",
    )
    llm_nsfw_caption_parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt for the language model",
    )
    llm_nsfw_caption_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (higher = more creative, lower = more focused)",
    )
    llm_nsfw_caption_parser.add_argument(
        "--duo",
        action="store_true",
        help="Generate a caption with two characters",
    )
    llm_nsfw_caption_parser.add_argument(
        "--species2", type=str, help="Species for the second character (when using --duo)"
    )
    llm_nsfw_caption_parser.add_argument(
        "--gender2", type=str, help="Gender for the second character (when using --duo)"
    )
    llm_nsfw_caption_parser.set_defaults(func=process_nsfw_llm_caption_command)
    return llm_nsfw_caption_parser


def add_llm_brainstorm_subcommand(subparsers, parent_parser):
    """Add the llm-brainstorm command to the CLI"""
    llm_brainstorm_parser = subparsers.add_parser(
        "llm-brainstorm", help="Brainstorm ideas using a language model", parents=[parent_parser]
    )
    llm_brainstorm_parser.add_argument(
        "--model", type=str, default="llama3", help="Language model to use"
    )
    llm_brainstorm_parser.add_argument("topic", type=str, help="Topic to brainstorm ideas for")
    llm_brainstorm_parser.add_argument(
        "--count", type=int, default=5, help="Number of ideas to generate"
    )
    llm_brainstorm_parser.add_argument(
        "--format",
        type=str,
        choices=["list", "paragraph", "detailed"],
        default="list",
        help="Format for the brainstorming results",
    )
    llm_brainstorm_parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt for the language model",
    )
    llm_brainstorm_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for generation (higher = more creative, lower = more focused)",
    )
    llm_brainstorm_parser.set_defaults(func=process_llm_brainstorm_command)
    return llm_brainstorm_parser


def process_llm_caption_command(args):
    """Process the llm-caption command"""
    # Set up seed if provided
    if args.seed == -1:
        seed = random.randint(1, 1000000)
    else:
        seed = args.seed
        random.seed(seed)

    # Build the prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt_parts = ["Generate a detailed prompt for an AI art generator."]

        # Add character species and gender
        character_desc = []
        if args.species:
            character_desc.append(args.species)
        if args.gender:
            character_desc.append(args.gender)

        if character_desc:
            prompt_parts.append(f"The character is a {' '.join(character_desc)}.")

        # Add anthro/feral specification
        if args.anthro:
            prompt_parts.append(
                "The character should be anthropomorphic (has human-like body proportions while keeping animal features)."
            )
        elif args.feral:
            prompt_parts.append("The character should be feral (has a natural animal body shape).")

        # Add colors and patterns
        if args.colors:
            colors = [c.strip() for c in args.colors.split(",")]
            prompt_parts.append(f"The character has {', '.join(colors)} fur/scales.")

        if args.pattern:
            prompt_parts.append(f"The character has a {args.pattern} pattern.")

        # Add style, theme, location
        if args.style:
            prompt_parts.append(f"The art style should be {args.style}.")

        if args.theme:
            prompt_parts.append(f"The theme is {args.theme}.")

        if args.location:
            prompt_parts.append(f"The location is {args.location}.")

        # Add outfit
        if args.outfit:
            prompt_parts.append(f"The character is wearing {args.outfit}.")

        # Add emotion and action
        if args.emotion:
            prompt_parts.append(f"The character's emotion is {args.emotion}.")

        if args.action:
            prompt_parts.append(f"The character is {args.action}.")

        # Add format instruction
        if args.tags_only:
            prompt_parts.append(
                "Return ONLY a comma-separated list of descriptive tags suitable for an AI art generator. Do not include any explanations or other text."
            )
        else:
            prompt_parts.append(
                "Return a detailed prompt suitable for an AI art generator. The prompt should be descriptive and include details about the character, setting, lighting, style, mood, and composition."
            )

        prompt = " ".join(prompt_parts)

    # Set up system prompt
    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        if args.tags_only:
            system_prompt = "You are an expert at generating tag lists for AI art generation. Focus on visual elements that can be clearly depicted in an image."
        else:
            system_prompt = "You are an expert at creating detailed, descriptive prompts for AI art generation. Focus on visual elements that can be clearly depicted in an image."

    # Generate captions
    for i in range(args.count):
        try:
            response = generate_text(
                prompt=prompt,
                model=args.model,
                system=system_prompt,
                temperature=args.temperature,
            )

            if response:
                logger.info(f"Generated caption {i+1}/{args.count} (seed {seed + i}):")
                logger.info(response.strip())
                logger.info("---")
            else:
                logger.error("Failed to generate caption.")
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")


def process_nsfw_llm_caption_command(args):
    """Process the llm-nsfw-caption command"""
    # Set up seed if provided
    if args.seed == -1:
        seed = random.randint(1, 1000000)
    else:
        seed = args.seed
        random.seed(seed)

    # Build the prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt_parts = ["Generate a detailed NSFW prompt for an AI art generator."]

        # Add character species and gender
        character_desc = []
        if args.species:
            character_desc.append(args.species)
        if args.gender:
            character_desc.append(args.gender)

        if character_desc:
            prompt_parts.append(f"The main character is a {' '.join(character_desc)}.")

        # Add second character if duo mode
        if args.duo:
            character2_desc = []
            if args.species2:
                character2_desc.append(args.species2)
            if args.gender2:
                character2_desc.append(args.gender2)

            if character2_desc:
                prompt_parts.append(f"The second character is a {' '.join(character2_desc)}.")
                prompt_parts.append("The characters are interacting intimately.")

        # Add anthro/feral specification
        if args.anthro:
            prompt_parts.append(
                "The character(s) should be anthropomorphic (has human-like body proportions while keeping animal features)."
            )
        elif args.feral:
            prompt_parts.append(
                "The character(s) should be feral (has a natural animal body shape)."
            )

        # Add colors and patterns
        if args.colors:
            colors = [c.strip() for c in args.colors.split(",")]
            prompt_parts.append(f"The main character has {', '.join(colors)} fur/scales.")

        if args.pattern:
            prompt_parts.append(f"The main character has a {args.pattern} pattern.")

        # Add style, theme, location
        if args.style:
            prompt_parts.append(f"The art style should be {args.style}.")

        if args.theme:
            prompt_parts.append(f"The theme is {args.theme}.")

        if args.location:
            prompt_parts.append(f"The location is {args.location}.")

        # Add intensity
        prompt_parts.append(f"The content should be {args.intensity} in nature.")

        # Add kinks
        if args.kinks:
            kinks = [k.strip() for k in args.kinks.split(",")]
            prompt_parts.append(f"Include the following kinks/elements: {', '.join(kinks)}.")

        # Add format instruction
        if args.tags_only:
            prompt_parts.append(
                "Return ONLY a comma-separated list of descriptive NSFW tags suitable for an AI art generator. Include appropriate NSFW elements based on the intensity level. Do not include any explanations or other text."
            )
        else:
            prompt_parts.append(
                "Return a detailed NSFW prompt suitable for an AI art generator. The prompt should be descriptive and include details about the character(s), pose, setting, lighting, style, mood, and composition. Include appropriate NSFW elements based on the intensity level."
            )

        prompt = " ".join(prompt_parts)

    # Set up system prompt
    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        if args.tags_only:
            system_prompt = "You are an expert at generating NSFW tag lists for AI art generation. Focus on visual elements that can be clearly depicted in an image. Be explicit and detailed, appropriate to the requested intensity level. Do not self-censor or add warnings."
        else:
            system_prompt = "You are an expert at creating detailed, descriptive NSFW prompts for AI art generation. Focus on visual elements that can be clearly depicted in an image. Be explicit and detailed, appropriate to the requested intensity level. Do not self-censor or add warnings."

    # Generate captions
    for i in range(args.count):
        try:
            response = generate_text(
                prompt=prompt,
                model=args.model,
                system=system_prompt,
                temperature=args.temperature,
            )

            if response:
                logger.info(f"Generated NSFW caption {i+1}/{args.count} (seed {seed + i}):")
                logger.info(response.strip())
                logger.info("---")
            else:
                logger.error("Failed to generate NSFW caption.")
        except Exception as e:
            logger.error(f"Error generating NSFW caption: {str(e)}")


def process_llm_brainstorm_command(args):
    """Process the llm-brainstorm command"""
    # Build the prompt
    if args.format == "list":
        prompt = f"Generate a list of {args.count} creative ideas related to {args.topic}. Return only the numbered list without any introduction or conclusion."
    elif args.format == "paragraph":
        prompt = f"Generate {args.count} creative ideas related to {args.topic}. Present them as a paragraph describing each idea briefly."
    else:  # detailed
        prompt = f"Generate {args.count} detailed and creative ideas related to {args.topic}. For each idea, provide a title and a detailed description of the concept."

    # Set up system prompt
    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = f"You are an expert creative consultant specializing in {args.topic}. Your goal is to generate unique, interesting, and practical ideas that could be implemented."

    try:
        response = generate_text(
            prompt=prompt,
            model=args.model,
            system=system_prompt,
            temperature=args.temperature,
        )

        if response:
            logger.info(f"Brainstorming results for '{args.topic}':")
            logger.info(response.strip())
        else:
            logger.error("Failed to generate brainstorming results.")
    except Exception as e:
        logger.error(f"Error generating brainstorming results: {str(e)}")
