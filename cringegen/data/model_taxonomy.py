"""
Model-specific data for cringegen.

This file contains model-specific settings for different model architectures
and families, including prompt prefixes, default generation parameters, and 
optimization settings.
"""

from typing import Dict, Any, List, Optional, Tuple, Union

# =============================================================================
# Model Architecture Default Parameters
# =============================================================================

MODEL_ARCHITECTURE_DEFAULTS = {
    # Stable Diffusion 1.5 defaults
    "sd15": {
        "optimal_resolution": (512, 512),  # width, height
        "optimal_pixel_count": 262144,     # 512*512
        "resolution_margin": 0.05,         # 5% margin for different aspect ratios
        "default_steps": 30,
        "default_cfg": 7.5,
        "default_sampler": "euler_a",
        "default_scheduler": "normal",
        "prompt_prefix": "masterpiece, best quality, ",
        "negative_prefix": "lowres, bad anatomy, bad hands, ",
    },
    
    # Stable Diffusion XL defaults
    "sdxl": {
        "optimal_resolution": (1024, 1024),  # width, height
        "optimal_pixel_count": 1048576,      # 1024*1024
        "resolution_margin": 0.05,           # 5% margin for different aspect ratios
        "default_steps": 30,
        "default_cfg": 7.0,
        "default_sampler": "dpmpp_2m",
        "default_scheduler": "karras",
        "prompt_prefix": "masterpiece, best quality, ",
        "negative_prefix": "lowres, bad anatomy, bad hands, ",
    },
    
    # Stable Diffusion 3.5 defaults
    "sd35": {
        "optimal_resolution": (1024, 1024),  # width, height 
        "optimal_pixel_count": 1048576,      # 1024*1024
        "resolution_margin": 0.05,           # 5% margin for different aspect ratios
        "default_steps": 30,
        "default_cfg": 5.0,
        "default_sampler": "dpmpp_2m_sde",
        "default_scheduler": "karras",
        "prompt_prefix": "a photo of ",
        "negative_prefix": "lowres, blurry, ",
    },
    
    # Flux model defaults (Chroma)
    "flux": {
        "optimal_resolution": (512, 512),    # width, height
        "optimal_pixel_count": 262144,       # 512*512
        "resolution_margin": 0.05,           # 5% margin for different aspect ratios
        "default_steps": 30,
        "default_cfg": 7.0,
        "default_sampler": "dpmpp_2m",
        "default_scheduler": "karras",
        "prompt_prefix": "a fantasy artwork of ",
        "negative_prefix": "lowres, bad anatomy, ",
    },
    
    # LTX model defaults
    "ltx": {
        "optimal_resolution": (1024, 576),   # width, height (16:9 aspect ratio)
        "optimal_pixel_count": 589824,       # 1024*576
        "resolution_margin": 0.05,           # 5% margin for different aspect ratios
        "default_steps": 50,
        "default_cfg": 5.0,
        "default_sampler": "ddim",
        "default_scheduler": "karras",
        "prompt_prefix": "a cinematic scene of ",
        "negative_prefix": "blurry, pixelated, ",
    },
    
    # Lumina model defaults
    "lumina": {
        "optimal_resolution": (768, 768),    # width, height
        "optimal_pixel_count": 589824,       # 768*768
        "resolution_margin": 0.05,           # 5% margin for different aspect ratios
        "default_steps": 30,
        "default_cfg": 3.0,
        "default_sampler": "dpmpp_2m",
        "default_scheduler": "karras",
        "prompt_prefix": "a photo of ",
        "negative_prefix": "low quality, ",
    },
}

# Default for unknown model architectures (fallback to SD 1.5)
DEFAULT_ARCHITECTURE = "sd15"

# =============================================================================
# Model Family Parameters
# =============================================================================

# Model-specific prompt prefixes by family
MODEL_FAMILY_PREFIXES = {
    # NoobAI models
    "noob": {
        "prompt_prefix": "masterpiece, best quality, newest, absurdres, highres, ",
        "negative_prefix": "worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, mammal, ambiguous form, feral, semi-anthro, ",
        "trigger_words": ["absurdres", "highres"],
        "default_cfg": 3.5,
        "background_optimizations": {
            "simple background": "",
            "detailed background": "scenery porn, amazing background, ",
            "forest": "scenery porn, amazing background, dense forest, trees, ",
            "city": "scenery porn, amazing background, urban environment, buildings, ",
            "beach": "scenery porn, amazing background, ocean, sand, waves, ",
            "mountain": "scenery porn, amazing background, peaks, rocks, vista, ",
            "desert": "scenery porn, amazing background, sand dunes, arid, ",
            "space": "scenery porn, amazing background, stars, nebula, cosmic, ",
            "castle": "scenery porn, amazing background, medieval, stone walls, fortress, ",
            "landscape": "scenery porn, amazing background, panoramic view, ",
        },
    },
    
    # YiffyMix models
    "yiffymix": {
        "prompt_prefix": "masterpiece, best quality, highly detailed, ",
        "trigger_words": [],
        "default_cfg": 7.5,
    },
    
    # Dreamshaper models
    "dreamshaper": {
        "prompt_prefix": "masterpiece, best quality, ",
        "trigger_words": [],
        "default_cfg": 7.0,
    },
    
    # EpicRealism models
    "epicrealism": {
        "prompt_prefix": "RAW, analog, nikon, film grain, ",
        "trigger_words": ["RAW photo"],
        "default_cfg": 6.5,
    },
    
    # Juggernaut models
    "juggernaut": {
        "prompt_prefix": "professional, highly detailed, ",
        "trigger_words": [],
        "default_cfg": 7.0,
    },
    
    # Pony Diffusion models
    "pony": {
        "prompt_prefix": "anthro, attractive, ",
        "trigger_words": ["anthro"],
        "default_cfg": 7.0,
    },
    
    # Animagine models
    "animagine": {
        "prompt_prefix": "masterpiece, best quality, highly detailed, anime style, ",
        "trigger_words": ["anime style"],
        "default_cfg": 6.5,
    },
    
    # Chroma models
    "chroma": {
        "prompt_prefix": "concept art, fantasy, vibrant colors, ",
        "trigger_words": ["fantasy art"],
        "default_cfg": 7.0,
    },
    
    # ZavyChroma models
    "zavychroma": {
        "prompt_prefix": "epic art, fantasy, high contrast, storybook, ",
        "trigger_words": ["epic fantasy art"],
        "default_cfg": 7.0,
    },
    
    # Illustrious models
    "illustrious": {
        "prompt_prefix": "award-winning, professional, highly detailed, ",
        "trigger_words": [],
        "default_cfg": 7.0,
    },
    
    # SD3.5 Large
    "sd35l": {
        "prompt_prefix": "a photo of ",
        "trigger_words": [],
        "default_cfg": 5.0,
    },
    
    # SD3.5 Medium
    "sd35m": {
        "prompt_prefix": "a photo of ",
        "trigger_words": [],
        "default_cfg": 5.0,
    },
    
    # SD3.5 Medium Turbo
    "sd35mt": {
        "prompt_prefix": "a photo of ",
        "trigger_words": [],
        "default_cfg": 3.0,
        "default_steps": 15,  # Turbo model uses fewer steps
    },
    
    # Vanilla SDXL
    "sdxl": {
        "prompt_prefix": "a photo of ",
        "trigger_words": [],
        "default_cfg": 7.0,
    },
    
    # Vanilla LTX
    "ltx": {
        "prompt_prefix": "a cinematic scene of ",
        "trigger_words": [],
        "default_cfg": 5.0,
    },
    
    # Vanilla Lumina
    "lumina": {
        "prompt_prefix": "a photo of ",
        "trigger_words": [],
        "default_cfg": 3.0,
    },
}

# Default for unknown model families (fallback to generic prefix)
DEFAULT_FAMILY_PREFIX = {
    "prompt_prefix": "masterpiece, best quality, ",
    "trigger_words": [],
    "default_cfg": 7.0,
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_architecture_defaults(architecture: str) -> Dict[str, Any]:
    """Get default parameters for a specific model architecture.
    
    Args:
        architecture: The model architecture type (e.g., 'sdxl', 'sd15')
        
    Returns:
        Dictionary with default parameters for the architecture
    """
    return MODEL_ARCHITECTURE_DEFAULTS.get(architecture.lower(), MODEL_ARCHITECTURE_DEFAULTS[DEFAULT_ARCHITECTURE])

def get_model_family_prefix(family: str) -> Dict[str, Any]:
    """Get prompt prefix and parameters for a specific model family.
    
    Args:
        family: The model family (e.g., 'noob', 'yiffymix')
        
    Returns:
        Dictionary with prefix and parameter information
    """
    return MODEL_FAMILY_PREFIXES.get(family.lower(), DEFAULT_FAMILY_PREFIX)

def get_optimal_resolution_for_model(architecture: str, aspect_ratio: float = 1.0) -> Tuple[int, int]:
    """Calculate the optimal resolution for a model architecture and aspect ratio.
    
    Args:
        architecture: Model architecture type (e.g., 'sdxl', 'sd15')
        aspect_ratio: Width to height ratio (default is 1.0 for square)
        
    Returns:
        Tuple of (width, height) that's optimal for the model
    """
    defaults = get_model_architecture_defaults(architecture)
    target_pixel_count = defaults["optimal_pixel_count"]
    
    # Calculate dimensions based on aspect ratio and target pixels
    width = int((target_pixel_count * aspect_ratio) ** 0.5)
    height = int(width / aspect_ratio)
    
    # Ensure dimensions are divisible by 8 (required by Stable Diffusion)
    width = width - (width % 8)
    height = height - (height % 8)
    
    return (width, height)

def get_combined_prefix(architecture: str, family: str, include_trigger_words: bool = True) -> str:
    """Generate a combined prompt prefix using both architecture and family settings.
    
    Args:
        architecture: Model architecture type (e.g., 'sdxl', 'sd15')
        family: Model family (e.g., 'noob', 'yiffymix')
        include_trigger_words: Whether to include trigger words in the prefix
        
    Returns:
        A combined prompt prefix string
    """
    arch_defaults = get_model_architecture_defaults(architecture)
    family_settings = get_model_family_prefix(family)
    
    # Start with family prefix, which is more specific
    prefix = family_settings["prompt_prefix"]
    
    # Add trigger words if they're not already in the prefix and if requested
    if include_trigger_words and family_settings["trigger_words"]:
        for trigger in family_settings["trigger_words"]:
            if trigger not in prefix:
                prefix += f"{trigger}, "
    
    return prefix

def get_generation_parameters(architecture: str, family: str) -> Dict[str, Any]:
    """Get recommended generation parameters for a specific model architecture and family.
    
    Args:
        architecture: Model architecture type (e.g., 'sdxl', 'sd15')
        family: Model family (e.g., 'noob', 'yiffymix')
        
    Returns:
        Dictionary with recommended generation parameters
    """
    arch_defaults = get_model_architecture_defaults(architecture)
    family_settings = get_model_family_prefix(family)
    
    # Start with architecture defaults
    params = {
        "steps": arch_defaults.get("default_steps", 30),
        "cfg": arch_defaults.get("default_cfg", 7.0),
        "sampler": arch_defaults.get("default_sampler", "euler_a"),
        "scheduler": arch_defaults.get("default_scheduler", "normal"),
    }
    
    # Override with family-specific settings if available
    if "default_steps" in family_settings:
        params["steps"] = family_settings["default_steps"]
    if "default_cfg" in family_settings:
        params["cfg"] = family_settings["default_cfg"]
    if "default_sampler" in family_settings:
        params["sampler"] = family_settings["default_sampler"]
    if "default_scheduler" in family_settings:
        params["scheduler"] = family_settings["default_scheduler"]
    
    return params 