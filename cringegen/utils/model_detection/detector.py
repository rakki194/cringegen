"""
Model detection utilities for cringegen.

This module provides functions to detect the type and architecture of Stable Diffusion models.
"""


def is_sdxl_model(model_or_checkpoint) -> bool:
    """Determine if a model is SDXL based on model properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        True if the model is detected as SDXL
    """
    # If a string is passed, use name-based detection as fallback
    if isinstance(model_or_checkpoint, str):
        return _is_sdxl_by_name(model_or_checkpoint)
    
    # For ComfyUI model objects
    try:
        # Check if model has the SDXL-specific attributes
        if hasattr(model_or_checkpoint, 'model') and hasattr(model_or_checkpoint.model, 'model_config'):
            config = model_or_checkpoint.model.model_config
            # SDXL has specific dimensions in its configuration
            if hasattr(config, 'unet_config'):
                # SDXL has 4 transformer blocks in middle
                if config.unet_config.get('num_transformer_blocks_middle', 0) == 4:
                    return True
                # Check input channels - SDXL has specific channel counts
                if config.unet_config.get('in_channels', 0) == 4 and config.unet_config.get('out_channels', 0) == 4:
                    if config.unet_config.get('model_channels', 0) == 320:
                        return True
            
            # Another way to detect SDXL: check for specific dimensions in the model
            if hasattr(config, 'model_channels') and config.model_channels == 320:
                if hasattr(config, 'attention_resolutions') and len(config.attention_resolutions) == 4:
                    return True
                    
        # Check model dimensions directly
        if hasattr(model_or_checkpoint, 'conditioner'):
            if hasattr(model_or_checkpoint.conditioner, 'embedders'):
                # SDXL has specific CLIP models
                for embedder in model_or_checkpoint.conditioner.embedders:
                    if hasattr(embedder, 'clip_model'):
                        if getattr(embedder.clip_model, 'hidden_size', 0) == 1280:
                            return True
                            
        # Additional checks for in_channels can help identify SDXL
        if hasattr(model_or_checkpoint, 'in_channels') and model_or_checkpoint.in_channels == 4:
            if hasattr(model_or_checkpoint, 'adm_channels') and model_or_checkpoint.adm_channels > 0:
                return True
    
        return False
        
    except (AttributeError, TypeError):
        # If we can't access model properties, fall back to name-based detection
        if hasattr(model_or_checkpoint, 'name'):
            return _is_sdxl_by_name(model_or_checkpoint.name)
        return False


def _is_sdxl_by_name(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is an SDXL model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be an SDXL model
    """
    sdxl_patterns = [
        "sdxl", 
        "sd-xl", 
        "sd_xl",
        "stableDiffusionXL",
        "stable-diffusion-xl",
        "pixartsigma",
        "xl_",
        "_xl"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    # Make sure we don't match SD3.5 models
    if _is_sd35_by_name(checkpoint_name):
        return False
    
    # RealNoob models are SDXL based
    if "realnoob" in checkpoint_lower:
        return True
    
    # Check for XL in the name - match patterns like animagine-xl, epicrealismXL, etc.
    if ("xl" in checkpoint_lower or "XL" in checkpoint_name):
        # But exclude cases where 'xl' is part of another word
        for word in checkpoint_lower.replace('-', ' ').replace('_', ' ').split():
            if word == "xl" or word.endswith("xl"):
                return True
    
    # Special case for NoobAI models - specifically check for XL in the name
    if ("noob" in checkpoint_lower or "noobai" in checkpoint_lower) and ("xl" in checkpoint_lower or "XL" in checkpoint_name):
        return True
        
    return any(pattern in checkpoint_lower for pattern in sdxl_patterns)


def is_sd15_model(model_or_checkpoint) -> bool:
    """Determine if a model is SD1.5 based on model properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        True if the model is detected as SD1.5
    """
    # If a string is passed, use name-based detection as fallback
    if isinstance(model_or_checkpoint, str):
        return _is_sd15_by_name(model_or_checkpoint)
    
    # For ComfyUI model objects
    try:
        # Check if model has the SD1.5-specific attributes
        if hasattr(model_or_checkpoint, 'model') and hasattr(model_or_checkpoint.model, 'model_config'):
            config = model_or_checkpoint.model.model_config
            
            # SD1.5 has specific dimensions and configuration
            if hasattr(config, 'unet_config'):
                # SD1.5 typically has 320 channels and no transformer blocks in middle
                if config.unet_config.get('model_channels', 0) == 320 and \
                   config.unet_config.get('num_transformer_blocks_middle', 1) == 0:
                    # Make sure it's not SDXL or SD3.5
                    if not is_sdxl_model(model_or_checkpoint) and not is_sd35_model(model_or_checkpoint):
                        return True
            
            # Check model dimensions directly
            if hasattr(config, 'model_channels') and config.model_channels == 320:
                if hasattr(config, 'attention_resolutions') and len(config.attention_resolutions) == 3:
                    # Make sure it's not SD2.0 or SD3.5
                    if not is_sd2_model(model_or_checkpoint) and not is_sd35_model(model_or_checkpoint):
                        return True
                
        # Additional check for CLIP embedding dimensions
        if hasattr(model_or_checkpoint, 'conditioner'):
            if hasattr(model_or_checkpoint.conditioner, 'embedders'):
                for embedder in model_or_checkpoint.conditioner.embedders:
                    if hasattr(embedder, 'clip_model'):
                        # SD1.5 uses 768-dimensional CLIP embeddings
                        if getattr(embedder.clip_model, 'hidden_size', 0) == 768:
                            # Make sure it's not SD2.0 or SD3.5
                            if not is_sd2_model(model_or_checkpoint) and not is_sd35_model(model_or_checkpoint):
                                return True
                            
        return False
        
    except (AttributeError, TypeError):
        # If we can't access model properties, fall back to name-based detection
        if hasattr(model_or_checkpoint, 'name'):
            return _is_sd15_by_name(model_or_checkpoint.name)
        return False


def _is_sd15_by_name(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is an SD1.5 model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be an SD1.5 model
    """
    sd15_patterns = [
        "sd1.5", 
        "sd-1.5", 
        "sd_1.5",
        "sd15",
        "v1-5",
        "v1.5",
        "yiffymix"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    # Exclude SDXL, SD2.0, and SD3.5 models
    if _is_sdxl_by_name(checkpoint_name) or _is_sd2_by_name(checkpoint_name) or _is_sd35_by_name(checkpoint_name):
        return False
        
    # RealNoob models are SDXL, not SD1.5
    if "realnoob" in checkpoint_lower:
        return False
        
    # NoobAI models that aren't XL are typically SD1.5
    if ("noob" in checkpoint_lower or "noobai" in checkpoint_lower) and not ("xl" in checkpoint_lower or "XL" in checkpoint_name):
        return True
        
    if "dreamshaper" in checkpoint_lower and not ("xl" in checkpoint_lower or "XL" in checkpoint_name):
        return True
        
    if "epicrealism" in checkpoint_lower and not ("xl" in checkpoint_lower or "XL" in checkpoint_name):
        return True
        
    return any(pattern in checkpoint_lower for pattern in sd15_patterns)


def is_sd35_model(model_or_checkpoint) -> bool:
    """Determine if a model is SD3.5 based on model properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        True if the model is detected as SD3.5
    """
    # If a string is passed, use name-based detection as fallback
    if isinstance(model_or_checkpoint, str):
        return _is_sd35_by_name(model_or_checkpoint)
    
    # For ComfyUI model objects
    try:
        # Check if model has SD3.5-specific attributes
        if hasattr(model_or_checkpoint, 'model') and hasattr(model_or_checkpoint.model, 'model_config'):
            config = model_or_checkpoint.model.model_config
            
            # SD3.5 has specific architecture identifiers
            if hasattr(config, 'architecture'):
                if 'sd3.5' in config.architecture.lower() or 'sd35' in config.architecture.lower():
                    return True
            
            # Check for SD3.5 UNet configurations
            if hasattr(config, 'unet_config'):
                # SD3.5 has transformer backbone but differs from SDXL
                if config.unet_config.get('use_transformer_backbone', False):
                    # Need to distinguish from SDXL
                    if config.unet_config.get('num_transformers', 0) > 0:
                        return True
                        
                # SD3.5-Turbo has specific parameters
                if config.unet_config.get('in_channels', 0) == 4:
                    if config.unet_config.get('model_channels', 0) == 384:  # Turbo has 384 channels
                        return True
            
            # SD3.5-Large has different dimensions
            if hasattr(config, 'input_size') and config.input_size == [64, 64]:
                if hasattr(config, 'depth') and config.depth > 16:  # Deeper network than SDXL
                    return True
                
        # Check for SD3.5-specific tokenizers
        if hasattr(model_or_checkpoint, 'tokenizer'):
            if hasattr(model_or_checkpoint.tokenizer, 'model_max_length'):
                # SD3.5 uses a different tokenizer with larger context
                if model_or_checkpoint.tokenizer.model_max_length > 4096:
                    return True
                    
        return False
        
    except (AttributeError, TypeError):
        # If we can't access model properties, fall back to name-based detection
        if hasattr(model_or_checkpoint, 'name'):
            return _is_sd35_by_name(model_or_checkpoint.name)
        return False


def _is_sd35_by_name(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is an SD3.5 model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be an SD3.5 model
    """
    sd35_patterns = [
        "sd3.5", 
        "sd-3.5", 
        "sd_3.5",
        "sd35",
        "sd3"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    return any(pattern in checkpoint_lower for pattern in sd35_patterns)


def get_sd35_variant(model_or_checkpoint) -> str:
    """Determine the SD3.5 variant (medium, large, turbo) from a model.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        String identifying the variant: "medium", "large", "turbo", or "unknown"
    """
    # If it's not an SD3.5 model, return None
    if not is_sd35_model(model_or_checkpoint):
        return None
        
    # If a string is passed, use name-based detection
    if isinstance(model_or_checkpoint, str):
        checkpoint_lower = model_or_checkpoint.lower()
        
        if "large" in checkpoint_lower:
            return "large"
        elif "turbo" in checkpoint_lower:
            return "turbo"
        elif "medium" in checkpoint_lower:
            return "medium"
        else:
            return "unknown"
    
    # For ComfyUI model objects
    try:
        # Check model properties to determine variant
        if hasattr(model_or_checkpoint, 'model') and hasattr(model_or_checkpoint.model, 'model_config'):
            config = model_or_checkpoint.model.model_config
            
            # Large variant has higher dimensions
            if hasattr(config, 'unet_config'):
                # Turbo has different channel dimensions
                if config.unet_config.get('model_channels', 0) == 384:
                    return "turbo"
                # Large has more attention heads
                elif config.unet_config.get('num_head_channels', 0) > 80:
                    return "large"
                else:
                    return "medium"
                    
        # Fallback to name-based detection if available
        if hasattr(model_or_checkpoint, 'name'):
            return get_sd35_variant(model_or_checkpoint.name)
            
        return "unknown"
        
    except (AttributeError, TypeError):
        return "unknown"


def is_sd2_model(model_or_checkpoint) -> bool:
    """Determine if a model is SD2.0 based on model properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        True if the model is detected as SD2.0
    """
    # If a string is passed, use name-based detection as fallback
    if isinstance(model_or_checkpoint, str):
        return _is_sd2_by_name(model_or_checkpoint)
    
    # For ComfyUI model objects
    try:
        # Check if model has the SD2.0-specific attributes
        if hasattr(model_or_checkpoint, 'model') and hasattr(model_or_checkpoint.model, 'model_config'):
            config = model_or_checkpoint.model.model_config
            
            # SD2.0 has 768-dimensional embeddings and uses OpenCLIP
            if hasattr(config, 'unet_config'):
                if config.unet_config.get('model_channels', 0) == 320:
                    # SD2.0 uses different attention mechanism
                    if config.unet_config.get('use_spatial_transformer', False):
                        # Make sure it's not SD3.5
                        if not is_sd35_model(model_or_checkpoint):
                            return True
            
            # Another way to check is for the presence of v-prediction
            if hasattr(config, 'parameterization') and config.parameterization == 'v':
                # Make sure it's not SD3.5
                if not is_sd35_model(model_or_checkpoint):
                    return True
                
        # Check CLIP model type - SD2 uses OpenCLIP
        if hasattr(model_or_checkpoint, 'conditioner'):
            if hasattr(model_or_checkpoint.conditioner, 'embedders'):
                for embedder in model_or_checkpoint.conditioner.embedders:
                    if hasattr(embedder, 'clip_model'):
                        # SD2 uses OpenCLIP with 1024-dimensional embeddings
                        if getattr(embedder.clip_model, 'hidden_size', 0) == 1024:
                            # Make sure it's not SD3.5
                            if not is_sd35_model(model_or_checkpoint):
                                return True

        return False
        
    except (AttributeError, TypeError):
        # If we can't access model properties, fall back to name-based detection
        if hasattr(model_or_checkpoint, 'name'):
            return _is_sd2_by_name(model_or_checkpoint.name)
        return False


def _is_sd2_by_name(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is an SD2.0 model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be an SD2.0 model
    """
    sd2_patterns = [
        "sd2", 
        "sd-2", 
        "sd_2",
        "sd2.0",
        "sd-2.0",
        "sd_2.0",
        "stable-diffusion-2",
        "v2-"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    # Exclude SD3.5 models
    if _is_sd35_by_name(checkpoint_name):
        return False
    
    return any(pattern in checkpoint_lower for pattern in sd2_patterns)


def is_flux_model(model_or_checkpoint) -> bool:
    """Determine if a model is a FLUX model based on model properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        True if the model is detected as FLUX
    """
    # If a string is passed, use name-based detection as fallback
    if isinstance(model_or_checkpoint, str):
        return _is_flux_by_name(model_or_checkpoint)
    
    # For ComfyUI model objects
    try:
        # Check if model has FLUX-specific attributes
        if hasattr(model_or_checkpoint, 'model'):
            # FLUX models have specific architecture properties
            if hasattr(model_or_checkpoint.model, 'diffusion_model'):
                diffusion_model = model_or_checkpoint.model.diffusion_model
                # Check for FLUX-specific architecture elements
                if hasattr(diffusion_model, 'flux_blocks'):
                    return True
                
                # Check for FLUX tokens or heads
                if hasattr(diffusion_model, 'flux_tokens') or hasattr(diffusion_model, 'flux_heads'):
                    return True
                    
        # Check for FLUX-specific configuration
        if hasattr(model_or_checkpoint, 'model_config'):
            if hasattr(model_or_checkpoint.model_config, 'architecture'):
                if 'flux' in model_or_checkpoint.model_config.architecture.lower():
                    return True

        return False
        
    except (AttributeError, TypeError):
        # If we can't access model properties, fall back to name-based detection
        if hasattr(model_or_checkpoint, 'name'):
            return _is_flux_by_name(model_or_checkpoint.name)
        return False


def _is_flux_by_name(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is a FLUX model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be a FLUX model
    """
    flux_patterns = [
        "flux", 
        "chroma-unlocked",
        "dcs",
        "lightning"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    return any(pattern in checkpoint_lower for pattern in flux_patterns)


def is_stable_cascade_model(model_or_checkpoint) -> bool:
    """Determine if a model is a Stable Cascade model based on model properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        True if the model is detected as Stable Cascade
    """
    # If a string is passed, use name-based detection as fallback
    if isinstance(model_or_checkpoint, str):
        return _is_stable_cascade_by_name(model_or_checkpoint)
    
    # For ComfyUI model objects
    try:
        # Check if model has Stable Cascade specific attributes
        if hasattr(model_or_checkpoint, 'model'):
            # Stable Cascade models have specific structure
            if hasattr(model_or_checkpoint.model, 'stage'):
                return True
                
            # Check for cascade-specific modules
            if hasattr(model_or_checkpoint.model, 'effnet_encoder') or hasattr(model_or_checkpoint.model, 'prior'):
                return True
                
        # Another way to identify Stable Cascade is through its tokenizer
        if hasattr(model_or_checkpoint, 'tokenizer'):
            if hasattr(model_or_checkpoint.tokenizer, 'stage'):
                return True

        return False
        
    except (AttributeError, TypeError):
        # If we can't access model properties, fall back to name-based detection
        if hasattr(model_or_checkpoint, 'name'):
            return _is_stable_cascade_by_name(model_or_checkpoint.name)
        return False


def _is_stable_cascade_by_name(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is a Stable Cascade model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be a Stable Cascade model
    """
    cascade_patterns = [
        "stable-cascade", 
        "stablecascade",
        "cascade",
        "stagec"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    return any(pattern in checkpoint_lower for pattern in cascade_patterns)


def is_ltx_model(model_or_checkpoint) -> bool:
    """Determine if a model is an LTX model based on model properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        True if the model is detected as LTX
    """
    # If a string is passed, use name-based detection as fallback
    if isinstance(model_or_checkpoint, str):
        return _is_ltx_by_name(model_or_checkpoint)
    
    # For ComfyUI model objects
    try:
        # Check if model has LTX-specific attributes
        if hasattr(model_or_checkpoint, 'model'):
            # LTX models have video-specific properties
            if hasattr(model_or_checkpoint.model, 'temporal_attention'):
                return True
                
            # Check for LTX-specific modules
            if hasattr(model_or_checkpoint.model, 'temporal_encoder'):
                return True
                
        # Check for LTX-specific configuration
        if hasattr(model_or_checkpoint, 'model_config'):
            if hasattr(model_or_checkpoint.model_config, 'architecture'):
                if 'ltx' in model_or_checkpoint.model_config.architecture.lower():
                    return True

        return False
        
    except (AttributeError, TypeError):
        # If we can't access model properties, fall back to name-based detection
        if hasattr(model_or_checkpoint, 'name'):
            return _is_ltx_by_name(model_or_checkpoint.name)
        return False


def _is_ltx_by_name(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is an LTX model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be an LTX model
    """
    ltx_patterns = [
        "ltx", 
        "ltx-video",
        "latent-transformer"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    return any(pattern in checkpoint_lower for pattern in ltx_patterns)


def is_lumina_model(model_or_checkpoint) -> bool:
    """Determine if a model is a Lumina model based on model properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        True if the model is detected as Lumina
    """
    # If a string is passed, use name-based detection as fallback
    if isinstance(model_or_checkpoint, str):
        return _is_lumina_by_name(model_or_checkpoint)
    
    # For ComfyUI model objects
    try:
        # Check if model has Lumina-specific attributes
        if hasattr(model_or_checkpoint, 'model'):
            # Check for Lumina-specific config
            if hasattr(model_or_checkpoint.model, 'config'):
                if hasattr(model_or_checkpoint.model.config, 'architecture'):
                    if 'lumina' in model_or_checkpoint.model.config.architecture.lower():
                        return True
                        
        # Check for Lumina-specific model config
        if hasattr(model_or_checkpoint, 'model_config'):
            if hasattr(model_or_checkpoint.model_config, 'architecture'):
                if 'lumina' in model_or_checkpoint.model_config.architecture.lower():
                    return True

        return False
        
    except (AttributeError, TypeError):
        # If we can't access model properties, fall back to name-based detection
        if hasattr(model_or_checkpoint, 'name'):
            return _is_lumina_by_name(model_or_checkpoint.name)
        return False


def _is_lumina_by_name(checkpoint_name: str) -> bool:
    """Determine if a checkpoint is a Lumina model based on name patterns.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        True if the checkpoint appears to be a Lumina model
    """
    lumina_patterns = [
        "lumina", 
        "illuminati"
    ]
    
    checkpoint_lower = checkpoint_name.lower()
    
    return any(pattern in checkpoint_lower for pattern in lumina_patterns)


def detect_model_architecture(model_or_checkpoint) -> str:
    """Detect the architecture of a model based on its properties or name.
    
    Args:
        model_or_checkpoint: Either a model object or checkpoint name string
        
    Returns:
        String identifying the model architecture (sdxl, sd15, sd2, sd35, flux, stable_cascade, etc.)
    """
    # Check for each architecture in order of specificity
    if is_sd35_model(model_or_checkpoint):
        # For SD3.5, also get the variant
        if isinstance(model_or_checkpoint, str) or hasattr(model_or_checkpoint, 'name'):
            variant = get_sd35_variant(model_or_checkpoint)
            if variant:
                return f"sd35_{variant}"
        return "sd35"
    elif is_stable_cascade_model(model_or_checkpoint):
        return "stable_cascade"
    elif is_flux_model(model_or_checkpoint):
        return "flux"
    elif is_ltx_model(model_or_checkpoint):
        return "ltx"
    elif is_lumina_model(model_or_checkpoint):
        return "lumina"  
    elif is_sdxl_model(model_or_checkpoint):
        return "sdxl"
    elif is_sd2_model(model_or_checkpoint):
        return "sd2"
    elif is_sd15_model(model_or_checkpoint):
        return "sd15"
    
    # If all checks fail, return unknown
    return "unknown"


def get_model_info(checkpoint_name: str) -> tuple:
    """Detect model architecture and family from checkpoint name.
    
    Args:
        checkpoint_name: The name of the checkpoint file
        
    Returns:
        Tuple of (architecture_type, model_family)
        Where architecture_type is one of: "sdxl", "sd15", "sd35", "flux", "ltx", "lumina", etc.
        And model_family is the specific model series like "noob", "yiffymix", "dreamshaper", etc.
    """
    checkpoint_lower = checkpoint_name.lower()
    
    # First detect architecture using our new method
    architecture = detect_model_architecture(checkpoint_name)
    
    # Now detect model family
    family = "unknown"
    
    # Common model families
    family_patterns = {
        "animagine": ["animagine"],
        "dreamshaper": ["dreamshaper"],
        "epicrealism": ["epicrealism"],
        "illustrious": ["illustrious"],
        "juggernaut": ["juggernaut"],
        "noob": ["noob", "realnoob", "noobai"],
        "pony": ["pony"],
        "zavychroma": ["zavychroma"],
        "chroma": ["chroma"],
        "yiffymix": ["yiffymix"],
        "pixartsigma": ["pixartsigma"]
    }
    
    for family_name, patterns in family_patterns.items():
        if any(pattern in checkpoint_lower for pattern in patterns):
            family = family_name
            break
    
    # Handle special cases for SD3.5
    if architecture.startswith("sd35"):
        variant = None
        if "large" in checkpoint_lower:
            variant = "l"
        elif "turbo" in checkpoint_lower:
            variant = "mt"
        elif "medium" in checkpoint_lower:
            variant = "m"
        
        if variant:
            family = f"sd35{variant}"
    
    # Handle vanilla SDXL case
    if architecture == "sdxl" and family == "unknown":
        if any(pattern in checkpoint_lower for pattern in ["base", "1.0"]):
            family = "sdxl"
    
    # Handle LTX and Lumina where architecture = family
    if architecture in ["ltx", "lumina", "flux"] and family == "unknown":
        family = architecture
    
    # If still unknown but we have an architecture, use that for family
    if family == "unknown" and architecture != "unknown":
        family = architecture
        
    # Fix special cases
    if checkpoint_lower == "sd3.5m_turbo.safetensors":
        family = "sd35mt"
        
    # XL models should have SDXL architecture
    if "xl" in checkpoint_lower or "XL" in checkpoint_name:
        if family in ["animagine", "dreamshaper", "epicrealism", "illustrious", "juggernaut", "noob", "pony", "zavychroma"]:
            architecture = "sdxl"
    
    return architecture, family


def test_model_detection(model_names: list) -> dict:
    """Test the model detection functions against a list of model names.
    
    Args:
        model_names: List of model checkpoint names to test
        
    Returns:
        Dictionary with detection results for each model
    """
    results = {}
    
    for name in model_names:
        # Clean up name if it has 'INFO:' prefix
        clean_name = name.strip()
        if clean_name.startswith("INFO:"):
            clean_name = clean_name[5:].strip()
        # Remove leading dash or spaces if present
        clean_name = clean_name.lstrip("- ").strip()
        
        # Skip empty names
        if not clean_name:
            continue
            
        # Get architecture detection
        architecture = detect_model_architecture(clean_name)
        
        # Store detailed detection results
        results[clean_name] = {
            "architecture": architecture,
            "is_sdxl": is_sdxl_model(clean_name),
            "is_sd15": is_sd15_model(clean_name),
            "is_sd2": is_sd2_model(clean_name),
            "is_sd35": is_sd35_model(clean_name),
            "is_flux": is_flux_model(clean_name),
            "is_stable_cascade": is_stable_cascade_model(clean_name),
            "is_ltx": is_ltx_model(clean_name),
            "is_lumina": is_lumina_model(clean_name)
        }
        
        # Add SD3.5 variant if applicable
        if is_sd35_model(clean_name):
            results[clean_name]["sd35_variant"] = get_sd35_variant(clean_name)
            
    return results


# For testing on the command line
if __name__ == "__main__":
    # List of model names to test
    model_list = [
        "animagine-xl-4.0-opt.safetensors",
        "animagine-xl-4.0.safetensors",
        "chroma-unlocked-v13.safetensors",
        "chroma-unlocked-v18.safetensors",
        "dreamshaperXL_lightningDPMSDE.safetensors",
        "epicrealismXL_v9.safetensors",
        "epicrealismXL_vxiAbeast.safetensors",
        "epicrealismXL_vxiiAbea2t.safetensors",
        "illustriousXLV01.safetensors",
        "juggernautXL_version6Rundiffusion.safetensors",
        "ltx-video-2b-v0.9.safetensors",
        "lumina_2.safetensors",
        "lumina_2_model_bf16.safetensors",
        "noobaiXLVpredv10.safetensors",
        "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        "realnoob-e1.safetensors",
        "realnoob-v2.safetensors",
        "realnoob-v3.1.safetensors",
        "realnoob-v3.2.safetensors",
        "realnoob-v3.safetensors",
        "realnoob-v4.safetensors",
        "sd3.5_large_fp8_scaled.safetensors",
        "sd3.5_medium.safetensors",
        "sd3.5m_turbo.safetensors",
        "sd_xl_base_1.0_0.9vae.safetensors",
        "yiffymix_v36.safetensors",
        "yiffymix_v44.safetensors",
        "zavychromaxlV100.vJmm.safetensors"
    ]
    
    # Test the models
    results = test_model_detection(model_list)
    
    # Print results
    for name, result in results.items():
        print(f"Model: {name}")
        print(f"  Architecture: {result['architecture']}")
        print("  Detections:")
        for key, value in result.items():
            if key != 'architecture':
                print(f"    {key}: {value}")
        print() 