"""
Model detection utilities for cringegen.

This module provides functions to detect model architectures and calculate optimal resolutions.
"""

from .detector import (
    is_sdxl_model,
    is_sd15_model,
    is_sd2_model,
    is_sd35_model,
    is_flux_model,
    is_stable_cascade_model,
    is_ltx_model,
    is_lumina_model,
    detect_model_architecture,
    get_model_info,
    get_sd35_variant,
    test_model_detection
)

from .resolution import (
    is_optimal_resolution,
    get_optimal_resolution,
    get_optimal_resolution_suggestions
)

__all__ = [
    # Model detection functions
    "is_sdxl_model",
    "is_sd15_model",
    "is_sd2_model",
    "is_sd35_model",
    "is_flux_model",
    "is_stable_cascade_model", 
    "is_ltx_model",
    "is_lumina_model",
    "detect_model_architecture",
    "get_model_info",
    "get_sd35_variant",
    "test_model_detection",
    
    # Resolution functions
    "is_optimal_resolution",
    "get_optimal_resolution",
    "get_optimal_resolution_suggestions"
] 