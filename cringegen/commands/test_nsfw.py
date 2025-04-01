#!/usr/bin/env python
"""
Test script for the nsfw-furry command's websocket polling functionality
"""

import argparse
import logging
import sys
import time
from argparse import Namespace

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()

# Add the parent directory to the path to import the module
sys.path.insert(0, '.')

from cringegen.commands.nsfw import generate_nsfw_furry

def main():
    """Test the nsfw-furry command's websocket polling functionality"""
    # Create a Namespace object to simulate command-line arguments
    args = Namespace(
        # Basic parameters
        seed=42,
        prompt="a detailed painting of a furry character",
        negative_prompt="low quality, text, watermark",
        no_generate=False,  # We want to generate an image
        copy_output=True,   # We want to copy the image
        count=1,
        increment=False,
        # NSFW parameters
        intensity="mild",
        species=None,
        gender=None,
        colors=None,
        duo=False,
        group=False,
        species2=None,
        gender2=None,
        colors2=None,
        pattern=None,
        pattern2=None,
        no_kinks=False,
        no_art_style=False,
        anthro=True,
        feral=False,
        kinks=None,
        # ComfyUI parameters
        comfy_url="http://localhost:8188",
        comfy_output_dir="/home/kade/toolkit/diffusion/comfy/ComfyUI/output",
        output_dir="output",
        # Image generation parameters
        width=512,
        height=512,
        steps=20,
        cfg=7,
        sampler="dpmpp_2m",
        scheduler="karras",
        checkpoint=None,
        lora=None,
        additional_loras=None,
        lora_weights=None,
        lora_strength=0.7,
        # Remote options (disabled)
        remote=False,
        ssh_host=None,
        ssh_port=22,
        ssh_user=None,
        ssh_key=None,
        # Other options
        show=False,
        # Advanced options
        pag=False,
        pag_scale=3.0,
        pag_sigma_start=-1.0,
        pag_sigma_end=-1.0,
        detail_daemon=False,
        detail_amount=0.2,
        detail_start=0.0,
        detail_end=0.35,
        split_sigmas=False,
        split_first_cfg=7.0,
        split_second_cfg=9.0,
        use_deepshrink=False,
        deepshrink_factor=2.0,
        deepshrink_start=0.0,
        deepshrink_end=0.35,
        deepshrink_gradual=0.6,
        use_zsnr=False,
        use_vpred=False,
    )

    # Run the function
    logger.info("Starting test of nsfw-furry command's websocket polling functionality")
    start_time = time.time()
    prompts = generate_nsfw_furry(args)
    end_time = time.time()
    logger.info(f"Test completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Generated prompts: {prompts}")

if __name__ == "__main__":
    main() 