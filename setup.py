#!/usr/bin/env python
"""
Setup script for CringeGen
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cringegen",
    version="0.1.0",
    author="kade",
    author_email="acsipont@gmail.com",
    description="Prompt generation and workflow creation for Stable Diffusion models in ComfyUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rakki194/cringegen",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.13",
    install_requires=[
        "pyyaml",
        "requests",
        "tqdm",
        "pillow",
        "safetensors",
        "piexif",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "cringegen=cringegen.cli:main",
            "cringegen-gallery=cringegen.web.server:run_server",
            "cringegen-completions=cringegen.completion_cli.completion_cli:main",
        ],
    },
) 