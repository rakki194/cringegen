#!/usr/bin/env python3
"""
Simple test for cringegen imports
"""

import sys
import os

# Print current directory
print(f"Current directory: {os.getcwd()}")

# Add the cringegen module to the path
sys.path.append(os.getcwd())
print(f"Python path: {sys.path}")

# Try to import cringegen module
try:
    import cringegen
    print(f"Successfully imported cringegen")
    print(f"cringegen path: {cringegen.__file__}")
    print(f"cringegen contents: {dir(cringegen)}")
except ImportError as e:
    print(f"Failed to import cringegen: {e}")

# Try to import the specific module
try:
    from cringegen.cringegen.prompt_generation import nlp
    print(f"Successfully imported nlp module")
    print(f"NLP module path: {nlp.__file__}")
    print(f"NLP module contents: {dir(nlp)}")
except ImportError as e:
    print(f"Failed to import nlp module: {e}") 