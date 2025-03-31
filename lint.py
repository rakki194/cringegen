#!/usr/bin/env python
"""
Run linting tools on the codebase.
"""
import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and print its output."""
    print(f"Running {description}...")
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Always print stdout regardless of success/failure
    if result.stdout:
        print(result.stdout)
        
    if result.returncode != 0:
        print(f"Error running {description}:")
        print(result.stderr)
        return False
    
    print(f"{description} completed successfully.")
    return True


def main():
    """Main function."""
    # Make sure we're in the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Format with isort
    if not run_command(["isort", "cringegen"], "isort"):
        return 1

    # Format with black
    if not run_command(["black", "cringegen"], "black"):
        return 1

    # Check with flake8 - remove --count and --statistics to show full output
    flake8_result = run_command(
        ["flake8", "cringegen"], "flake8"
    )

    if not flake8_result:
        print("Linting failed. Please fix the issues above.")
        return 1

    print("\nAll linting checks completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 