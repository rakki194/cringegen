#!/usr/bin/env python
"""
Test script for the cringegen logging system.
"""

import os
import argparse
from cringegen.utils.logger import get_logger, configure_logging, configure_cli_logging

def test_direct_logging():
    """Test direct use of the logging utilities."""
    # Set up logging
    print("\n=== Testing direct logging configuration ===")
    configure_logging(level="DEBUG")
    
    # Get a logger
    logger = get_logger("test_direct")
    
    # Log at different levels
    print("You should see 4 log messages below:")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

def test_file_logging(log_file):
    """Test logging to a file."""
    # Set up logging to file
    print("\n=== Testing file logging ===")
    configure_logging(level="DEBUG", log_file=log_file)
    
    # Get a logger
    logger = get_logger("test_file")
    
    # Log at different levels
    logger.debug("This is a debug message in the log file")
    logger.info("This is an info message in the log file")
    logger.warning("This is a warning message in the log file")
    logger.error("This is an error message in the log file")
    
    print(f"Logged messages to file: {log_file}")
    print(f"File content:")
    with open(log_file, 'r') as f:
        print(f.read())

def test_cli_logging():
    """Test the CLI logging configuration."""
    print("\n=== Testing CLI logging configuration ===")
    
    # Create mock args objects with different configurations
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Test with debug flag
    print("\n--- Testing with debug=True ---")
    args_debug = MockArgs(debug=True, verbose=False, log_level=None, log_file=None)
    configure_cli_logging(args_debug)
    logger_debug = get_logger("test_cli_debug")
    logger_debug.debug("This is a debug message (should appear)")
    logger_debug.info("This is an info message (should appear)")
    
    # Test with verbose flag
    print("\n--- Testing with verbose=True ---")
    args_verbose = MockArgs(debug=False, verbose=True, log_level=None, log_file=None)
    configure_cli_logging(args_verbose)
    logger_verbose = get_logger("test_cli_verbose")
    logger_verbose.debug("This is a debug message (should NOT appear)")
    logger_verbose.info("This is an info message (should appear)")
    
    # Test with explicit log level
    print("\n--- Testing with log_level=WARNING ---")
    args_warning = MockArgs(debug=True, verbose=True, log_level="WARNING", log_file=None)
    configure_cli_logging(args_warning)
    logger_warning = get_logger("test_cli_warning")
    logger_warning.debug("This is a debug message (should NOT appear)")
    logger_warning.info("This is an info message (should NOT appear)")
    logger_warning.warning("This is a warning message (should appear)")
    logger_warning.error("This is an error message (should appear)")

def main():
    """Main entry point for the test script."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test cringegen logging system")
    parser.add_argument("--log-file", type=str, default="/tmp/cringegen_test_log.txt",
                        help="Log file for testing file logging")
    args = parser.parse_args()
    
    # Run the tests
    test_direct_logging()
    test_file_logging(args.log_file)
    test_cli_logging()
    
    print("\n=== Logging test complete ===")

if __name__ == "__main__":
    main() 