#!/bin/bash
# Test script to verify examples in the README.md file

# Set base parameters
CHECKPOINT="ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
PROMPT="score_9, anthro male fox, detailed background"
OUTPUT_DIR="/tmp/cringegen-test"
OUTPUT_NAME="readme_test"

# Create the output directory
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "Testing basic XYplot example from README.md"
echo "======================================================================"

# Simple test with cfg and steps variation
cd /home/kade/code/cringe.live/cringegen && python -m cringegen xyplot \
  --workflow furry \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --x-param cfg --x-values "4.0,7.0" \
  --y-param steps --y-values "20,30" \
  --width 1024 \
  --height 1024 \
  --output-dir "$OUTPUT_DIR" \
  --output-name "${OUTPUT_NAME}_basic" \
  --remote

# Test with detailed output
echo "Test complete!"
echo "✓ If no errors appeared, the command completed successfully"
echo "Output saved to $OUTPUT_DIR/${OUTPUT_NAME}_basic.png" 