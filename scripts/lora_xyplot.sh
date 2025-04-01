#!/bin/bash
# Example script for creating an XYplot with multiple LoRAs and varying strengths
# This script demonstrates how to use the 'loras' parameter in xyplot

# Set base parameters
CHECKPOINT="ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
PROMPT="score_9, score_8_up, score_7_up, rating_explicit, source_furry, anthro male fox, detailed background, bedroom, canine genitalia, presenting sheath"
OUTPUT_DIR="/tmp"
OUTPUT_NAME="furry_lora_test"

# X axis: Multiple LoRAs with varying combinations
X_VALUES="pony/by_wolfy-nail-v3s3000.safetensors,pony/by_latrans-v3s1200.safetensors:0.4,pony/cotw-v1s400.safetensors:0.3"
X_VALUES="$X_VALUES;pony/by_wolfy-nail-v3s3000.safetensors:0.5,pony/cotw-v1s400.safetensors:0.4"
X_VALUES="$X_VALUES;pony/by_latrans-v3s1200.safetensors:0.3,pony/cotw-v1s400.safetensors:0.3"

# Y axis: Different model and clip strengths for the existing LoRAs
Y_VALUES="0.2;0.35;0.5"

# Run the xyplot command with DeepShrink enabled
cd /home/kade/code/cringe.live/cringegen && python -m cringegen xyplot \
  --workflow furry \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --x-param loras \
  --x-values "$X_VALUES" \
  --y-param lora_weight \
  --y-values "$Y_VALUES" \
  --width 1024 \
  --height 1024 \
  --output-dir "$OUTPUT_DIR" \
  --output-name "$OUTPUT_NAME" \
  --remote \
  --dump-workflows \
  --split-sigmas 7.0 \
  --pag \
  --deepshrink \
  --seed 1111

# Alternative example with seed variations
echo "====================================================="
echo "Running variation with different seeds..."
echo "====================================================="

# X axis: Various seeds
X_VALUES="1111;2222;3333"

# Y axis: Different LoRA combinations
Y_VALUES="pony/by_wolfy-nail-v3s3000.safetensors"
Y_VALUES="$Y_VALUES;pony/by_wolfy-nail-v3s3000.safetensors,pony/cotw-v1s400.safetensors:0.4"
Y_VALUES="$Y_VALUES;pony/by_wolfy-nail-v3s3000.safetensors,pony/by_latrans-v3s1200.safetensors:0.4,pony/cotw-v1s400.safetensors:0.3"

cd /home/kade/code/cringe.live/cringegen && python -m cringegen xyplot \
  --workflow furry \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --x-param seed \
  --x-values "$X_VALUES" \
  --y-param loras \
  --y-values "$Y_VALUES" \
  --width 1024 \
  --height 1024 \
  --output-dir "$OUTPUT_DIR" \
  --output-name "${OUTPUT_NAME}_seeds" \
  --remote \
  --dump-workflows \
  --split-sigmas 7.0 \
  --pag \
  --deepshrink

echo "Done! Output saved to $OUTPUT_DIR" 