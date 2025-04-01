#!/bin/bash

# Activate the virtual environment
source /home/kade/venv/bin/activate

# Clean up previous test runs
echo "Cleaning up previous test runs..."
rm -rf /tmp/latrans-seeds-test/
mkdir -p /tmp/latrans-seeds-test/debug

# Define the prompt
PROMPT="score_9, score_8_up, score_7_up, rating_explicit, source_furry, anthro male fox, detailed background, bedroom, canine genitalia, presenting sheath"

echo "Running XY Plot with latrans LoRA strength vs seeds..."

cringegen xyplot \
  --log-level INFO \
  --workflow furry \
  --checkpoint ponyDiffusionV6XL_v6StartWithThisOne.safetensors \
  --prompt "$PROMPT" \
  --negative-prompt "human, low quality, worst quality" \
  --x-param seed \
  --y-param lora_weight \
  --x-values "1111,2222,3333,4444" \
  --y-values "0.0,0.33,0.67,1.0" \
  --width 1024 \
  --height 1024 \
  --label-alignment center \
  --font-size 40.0 \
  --horizontal-spacing 0 \
  --vertical-spacing 0 \
  --dump-workflows \
  --output-name latrans_strength_seeds \
  --output-dir /tmp/latrans-seeds-test \
  --lora "pony/by_latrans-v3s1200.safetensors" \
  --pag \
  --pag-scale 3.0 \
  --deepshrink \
  --deepshrink-factor 2.0 \
  --deepshrink-start 0.0 \
  --deepshrink-end 0.35 \
  --deepshrink-gradual 0.6 \
  --remote \
  --ssh-host otter_den \
  --ssh-port 1487 \
  --ssh-user kade

echo "XY plot generation complete. Check the output at /tmp/latrans-seeds-test/latrans_strength_seeds.png" 