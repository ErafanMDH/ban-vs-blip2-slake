#!/bin/bash
# Usage: sh scripts/compare.sh

echo "Starting Model Comparison..."
python src/compare_models.py \
    --ban_checkpoint results/ban_model.pth \
    --blip_checkpoint results/blip2_model.pth