#!/bin/bash
MODEL=$1
CHECKPOINT=$2

if [ -z "$MODEL" ] || [ -z "$CHECKPOINT" ]; then
  echo "Usage: sh scripts/evaluate.sh <model_name> <checkpoint_path>"
  exit 1
fi

# Fixed: evaluate_script.py changed to evaluate.py
python src/evaluate.py --model $MODEL --checkpoint $CHECKPOINT