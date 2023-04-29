#!/bin/bash
# Run full experiment pipeline: train, evaluate, and generate submission.
set -e

CONFIG=${1:-configs/default.yaml}
echo "Using config: $CONFIG"

echo "=========================================="
echo "Step 1: Training U-Net"
echo "=========================================="
python -m src.train --config "$CONFIG"

echo ""
echo "=========================================="
echo "Step 2: Evaluating on Validation Set"
echo "=========================================="
python -m src.evaluate --checkpoint checkpoints/best_model.pth

echo ""
echo "=========================================="
echo "Step 3: Generating Kaggle Submission"
echo "=========================================="
python -m src.predict \
    --checkpoint checkpoints/best_model.pth \
    --test data/test_images.npz \
    --output submission.csv

echo ""
echo "Done! Results saved in results/ and submission.csv"
