#!/bin/bash
# Full evaluation script for classical filters on test datasets

echo "=========================================="
echo "Classical Filters - Full Evaluation"
echo "=========================================="
echo ""

# Activate conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate Event

# Run evaluation on test_50
echo "Starting evaluation on test_50 dataset..."
python evaluate_filters.py --test_folder test_50

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Check results/ directory for output files"
echo "=========================================="
