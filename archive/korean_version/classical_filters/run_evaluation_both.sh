#!/bin/bash
# Evaluate on both test_50 and test_100 datasets

echo "=========================================="
echo "Classical Filters - Full Evaluation"
echo "Both test_50 and test_100 datasets"
echo "=========================================="
echo ""

# Activate conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate Event

# Run evaluation on both datasets
echo "Starting evaluation on BOTH datasets..."
python evaluate_filters.py --test_folder both

# Regenerate summary to ensure all metrics (SNR, NRR, etc.) are included
echo "Regenerating summary reports with extended metrics..."
python regenerate_summary.py

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Check results/ directory for output files"
echo "=========================================="
