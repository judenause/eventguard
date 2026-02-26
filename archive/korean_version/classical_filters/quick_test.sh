#!/bin/bash
# Quick test script for classical filters

echo "==================================="
echo "Classical Filters - Quick Test"
echo "==================================="
echo ""

# Activate conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate Event

# Run quick test on sample data
python3 << 'EOF'
import numpy as np
from filters import create_filter
from utils import compute_event_stream_metrics, format_hw_ops_summary

print("Loading sample data...")
data_path = '/local_data/EventGuard/EventSNN/data/esd/total/test_50/MAH00444_50.npy'
events_full = np.load(data_path)
events = events_full[:10000]  # Use first 10K events for quick test

print(f"Sample: {len(events):,} events")
print(f"GT - Signal: {(events[:, 0] == 0).sum():,}, Noise: {(events[:, 0] != 0).sum():,}\n")

# Test each filter
filters_to_test = ['BAF', 'Refractory']  # Fast filters only for quick test

for filter_name in filters_to_test:
    print(f"Testing {filter_name}...")
    try:
        filter_obj = create_filter(filter_name)
        predictions, hw_ops = filter_obj.filter_events(events)
        
        metrics = compute_event_stream_metrics(events, predictions)
        hw_summary = format_hw_ops_summary(hw_ops, len(events))
        
        print(f"  DA: {metrics['stream_da']:.4f}")
        print(f"  F1: {metrics['stream_f1']:.4f}")
        print(f"  Precision: {metrics['stream_precision']:.4f}")
        print(f"  Recall: {metrics['stream_recall']:.4f}")
        print(f"  Ops/Event: {hw_summary['ops_per_event']:.2f}")
        print()
    except Exception as e:
        print(f"  Error: {e}\n")

print("✅ Quick test completed!")
EOF
