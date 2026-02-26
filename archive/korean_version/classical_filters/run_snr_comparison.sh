#!/bin/bash

# evaluate_filters.py wrapper script for SNR comparison
# Usage: ./run_snr_comparison.sh

# Directory Check
# DVSCLEAN
if [ ! -d "/local_data/EventGuard/EventSNN/data/DVSCLEAN_FRAME/fps30/test/test_50" ]; then
    echo "❌ DVSCLEAN fps30 path not found!"
fi

# Output dir
OUT_DIR="./results/snr_comparison"
mkdir -p $OUT_DIR

FILTERS="BAF STCF ONF STCF_Sub"

# --- DVSCLEAN_FRAME (FPS 30, 60, 90, 120) ---

# Loop over FPS
for FPS in 30 60 90 120; do
    echo ">>> Running DVSCLEAN FPS $FPS..."
    
    # Test 50
    DATA_PATH="/local_data/EventGuard/EventSNN/data/DVSCLEAN_FRAME/fps${FPS}/test/test_50"
    if [ -d "$DATA_PATH" ]; then
        python evaluate_filters.py \
            --test_folder custom \
            --custom_path "$DATA_PATH" \
            --fps $FPS \
            --filters $FILTERS \
            --output_dir "$OUT_DIR/dvsclean_fps${FPS}_test50" \
            | tee "$OUT_DIR/dvsclean_fps${FPS}_test50.log"
    else
        echo "Skipping $DATA_PATH (Not found)"
    fi
            
    # Test 100
    DATA_PATH="/local_data/EventGuard/EventSNN/data/DVSCLEAN_FRAME/fps${FPS}/test/test_100"
    if [ -d "$DATA_PATH" ]; then
        python evaluate_filters.py \
            --test_folder custom \
            --custom_path "$DATA_PATH" \
            --fps $FPS \
            --filters $FILTERS \
            --output_dir "$OUT_DIR/dvsclean_fps${FPS}_test100" \
            | tee "$OUT_DIR/dvsclean_fps${FPS}_test100.log"
    else
         echo "Skipping $DATA_PATH (Not found)"
    fi
done


# --- ESD (Total) ---
# Running with FPS 30 assumption (Default Tau)
echo ">>> Running ESD (FPS 30 Assumption)..."

# Test 50
DATA_PATH="/local_data/EventGuard/EventSNN/data/esd/total/test_50"
if [ -d "$DATA_PATH" ]; then
    python evaluate_filters.py \
        --test_folder custom \
        --custom_path "$DATA_PATH" \
        --fps 30 \
        --filters $FILTERS \
        --output_dir "$OUT_DIR/esd_test50" \
        | tee "$OUT_DIR/esd_test50.log"
fi

# Test 100
DATA_PATH="/local_data/EventGuard/EventSNN/data/esd/total/test_100"
if [ -d "$DATA_PATH" ]; then
    python evaluate_filters.py \
        --test_folder custom \
        --custom_path "$DATA_PATH" \
        --fps 30 \
        --filters $FILTERS \
        --output_dir "$OUT_DIR/esd_test100" \
        | tee "$OUT_DIR/esd_test100.log"
fi
