#!/bin/bash
# SNR 평가 스크립트 (DVSCLEAN, ESD)
# 사용: ./run_snr_eval.sh

cd /local_data/EventGuard/EventSNN/code/classical_filters

FILTERS="BAF STCF ONF STCF_Sub"
OUT_DIR="./results/snr_eval"
mkdir -p $OUT_DIR

echo "=========================================="
echo "SNR Evaluation for Classical Filters"
echo "SNR = 10 * log10(TP / FP)"
echo "=========================================="

# --- DVSCLEAN_FRAME ---
for FPS in 30 60 90 120; do
    echo ""
    echo ">>> DVSCLEAN FPS $FPS..."
    
    # Test 50
    DATA_PATH="/local_data/EventGuard/EventSNN/data/DVSCLEAN_FRAME/fps${FPS}/test/test_50"
    if [ -d "$DATA_PATH" ]; then
        echo "    - test_50"
        python evaluate_filters.py \
            --test_folder custom \
            --custom_path "$DATA_PATH" \
            --fps $FPS \
            --filters $FILTERS \
            --output_dir "$OUT_DIR/dvsclean_fps${FPS}_test50" \
            2>&1 | tee "$OUT_DIR/dvsclean_fps${FPS}_test50.log"
    else
        echo "    [SKIP] $DATA_PATH not found"
    fi
    
    # Test 100
    DATA_PATH="/local_data/EventGuard/EventSNN/data/DVSCLEAN_FRAME/fps${FPS}/test/test_100"
    if [ -d "$DATA_PATH" ]; then
        echo "    - test_100"
        python evaluate_filters.py \
            --test_folder custom \
            --custom_path "$DATA_PATH" \
            --fps $FPS \
            --filters $FILTERS \
            --output_dir "$OUT_DIR/dvsclean_fps${FPS}_test100" \
            2>&1 | tee "$OUT_DIR/dvsclean_fps${FPS}_test100.log"
    else
        echo "    [SKIP] $DATA_PATH not found"
    fi
done

# --- ESD ---
echo ""
echo ">>> ESD Total..."

# ESD Test 50
DATA_PATH="/local_data/EventGuard/EventSNN/data/esd/total/test_50"
if [ -d "$DATA_PATH" ]; then
    echo "    - test_50"
    python evaluate_filters.py \
        --test_folder custom \
        --custom_path "$DATA_PATH" \
        --filters $FILTERS \
        --output_dir "$OUT_DIR/esd_test50" \
        2>&1 | tee "$OUT_DIR/esd_test50.log"
fi

# ESD Test 100
DATA_PATH="/local_data/EventGuard/EventSNN/data/esd/total/test_100"
if [ -d "$DATA_PATH" ]; then
    echo "    - test_100"
    python evaluate_filters.py \
        --test_folder custom \
        --custom_path "$DATA_PATH" \
        --filters $FILTERS \
        --output_dir "$OUT_DIR/esd_test100" \
        2>&1 | tee "$OUT_DIR/esd_test100.log"
fi

echo ""
echo "=========================================="
echo "SNR Evaluation Complete!"
echo "Results saved to: $OUT_DIR"
echo "=========================================="
