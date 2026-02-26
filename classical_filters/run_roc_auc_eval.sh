#!/bin/bash
# ROC-AUC 평가 스크립트 (driving, hotelbar)
# 사용: ./run_roc_auc_eval.sh

cd /local_data/EventGuard/EventSNN/code/classical_filters

FILTERS="BAF STCF ONF STCF_Sub"
OUT_DIR="./results/roc_auc"
mkdir -p $OUT_DIR

echo "=========================================="
echo "ROC-AUC Evaluation for Classical Filters"
echo "=========================================="

# 1. Hotelbar Poisson
echo ""
echo ">>> Evaluating Hotelbar Poisson..."
python evaluate_roc_auc.py \
    --data_path /local_data/EventGuard/EventSNN/data/hotelbar/test/hotelbar_poisson_5hz.npy \
    --output_dir $OUT_DIR \
    --filters $FILTERS \
    --dataset_name hotelbar_poisson

# 2. Driving Poisson
echo ""
echo ">>> Evaluating Driving Poisson..."
python evaluate_roc_auc.py \
    --data_path /local_data/EventGuard/EventSNN/data/driving/test/driving_poisson_5hz.npy \
    --output_dir $OUT_DIR \
    --filters $FILTERS \
    --dataset_name driving_poisson

echo ""
echo "=========================================="
echo "ROC-AUC Evaluation Complete!"
echo "Results saved to: $OUT_DIR"
echo "=========================================="
