#!/bin/bash
# ==============================================================================
# V9 FPS 30 - Beta 0.4 학습
# GPU: 0,1 (2개)
# ==============================================================================

FPS=30
BETA=0.4
PROJECT_ROOT="/local_data/EventGuard/EventSNN/code/v8_bconvsnn"
DATA_ROOT="/local_data/EventGuard/EventSNN/data/esd/total"

SAVE_DIR="${PROJECT_ROOT}/results/v9_qat_fps30_beta04"
mkdir -p "${SAVE_DIR}"

LOG_FILE="${SAVE_DIR}/training.log"

echo "============================================" | tee "${LOG_FILE}"
echo "V9 FPS30 Beta ${BETA} Training: $(date)" | tee -a "${LOG_FILE}"
echo "FPS: ${FPS}, Beta: ${BETA}, Window: 10" | tee -a "${LOG_FILE}"
echo "GPU: 0,1" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

/opt/miniconda/envs/Event/bin/torchrun --nproc_per_node=2 --master_port=29903 main_train.py \
    --fps ${FPS} \
    --save_dir "${SAVE_DIR}" \
    --train_data_folder "${DATA_ROOT}/train" \
    --val_data_folder "${DATA_ROOT}/val" \
    --test_data_folder "${DATA_ROOT}/test_50" \
    --learning_rate 1e-4 \
    --lr_scheduler_type "Cosine" \
    --patience 40 \
    --base_batch_size 2 \
    --window_size 10 \
    --num_epochs 150 \
    --save_metric f1 \
    --snn_threshold 1.0 \
    --snn_beta ${BETA} \
    --use_multi_gpu \
    --num_workers 2 \
    --loss_type "Focal" \
    --focal_alpha 0.7 \
    --focal_gamma 2.0 \
    --processed_data_save_dir "./processed_cache_fps${FPS}" \
    --csv_name "v9_qat_fps${FPS}_beta04" \
    2>&1 | tee -a "${LOG_FILE}"

echo "============================================" | tee -a "${LOG_FILE}"
echo "Training Finished: $(date)" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"
