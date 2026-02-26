#!/bin/bash
# MLPF 다중 데이터셋 학습/평가 스크립트
#
# 사용법:
#   ./run_all_datasets.sh           # 전체 (학습 + 평가)
#   ./run_all_datasets.sh train     # 학습만
#   ./run_all_datasets.sh eval      # 평가만
#
# 데이터셋:
#   1. DVSCLEAN_FRAME (FPS별) → SNR
#   2. esd/total → SNR
#   3. driving, hotelbar → AUC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 로그 디렉토리 생성
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "MLPF Multi-Dataset Training & Evaluation"
echo "=============================================="
echo "Script Directory: $SCRIPT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

MODE=${1:-"all"}

# ============================================
# 1. DVSCLEAN_FRAME (FPS별 학습/평가)
# ============================================
run_dvsclean() {
    echo ""
    echo "=============================================="
    echo "[1/3] DVSCLEAN_FRAME Dataset"
    echo "=============================================="
    
    for fps in 30 60 90 120; do
        echo ""
        echo "--- FPS $fps ---"
        
        if [ "$MODE" == "train" ] || [ "$MODE" == "all" ]; then
            LOG_FILE="$LOG_DIR/dvsclean_fps${fps}_train_${TIMESTAMP}.log"
            echo "Training FPS $fps..."
            python train.py \
                --dataset dvsclean \
                --fps $fps \
                --epochs 150 \
                --batch_size 4096 \
                2>&1 | tee "$LOG_FILE"
        fi
        
        if [ "$MODE" == "eval" ] || [ "$MODE" == "all" ]; then
            LOG_FILE="$LOG_DIR/dvsclean_fps${fps}_eval_${TIMESTAMP}.log"
            echo "Evaluating FPS $fps..."
            python evaluate.py \
                --dataset dvsclean \
                --fps $fps \
                2>&1 | tee "$LOG_FILE"
        fi
    done
}

# ============================================
# 2. ESD/Total 학습/평가
# ============================================
run_esd() {
    echo ""
    echo "=============================================="
    echo "[2/3] ESD/Total Dataset"
    echo "=============================================="
    
    if [ "$MODE" == "train" ] || [ "$MODE" == "all" ]; then
        LOG_FILE="$LOG_DIR/esd_train_${TIMESTAMP}.log"
        echo "Training on esd/total..."
        python train.py \
            --dataset esd \
            --epochs 50 \
            --batch_size 4096 \
            2>&1 | tee "$LOG_FILE"
    fi
    
    if [ "$MODE" == "eval" ] || [ "$MODE" == "all" ]; then
        LOG_FILE="$LOG_DIR/esd_eval_${TIMESTAMP}.log"
        echo "Evaluating on esd/total..."
        python evaluate.py \
            --dataset esd \
            2>&1 | tee "$LOG_FILE"
    fi
}

# ============================================
# 3. Driving & Hotelbar 학습/평가 (AUC 중심)
# ============================================
run_driving_hotelbar() {
    echo ""
    echo "=============================================="
    echo "[3/3] Driving & Hotelbar Datasets (AUC)"
    echo "=============================================="
    
    for dataset in driving hotelbar; do
        echo ""
        echo "--- $dataset ---"
        
        if [ "$MODE" == "train" ] || [ "$MODE" == "all" ]; then
            LOG_FILE="$LOG_DIR/${dataset}_train_${TIMESTAMP}.log"
            echo "Training on $dataset..."
            python train.py \
                --dataset $dataset \
                --epochs 50 \
                --batch_size 4096 \
                2>&1 | tee "$LOG_FILE"
        fi
        
        if [ "$MODE" == "eval" ] || [ "$MODE" == "all" ]; then
            LOG_FILE="$LOG_DIR/${dataset}_eval_${TIMESTAMP}.log"
            echo "Evaluating on $dataset..."
            python evaluate.py \
                --dataset $dataset \
                2>&1 | tee "$LOG_FILE"
        fi
    done
}

# ============================================
# 메인 실행
# ============================================
case "${2:-all}" in
    dvsclean)
        run_dvsclean
        ;;
    esd)
        run_esd
        ;;
    driving|hotelbar)
        run_driving_hotelbar
        ;;
    *)
        run_dvsclean
        run_esd
        run_driving_hotelbar
        ;;
esac

echo ""
echo "=============================================="
echo "✅ Pipeline Completed!"
echo "=============================================="
echo "Logs: $LOG_DIR"
echo "Results: $RESULTS_DIR"
