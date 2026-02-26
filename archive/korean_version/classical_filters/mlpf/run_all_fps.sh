#!/bin/bash
# MLPF 전체 FPS 학습 및 평가 스크립트
#
# Usage:
#   ./run_all_fps.sh          # 전체 학습 + 평가
#   ./run_all_fps.sh train    # 학습만
#   ./run_all_fps.sh eval     # 평가만

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 로그 디렉토리 생성
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "MLPF Training & Evaluation Pipeline"
echo "=============================================="
echo "Script Directory: $SCRIPT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# FPS 목록
FPS_LIST=(30 60 90 120)

# 모드 결정
MODE=${1:-"all"}

if [ "$MODE" == "train" ] || [ "$MODE" == "all" ]; then
    echo ""
    echo "=============================================="
    echo "Training Phase"
    echo "=============================================="
    
    for fps in "${FPS_LIST[@]}"; do
        echo ""
        echo "----------------------------------------------"
        echo "Training FPS $fps"
        echo "----------------------------------------------"
        
        LOG_FILE="$LOG_DIR/train_fps${fps}_${TIMESTAMP}.log"
        
        python train.py \
            --fps $fps \
            --epochs 50 \
            --batch_size 4096 \
            --lr 0.001 \
            2>&1 | tee "$LOG_FILE"
        
        echo "✅ FPS $fps training completed. Log: $LOG_FILE"
    done
fi

if [ "$MODE" == "eval" ] || [ "$MODE" == "all" ]; then
    echo ""
    echo "=============================================="
    echo "Evaluation Phase"
    echo "=============================================="
    
    for fps in "${FPS_LIST[@]}"; do
        echo ""
        echo "----------------------------------------------"
        echo "Evaluating FPS $fps"
        echo "----------------------------------------------"
        
        LOG_FILE="$LOG_DIR/eval_fps${fps}_${TIMESTAMP}.log"
        
        python evaluate.py \
            --fps $fps \
            2>&1 | tee "$LOG_FILE"
        
        echo "✅ FPS $fps evaluation completed. Log: $LOG_FILE"
    done
fi

echo ""
echo "=============================================="
echo "Pipeline Completed!"
echo "=============================================="
echo "Logs saved to: $LOG_DIR"
echo "Results saved to: $SCRIPT_DIR/results/"
