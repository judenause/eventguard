# MLPF Python Implementation

MLPF(MultiLayer Perceptron Filter)의 Python/PyTorch 구현입니다.

## 파일 구조

```
mlpf/
├── model.py       # MLPF 모델 정의 (49→20→1 or 98→20→1)
├── dataset.py     # TI 패치 생성 및 PyTorch Dataset
├── train.py       # 학습 스크립트
├── evaluate.py    # 평가 스크립트 (test_50/test_100 분리)
├── run_all_fps.sh # 전체 FPS 자동화 스크립트
└── README.md
```

## 사용법

### 1. 단일 FPS 학습

```bash
python train.py --fps 30 --epochs 50
python train.py --fps 60 --epochs 50 --debug  # 디버그 모드
```

### 2. 단일 FPS 평가

```bash
python evaluate.py --fps 30
python evaluate.py --fps 60 --checkpoint checkpoints/fps60/best_model.pt
```

### 3. 전체 FPS 학습 및 평가

```bash
chmod +x run_all_fps.sh
./run_all_fps.sh         # 전체 (학습 + 평가)
./run_all_fps.sh train   # 학습만
./run_all_fps.sh eval    # 평가만
```

## 모델 구조

| 항목 | 원본 (ipynb) | Java 버전 | 현재 구현 |
|------|-------------|-----------|----------|
| 입력 크기 | 49 (TI only) | 98 (TI+Pol) | 98 (기본) |
| 은닉층 | 20 | 20 | 20 |
| Dropout | 0.2 | - | 0.2 |
| 출력 | Sigmoid | Sigmoid | Sigmoid |

## τ (시간 윈도우) 설정

| FPS | τ (ms) |
|-----|--------|
| 30  | 33.3   |
| 60  | 16.7   |
| 90  | 11.1   |
| 120 | 8.3    |

## Reference

Guo & Delbruck, "Low Cost and Latency Event Camera Background Activity Denoising", T-PAMI 2022
