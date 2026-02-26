#!/usr/bin/env python3
"""
τ sweep을 통한 ROC-AUC 계산 스크립트

논문 방식대로 τ (correlation time)를 sweep하여 각 threshold에서
TPR/FPR을 측정하고 ROC curve를 생성하여 실제 AUC를 계산합니다.

대상 필터: BAF, ONF, STCF, STCF_Sub
"""

import numpy as np
import os
import argparse
import glob
from tqdm import tqdm
from sklearn.metrics import auc

# matplotlib는 선택적 (플롯 생성시에만 필요)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not found. ROC curve plot will be skipped.")

# τ sweep 범위 (초 단위)
TAU_VALUES = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.024, 0.03, 0.04, 0.05, 0.075, 0.1]

# STCF k값 (k=1: BAF와 유사하지만 polarity 매칭 적용)
STCF_K = 1


def apply_baf_filter(events: np.ndarray, tau: float) -> np.ndarray:
    """
    BAF (Background Activity Filter) 적용 - Java 구현과 일치
    
    각 이벤트에 대해, 3x3 이웃 중 하나라도 tau 시간 내에 이벤트가 있으면 signal로 판단
    (자기 자신 픽셀 제외)
    """
    predictions = np.zeros(len(events), dtype=np.int32)
    
    # 픽셀별 마지막 이벤트 시간 저장
    last_timestamp = {}
    
    for i, event in enumerate(events):
        x, y, t = int(event[1]), int(event[2]), event[3]
        
        # 3x3 이웃 확인 (Java 구현과 일치)
        is_correlated = False
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # 자기 자신 제외 (filterHotPixels)
                
                key = (x + dx, y + dy)
                if key in last_timestamp:
                    if t - last_timestamp[key] <= tau:
                        is_correlated = True
                        break
            if is_correlated:
                break
        
        if is_correlated:
            predictions[i] = 1  # Signal
        
        # 현재 픽셀 타임스탬프 업데이트
        last_timestamp[(x, y)] = t
    
    return predictions


def apply_stcf_filter(events: np.ndarray, tau: float, k: int = 2, radius: int = 1) -> np.ndarray:
    """
    STCF (Spatio-Temporal Correlation Filter) 적용 - Java 구현과 일치
    
    각 이벤트에 대해, 주변 픽셀에서 tau 시간 내에 k개 이상의 **같은 polarity** 이벤트가 있으면 signal로 판단
    (polaritiesMustMatch = true, 자기 자신 제외)
    """
    predictions = np.zeros(len(events), dtype=np.int32)
    
    # 픽셀별 최근 이벤트 시간 및 polarity 저장
    pixel_timestamps = {}
    pixel_polarities = {}
    
    for i, event in enumerate(events):
        x, y, t = int(event[1]), int(event[2]), event[3]
        pol = int(event[4])  # polarity: -1 또는 1
        
        # 주변 픽셀에서 같은 polarity의 최근 이벤트 확인
        neighbor_count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # 자기 자신 제외
                
                key = (x + dx, y + dy)
                if key in pixel_timestamps:
                    if t - pixel_timestamps[key] <= tau:
                        # polaritiesMustMatch: 같은 polarity만 카운트
                        if pixel_polarities.get(key, 0) == pol:
                            neighbor_count += 1
        
        if neighbor_count >= k:
            predictions[i] = 1  # Signal
        
        # 현재 픽셀 타임스탬프 및 polarity 업데이트
        pixel_timestamps[(x, y)] = t
        pixel_polarities[(x, y)] = pol
    
    return predictions


def apply_onf_filter(events: np.ndarray, tau: float, width: int = 1280, height: int = 720) -> np.ndarray:
    """
    ONF (Ordered Noise Filter) 적용
    
    픽셀별로 tau 시간 내에 연속된 이벤트가 있으면 signal로 판단
    """
    predictions = np.zeros(len(events), dtype=np.int32)
    
    # 1D Row/Column Timestamp Arrays (O(N) Memory)
    row_ts = np.full(height, -np.inf)
    col_ts = np.full(width, -np.inf)
    
    for i, event in enumerate(events):
        x, y, t = int(event[1]), int(event[2]), event[3]
        
        # Check boundary
        if not (0 <= x < width and 0 <= y < height):
            continue

        # Check connectivity in Row OR Column (Noise passing filter)
        # If any recent event exists in the same row OR same column, it supports this event.
        # (This creates 'ghost' events but follows the O(N) filter design for minimal memory)
        is_row_correlated = (t - row_ts[y] <= tau)
        is_col_correlated = (t - col_ts[x] <= tau)
        
        if is_row_correlated or is_col_correlated:
            predictions[i] = 1  # Signal
        
        # Update Memory
        row_ts[y] = t
        col_ts[x] = t
    
    return predictions


def apply_stcf_sub_filter(events: np.ndarray, tau: float, block_size: int = 2) -> np.ndarray:
    """
    STCF_Sub (Subsampled STCF) 적용 - 논문 Fig. 2와 일치
    
    원리: NxN 블록으로 서브샘플링, 같은 블록 내 이전 이벤트가 tau 내에 있으면 통과
    - 주변 블록이 아닌 **같은 블록**만 확인
    - 첫 번째 이벤트는 윈도우를 열고, 후속 이벤트가 통과
    """
    predictions = np.zeros(len(events), dtype=np.int32)
    
    # 블록별 마지막 이벤트 시간
    block_timestamps = {}
    
    for i, event in enumerate(events):
        x, y, t = int(event[1]), int(event[2]), event[3]
        
        # 블록 좌표 (서브샘플링)
        bx, by = x // block_size, y // block_size
        block_key = (bx, by)
        
        # 같은 블록에 이전 이벤트가 있는지 확인
        if block_key in block_timestamps:
            if t - block_timestamps[block_key] <= tau:
                predictions[i] = 1  # Signal (correlated)
        
        # 블록 타임스탬프 업데이트 (현재 이벤트가 다음 이벤트를 위한 윈도우를 연다)
        block_timestamps[block_key] = t
    
    return predictions


def compute_tpr_fpr(gt: np.ndarray, predictions: np.ndarray):
    """TPR과 FPR 계산"""
    tp = np.sum((predictions == 1) & (gt == 1))
    tn = np.sum((predictions == 0) & (gt == 0))
    fp = np.sum((predictions == 1) & (gt == 0))
    fn = np.sum((predictions == 0) & (gt == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return tpr, fpr


def evaluate_filter_with_tau_sweep(filter_name: str, events: np.ndarray, tau_values: list):
    """
    주어진 필터에 대해 τ sweep을 수행하고 ROC 포인트 반환
    """
    # Ground Truth: 데이터셋은 0=Signal, 1=Noise로 되어 있음
    # 평가를 위해 1=Signal, 0=Noise로 반전
    gt = 1 - events[:, 0].astype(np.int32)
    
    tpr_list = []
    fpr_list = []
    
    for tau in tqdm(tau_values, desc=f"  {filter_name} τ sweep"):
        if filter_name == 'BAF':
            predictions = apply_baf_filter(events, tau)
        elif filter_name == 'STCF':
            predictions = apply_stcf_filter(events, tau, k=STCF_K)
        elif filter_name == 'ONF':
            predictions = apply_onf_filter(events, tau)
        elif filter_name == 'STCF_Sub':
            predictions = apply_stcf_sub_filter(events, tau)
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        tpr, fpr = compute_tpr_fpr(gt, predictions)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list)


def compute_roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """ROC curve의 AUC 계산"""
    # FPR 기준으로 정렬
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # (0,0)과 (1,1) 점 추가
    fpr_sorted = np.concatenate([[0], fpr_sorted, [1]])
    tpr_sorted = np.concatenate([[0], tpr_sorted, [1]])
    
    return auc(fpr_sorted, tpr_sorted)


def main():
    parser = argparse.ArgumentParser(description="τ sweep ROC-AUC 계산")
    parser.add_argument('--data_path', type=str, required=True, help='평가할 .npy 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./results/roc_auc', help='결과 저장 디렉토리')
    parser.add_argument('--filters', nargs='+', default=['BAF', 'STCF', 'ONF', 'STCF_Sub'], 
                        help='평가할 필터 목록')
    parser.add_argument('--dataset_name', type=str, default='dataset', help='데이터셋 이름')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"τ Sweep ROC-AUC 계산")
    print(f"{'='*70}")
    print(f"데이터: {args.data_path}")
    print(f"필터: {args.filters}")
    print(f"τ 범위: {TAU_VALUES[0]}s ~ {TAU_VALUES[-1]}s ({len(TAU_VALUES)}개)")
    print(f"{'='*70}\n")
    
    # 데이터 로드
    print("📂 데이터 로딩...")
    events = np.load(args.data_path)
    print(f"   이벤트 수: {len(events):,}")
    
    # 타임스탬프 정규화 (필요시)
    t_max = events[:, 3].max()
    if t_max > 10000:
        print(f"   ℹ️  타임스탬프 정규화 적용 (t_max={t_max:.2f})")
        events[:, 3] = (events[:, 3] - events[:, 3].min()) / 1_000_000.0
    
    results = {}
    
    # 각 필터에 대해 평가
    for filter_name in args.filters:
        print(f"\n🔍 {filter_name} 필터 평가 중...")
        
        fpr, tpr = evaluate_filter_with_tau_sweep(filter_name, events, TAU_VALUES)
        roc_auc = compute_roc_auc(fpr, tpr)
        
        results[filter_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'tau_values': TAU_VALUES
        }
        
        print(f"   ✅ {filter_name} AUC: {roc_auc:.4f}")
    
    # 결과 저장
    print(f"\n{'='*70}")
    print("📊 결과 요약")
    print(f"{'='*70}")
    
    summary_path = os.path.join(args.output_dir, f'{args.dataset_name}_roc_auc_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"τ Sweep ROC-AUC 결과\n")
        f.write(f"{'='*50}\n")
        f.write(f"데이터셋: {args.dataset_name}\n")
        f.write(f"이벤트 수: {len(events):,}\n")
        f.write(f"τ 범위: {TAU_VALUES[0]}s ~ {TAU_VALUES[-1]}s\n")
        f.write(f"STCF k값: {STCF_K}\n\n")
        
        f.write(f"{'Filter':<15} {'AUC':<10}\n")
        f.write(f"{'-'*25}\n")
        
        for filter_name, data in sorted(results.items(), key=lambda x: -x[1]['auc']):
            print(f"   {filter_name:<15} AUC: {data['auc']:.4f}")
            f.write(f"{filter_name:<15} {data['auc']:.4f}\n")
        
        f.write(f"\n\nτ별 상세 결과\n")
        f.write(f"{'-'*50}\n")
        for filter_name, data in results.items():
            f.write(f"\n{filter_name}:\n")
            f.write(f"  {'τ (ms)':<10} {'FPR':<10} {'TPR':<10}\n")
            for i, tau in enumerate(TAU_VALUES):
                f.write(f"  {tau*1000:<10.1f} {data['fpr'][i]:<10.4f} {data['tpr'][i]:<10.4f}\n")
    
    print(f"\n   📁 결과 저장: {summary_path}")
    
    # ROC curve 플롯
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 8))
        for filter_name, data in results.items():
            fpr_plot = np.concatenate([[0], data['fpr'], [1]])
            tpr_plot = np.concatenate([[0], data['tpr'], [1]])
            sorted_indices = np.argsort(fpr_plot)
            plt.plot(fpr_plot[sorted_indices], tpr_plot[sorted_indices], 
                     marker='o', label=f"{filter_name} (AUC={data['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curves - {args.dataset_name}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(args.output_dir, f'{args.dataset_name}_roc_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 ROC curve 저장: {plot_path}")
    else:
        print("   ℹ️  matplotlib 없음 - ROC curve 플롯 생략")
    
    print(f"\n{'='*70}")
    print("✅ 완료!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
