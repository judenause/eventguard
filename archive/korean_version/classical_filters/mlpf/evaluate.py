# -*- coding: utf-8 -*-
"""
MLPF 평가 스크립트 (다중 데이터셋 지원)

데이터셋:
  - dvsclean: DVSCLEAN_FRAME → SNR
  - esd: esd/total → SNR
  - driving, hotelbar → AUC

Usage:
    python evaluate.py --dataset dvsclean --fps 30
    python evaluate.py --dataset esd
    python evaluate.py --dataset driving
    python evaluate.py --dataset hotelbar
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from model import MLPF
from dataset import MLPFDataset


# 데이터 경로
DATA_PATHS = {
    'dvsclean': '/local_data/EventGuard/EventSNN/data/DVSCLEAN_FRAME',
    'esd': '/local_data/EventGuard/EventSNN/data/esd/total',
    'driving': '/local_data/EventGuard/EventSNN/data/driving',
    'hotelbar': '/local_data/EventGuard/EventSNN/data/hotelbar',
}


def setup_logging(save_dir: str, dataset: str) -> logging.Logger:
    """로깅 설정"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"evaluate_{dataset}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """평가 메트릭 계산"""
    preds = (probs > threshold).astype(int)
    
    # 기본 메트릭
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # AUC
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    
    # SNR 관련 메트릭
    total_signal = (labels == 1).sum()
    total_noise = (labels == 0).sum()
    
    nrr = tn / total_noise if total_noise > 0 else 0
    sr = recall  # Signal Retention = Recall
    
    # Enhanced SNR
    esnr = tp / (fp + 1e-10)
    esnr_db = 10 * np.log10(esnr + 1e-10)
    
    # SNR (dB) = 10 * log10(TP/FP)
    snr_db = 10 * np.log10(tp / (fp + 1e-10)) if fp > 0 else float('inf')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'snr_db': snr_db,
        'esnr_db': esnr_db,
        'nrr': nrr,
        'sr': sr,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_events': len(labels),
        'total_signal': int(total_signal),
        'total_noise': int(total_noise),
    }


def evaluate_on_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> dict:
    """데이터셋에서 모델 평가"""
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for patches, labels in tqdm(data_loader, desc="Evaluating"):
            patches = patches.to(device)
            
            logits = model(patches)
            probs = torch.sigmoid(logits).squeeze(1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    return compute_metrics(all_labels, all_probs, threshold)


def get_test_paths(dataset: str, fps: int = None) -> dict:
    """테스트 데이터 경로 반환"""
    base_path = DATA_PATHS[dataset]
    
    if dataset == 'dvsclean':
        fps_folder = os.path.join(base_path, f"fps{fps}")
        test_folder = os.path.join(fps_folder, "test")
        return {
            'test_50': os.path.join(test_folder, 'test_50'),
            'test_100': os.path.join(test_folder, 'test_100'),
            'tau': 1.0 / fps,
        }
    elif dataset == 'esd':
        return {
            'test_50': os.path.join(base_path, 'test_50'),
            'test_100': os.path.join(base_path, 'test_100'),
            'tau': 0.1,
        }
    elif dataset in ['driving', 'hotelbar']:
        return {
            'test': os.path.join(base_path, 'test'),
            'tau': 0.1,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def evaluate_mlpf(
    dataset: str,
    fps: int = None,
    checkpoint_path: str = None,
    save_dir: str = None,
    threshold: float = 0.5,
    device: str = None
):
    """MLPF 모델 평가"""
    
    base_dir = Path("/local_data/EventGuard/EventSNN/code/classical_filters/mlpf")
    
    # 체크포인트 경로
    if checkpoint_path is None:
        if fps:
            checkpoint_path = base_dir / f"checkpoints/{dataset}_fps{fps}/best_model.pt"
        else:
            checkpoint_path = base_dir / f"checkpoints/{dataset}/best_model.pt"
    
    # 결과 저장 디렉토리
    if save_dir is None:
        save_dir = base_dir / "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 로깅 설정
    logger = setup_logging(str(save_dir), dataset)
    logger.info(f"="*60)
    logger.info(f"MLPF Evaluation - Dataset: {dataset}" + (f", FPS: {fps}" if fps else ""))
    logger.info(f"="*60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # 디바이스 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info(f"Device: {device}")
    
    # 모델 로드
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = MLPF(
        patch_size=checkpoint.get('patch_size', 7),
        hidden_size=checkpoint.get('hidden_size', 20),
        use_polarity=checkpoint.get('use_polarity', True)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # 테스트 경로 가져오기
    test_paths = get_test_paths(dataset, fps)
    tau_seconds = test_paths['tau']
    logger.info(f"τ = {tau_seconds*1000:.1f}ms")
    
    all_results = []
    
    # 데이터셋별 평가
    if dataset in ['dvsclean', 'esd']:
        # test_50, test_100 평가
        for test_name in ['test_50', 'test_100']:
            test_path = test_paths[test_name]
            logger.info(f"\n--- Evaluating {test_name} ---")
            
            test_dataset = MLPFDataset(
                test_path,
                tau_seconds=tau_seconds,
                verbose=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=4096, shuffle=False, num_workers=8,
                pin_memory=True, persistent_workers=True
            )
            
            metrics = evaluate_on_dataset(model, test_loader, device, threshold)
            metrics['dataset'] = dataset
            metrics['test_set'] = test_name
            metrics['fps'] = fps
            all_results.append(metrics)
            
            logger.info(f"{test_name} Results:")
            logger.info(f"  F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
            logger.info(f"  SNR: {metrics['snr_db']:.2f} dB, NRR: {metrics['nrr']:.4f}")
    
    else:
        # driving/hotelbar - 단일 테스트셋
        test_path = test_paths['test']
        logger.info(f"\n--- Evaluating test ---")
        
        test_dataset = MLPFDataset(
            test_path,
            tau_seconds=tau_seconds,
            verbose=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=4096, shuffle=False, num_workers=8,
            pin_memory=True, persistent_workers=True
        )
        
        metrics = evaluate_on_dataset(model, test_loader, device, threshold)
        metrics['dataset'] = dataset
        metrics['test_set'] = 'test'
        metrics['fps'] = fps
        all_results.append(metrics)
        
        logger.info(f"Test Results:")
        logger.info(f"  F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
        logger.info(f"  SNR: {metrics['snr_db']:.2f} dB, NRR: {metrics['nrr']:.4f}")
    
    # 결과 저장
    df_results = pd.DataFrame(all_results)
    
    if fps:
        results_path = os.path.join(save_dir, f"mlpf_{dataset}_fps{fps}_results.csv")
    else:
        results_path = os.path.join(save_dir, f"mlpf_{dataset}_results.csv")
    
    df_results.to_csv(results_path, index=False)
    logger.info(f"\n✅ Results saved to: {results_path}")
    
    # 요약 출력
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary")
    logger.info(f"{'='*60}")
    logger.info(f"{'Test Set':<12} {'F1':<8} {'AUC':<8} {'SNR(dB)':<10} {'NRR':<8}")
    logger.info(f"{'-'*50}")
    for r in all_results:
        logger.info(f"{r['test_set']:<12} {r['f1']:.4f}   {r['auc']:.4f}   {r['snr_db']:>7.2f}    {r['nrr']:.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="MLPF Evaluation (Multi-Dataset)")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['dvsclean', 'esd', 'driving', 'hotelbar'],
                        help='Dataset to evaluate')
    parser.add_argument('--fps', type=int, default=None, choices=[30, 60, 90, 120],
                        help='FPS value (required for dvsclean)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.dataset == 'dvsclean' and args.fps is None:
        parser.error("--fps is required for dvsclean dataset")
    
    evaluate_mlpf(
        dataset=args.dataset,
        fps=args.fps,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        threshold=args.threshold,
        device=args.device
    )


if __name__ == "__main__":
    main()
