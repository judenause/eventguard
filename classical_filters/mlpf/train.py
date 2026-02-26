# -*- coding: utf-8 -*-
"""
MLPF 학습 스크립트 (다중 데이터셋 지원)

데이터셋:
  - dvsclean: DVSCLEAN_FRAME (FPS별)
  - esd: esd/total
  - driving: driving dataset
  - hotelbar: hotelbar dataset

Usage:
    python train.py --dataset dvsclean --fps 30 --epochs 50
    python train.py --dataset esd --epochs 50
    python train.py --dataset driving --epochs 50
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MLPF
from dataset import MLPFDataset


# 데이터셋 경로 설정
DATA_PATHS = {
    'dvsclean': '/local_data/EventGuard/EventSNN/data/DVSCLEAN_FRAME',
    'esd': '/local_data/EventGuard/EventSNN/data/esd/total',
    'driving': '/local_data/EventGuard/EventSNN/data/driving',
    'hotelbar': '/local_data/EventGuard/EventSNN/data/hotelbar',
}


def setup_logging(save_dir: str, dataset: str, fps: int = None) -> logging.Logger:
    """로깅 설정"""
    os.makedirs(save_dir, exist_ok=True)
    
    if fps:
        log_file = os.path.join(save_dir, f"train_{dataset}_fps{fps}.log")
    else:
        log_file = os.path.join(save_dir, f"train_{dataset}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def get_data_paths(dataset: str, fps: int = None) -> dict:
    """데이터셋별 경로 반환"""
    base_path = DATA_PATHS[dataset]
    
    if dataset == 'dvsclean':
        fps_folder = os.path.join(base_path, f"fps{fps}")
        return {
            'train': os.path.join(fps_folder, 'train'),
            'val': os.path.join(fps_folder, 'val'),
            'tau': 1.0 / fps,  # τ = 1/FPS
        }
    elif dataset == 'esd':
        return {
            'train': os.path.join(base_path, 'train'),
            'val': os.path.join(base_path, 'val'),
            'tau': 0.1,  # 100ms
        }
    elif dataset in ['driving', 'hotelbar']:
        return {
            'train': os.path.join(base_path, 'train'),
            'val': os.path.join(base_path, 'val'),
            'tau': 0.1,  # 100ms (MLPF 기본값)
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """한 에폭 학습"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]")
    
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * patches.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += patches.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """검증"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for patches, labels in tqdm(val_loader, desc="[Validation]"):
            patches = patches.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            logits = model(patches)
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long()
            
            total_loss += loss.item() * patches.size(0)
            correct += (preds == labels.squeeze(1).long()).sum().item()
            total += patches.size(0)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.squeeze(1).cpu().numpy())
    
    # AUC 계산
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'auc': auc
    }


def train_mlpf(
    dataset: str,
    fps: int = None,
    epochs: int = 50,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    save_dir: str = None,
    debug: bool = False,
    device: str = None
):
    """MLPF 모델 학습"""
    
    # 저장 디렉토리 설정
    base_dir = Path("/local_data/EventGuard/EventSNN/code/classical_filters/mlpf")
    
    if save_dir is None:
        if fps:
            save_dir = base_dir / f"checkpoints/{dataset}_fps{fps}"
        else:
            save_dir = base_dir / f"checkpoints/{dataset}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 로깅 설정
    logger = setup_logging(str(save_dir), dataset, fps)
    logger.info(f"="*60)
    logger.info(f"MLPF Training - Dataset: {dataset}" + (f", FPS: {fps}" if fps else ""))
    logger.info(f"="*60)
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Save dir: {save_dir}")
    
    # 디바이스 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info(f"Device: {device}")
    
    # 데이터 경로 가져오기
    paths = get_data_paths(dataset, fps)
    tau_seconds = paths['tau']
    logger.info(f"τ = {tau_seconds*1000:.1f}ms")
    
    # 데이터셋 생성
    max_files = 3 if debug else None
    
    logger.info(f"\n📂 Train 데이터셋 로딩: {paths['train']}")
    train_dataset = MLPFDataset(
        paths['train'],
        tau_seconds=tau_seconds,
        max_files=max_files,
        verbose=True
    )
    
    logger.info(f"\n📂 Validation 데이터셋 로딩: {paths['val']}")
    val_dataset = MLPFDataset(
        paths['val'],
        tau_seconds=tau_seconds,
        max_files=max_files,
        verbose=True
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 모델 생성 (TI + Polarity = 98 inputs)
    model = MLPF(patch_size=7, hidden_size=20, use_polarity=True).to(device)
    logger.info(f"Model: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 루프
    best_val_auc = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{epochs} ---")
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.2f}%")
        
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%, AUC: {val_metrics['auc']:.4f}")
        
        scheduler.step(val_metrics['auc'])
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_accuracy': val_metrics['accuracy'],
                'dataset': dataset,
                'fps': fps,
                'tau': tau_seconds,
                'patch_size': 7,
                'hidden_size': 20,
                'use_polarity': True,
            }
            
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✅ Best model saved (AUC: {best_val_auc:.4f})")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed! Best epoch: {best_epoch}, Best Val AUC: {best_val_auc:.4f}")
    logger.info(f"{'='*60}")
    
    return best_val_auc


def main():
    parser = argparse.ArgumentParser(description="MLPF Training (Multi-Dataset)")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['dvsclean', 'esd', 'driving', 'hotelbar'],
                        help='Dataset to train on')
    parser.add_argument('--fps', type=int, default=None, choices=[30, 60, 90, 120],
                        help='FPS value (required for dvsclean)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--device', type=str, default=None, help='Device')
    
    args = parser.parse_args()
    
    # dvsclean은 FPS 필수
    if args.dataset == 'dvsclean' and args.fps is None:
        parser.error("--fps is required for dvsclean dataset")
    
    train_mlpf(
        dataset=args.dataset,
        fps=args.fps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        debug=args.debug,
        device=args.device
    )


if __name__ == "__main__":
    main()
