# -*- coding: utf-8 -*-
"""
MLPF Dataset: TI(Timestamp Image) 패치 생성 및 PyTorch Dataset

MLPF.java의 TI 패치 생성 로직을 Python으로 구현:
1. 각 이벤트에 대해 주변 픽셀의 TI 값 계산 (LinearDecay)
2. Polarity 채널 추가
3. PyTorch Dataset으로 래핑

Reference: Guo & Delbruck, T-PAMI 2022
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
from pathlib import Path
from tqdm import tqdm
import os


class TIPatchGenerator:
    """
    TI(Timestamp Image) 패치 생성기
    
    MLPF.java의 filterPacket 메서드를 Python으로 구현
    """
    
    def __init__(
        self,
        patch_size: int = 7,
        tau_seconds: float = 0.1,
        width: int = 1280,
        height: int = 720,
        decay_method: str = 'linear'  # 'linear' or 'exponential'
    ):
        """
        Args:
            patch_size: 패치 크기 (기본 7x7)
            tau_seconds: 시간 윈도우 τ (초 단위)
            width: 이미지 너비
            height: 이미지 높이
            decay_method: TI 값 감쇠 방법 ('linear' 또는 'exponential')
        """
        self.patch_size = patch_size
        self.radius = (patch_size - 1) // 2
        self.tau = tau_seconds
        self.width = width
        self.height = height
        self.decay_method = decay_method
        
        # Timestamp Image와 Polarity Map 초기화
        self.reset()
    
    def reset(self):
        """상태 초기화"""
        # DEFAULT_TIMESTAMP = Integer.MIN_VALUE in Java
        self.timestamp_image = np.full((self.width, self.height), -np.inf, dtype=np.float64)
        self.polarity_map = np.zeros((self.width, self.height), dtype=np.int8)
    
    def compute_ti_value(self, dt: float) -> float:
        """
        TI 값 계산
        
        Args:
            dt: 시간 차이 (현재 이벤트 - 이웃 이벤트), 음수여야 함
        
        Returns:
            TI 값 (0~1)
        """
        if dt > 0:
            # 비단조 타임스탬프 (무시)
            return 0.0
        
        abs_dt = -dt
        
        if abs_dt >= self.tau:
            return 0.0
        
        if self.decay_method == 'linear':
            # LinearDecay: v = 1 - |dt| / τ
            return 1.0 - abs_dt / self.tau
        else:
            # ExponentialDecay: v = exp(dt / τ)
            return np.exp(dt / self.tau)
    
    def extract_patch(self, x: int, y: int, t: float, polarity: int) -> np.ndarray:
        """
        단일 이벤트에 대한 TI 패치 추출
        
        Args:
            x, y: 이벤트 좌표
            t: 이벤트 타임스탬프 (초 단위)
            polarity: 이벤트 극성 (0=OFF, 1=ON)
        
        Returns:
            (2 * patch_size^2,) 형태의 패치 벡터
            - 앞쪽 patch_size^2: TI 값
            - 뒤쪽 patch_size^2: Polarity 값
        """
        patch_ti = np.zeros(self.patch_size * self.patch_size, dtype=np.float32)
        patch_pol = np.zeros(self.patch_size * self.patch_size, dtype=np.float32)
        
        idx = 0
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                nx, ny = x + dx, y + dy
                
                # 경계 검사
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    patch_ti[idx] = 0.0
                    patch_pol[idx] = 0.0
                else:
                    neighbor_ts = self.timestamp_image[nx, ny]
                    
                    if neighbor_ts == -np.inf:
                        # 해당 픽셀에 이벤트가 없음
                        patch_ti[idx] = 0.0
                        patch_pol[idx] = 0.0
                    else:
                        dt = neighbor_ts - t  # 음수여야 정상
                        ti_value = self.compute_ti_value(dt)
                        patch_ti[idx] = ti_value
                        
                        # Polarity 값 설정
                        if ti_value > 0:
                            # τ 이내의 이벤트인 경우에만 polarity 사용
                            patch_pol[idx] = self.polarity_map[nx, ny]
                        else:
                            patch_pol[idx] = 0.0
                
                idx += 1
        
        # 중앙 픽셀 polarity는 현재 이벤트의 polarity
        center_idx = (self.patch_size * self.patch_size) // 2
        pol_sign = 1 if polarity > 0 else -1  # 0->-1, 1->+1
        patch_pol[center_idx] = pol_sign
        
        # TI와 Polarity 연결
        return np.concatenate([patch_ti, patch_pol])
    
    def update_state(self, x: int, y: int, t: float, polarity: int):
        """
        Timestamp Image와 Polarity Map 업데이트
        
        Args:
            x, y: 이벤트 좌표
            t: 이벤트 타임스탬프
            polarity: 이벤트 극성
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.timestamp_image[x, y] = t
            self.polarity_map[x, y] = 1 if polarity > 0 else -1
    
    def process_events(
        self,
        events: np.ndarray,
        return_labels: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        이벤트 배열 전체 처리
        
        Args:
            events: (N, 5) 배열 [label, x, y, t, polarity]
            return_labels: 레이블도 반환할지 여부
        
        Returns:
            patches: (N, 2*patch_size^2) 형태의 패치 배열
            labels: (N,) 형태의 레이블 배열 (return_labels=True인 경우)
        """
        self.reset()
        
        n_events = len(events)
        patch_dim = 2 * self.patch_size * self.patch_size
        patches = np.zeros((n_events, patch_dim), dtype=np.float32)
        
        if return_labels:
            labels = events[:, 0].astype(np.int64)
        
        for i in range(n_events):
            label, x, y, t, pol = events[i]
            x, y = int(x), int(y)
            
            # 패치 추출 (현재 상태 기준)
            patches[i] = self.extract_patch(x, y, t, pol)
            
            # 상태 업데이트 (패치 추출 후 업데이트 - MLPF.java 동작 방식)
            self.update_state(x, y, t, pol)
        
        if return_labels:
            return patches, labels
        return patches, None


class MLPFDataset(Dataset):
    """
    MLPF 학습/평가용 PyTorch Dataset
    
    NPY 파일들을 로드하여 TI 패치와 레이블을 제공
    """
    
    def __init__(
        self,
        data_folder: str,
        patch_size: int = 7,
        tau_seconds: float = 0.1,
        width: int = 1280,
        height: int = 720,
        precompute: bool = True,
        max_files: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Args:
            data_folder: NPY 파일들이 있는 폴더 경로
            patch_size: 패치 크기
            tau_seconds: 시간 윈도우 τ
            width, height: 이미지 크기
            precompute: 패치를 미리 계산할지 여부
            max_files: 로드할 최대 파일 수 (디버깅용)
            verbose: 진행 상황 출력 여부
        """
        self.data_folder = Path(data_folder)
        self.patch_size = patch_size
        self.tau = tau_seconds
        self.width = width
        self.height = height
        self.verbose = verbose
        
        # NPY 파일 목록 로드
        self.file_list = sorted(self.data_folder.glob("*.npy"))
        if max_files is not None:
            self.file_list = self.file_list[:max_files]
        
        if len(self.file_list) == 0:
            raise ValueError(f"No .npy files found in {data_folder}")
        
        if verbose:
            print(f"📂 데이터 폴더: {data_folder}")
            print(f"📁 파일 수: {len(self.file_list)}")
        
        # 패치 생성기
        self.patch_generator = TIPatchGenerator(
            patch_size=patch_size,
            tau_seconds=tau_seconds,
            width=width,
            height=height,
            decay_method='linear'
        )
        
        # 패치 미리 계산
        if precompute:
            self._precompute_patches()
        else:
            self.patches = None
            self.labels = None
            self._load_file_indices()
    
    def _precompute_patches(self):
        """모든 파일의 패치를 미리 계산"""
        all_patches = []
        all_labels = []
        
        iterator = tqdm(self.file_list, desc="패치 생성") if self.verbose else self.file_list
        
        for file_path in iterator:
            events = np.load(file_path)
            patches, labels = self.patch_generator.process_events(events)
            all_patches.append(patches)
            all_labels.append(labels)
        
        self.patches = np.concatenate(all_patches, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        
        if self.verbose:
            n_signal = (self.labels == 1).sum()
            n_noise = (self.labels == 0).sum()
            print(f"✅ 패치 생성 완료: {len(self.patches):,} events")
            print(f"   Signal: {n_signal:,} ({100*n_signal/len(self.labels):.1f}%)")
            print(f"   Noise: {n_noise:,} ({100*n_noise/len(self.labels):.1f}%)")
    
    def _load_file_indices(self):
        """파일별 인덱스 계산 (lazy loading용)"""
        self.file_starts = []
        self.file_lengths = []
        total = 0
        
        for file_path in self.file_list:
            events = np.load(file_path)
            self.file_starts.append(total)
            self.file_lengths.append(len(events))
            total += len(events)
        
        self.total_events = total
    
    def __len__(self) -> int:
        if self.patches is not None:
            return len(self.patches)
        return self.total_events
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.patches is not None:
            patch = torch.from_numpy(self.patches[idx])
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return patch, label
        else:
            raise NotImplementedError("Lazy loading not implemented yet")
    
    def get_class_weights(self) -> torch.Tensor:
        """클래스 불균형 보정을 위한 가중치 계산"""
        if self.labels is None:
            raise ValueError("Precompute must be True to get class weights")
        
        n_signal = (self.labels == 1).sum()
        n_noise = (self.labels == 0).sum()
        total = n_signal + n_noise
        
        # Inverse frequency weighting
        weight_noise = total / (2 * n_noise)
        weight_signal = total / (2 * n_signal)
        
        return torch.tensor([weight_noise, weight_signal], dtype=torch.float32)


class MLPFInferenceDataset(Dataset):
    """
    MLPF 추론용 Dataset (레이블 없는 이벤트 처리)
    """
    
    def __init__(
        self,
        events: np.ndarray,
        patch_size: int = 7,
        tau_seconds: float = 0.1,
        width: int = 1280,
        height: int = 720
    ):
        """
        Args:
            events: (N, 4+) 배열 [x, y, t, polarity, ...]
            patch_size: 패치 크기
            tau_seconds: 시간 윈도우
            width, height: 이미지 크기
        """
        self.patch_generator = TIPatchGenerator(
            patch_size=patch_size,
            tau_seconds=tau_seconds,
            width=width,
            height=height
        )
        
        # label 없는 이벤트 처리
        if events.shape[1] >= 5:
            self.patches, _ = self.patch_generator.process_events(events)
        else:
            # label이 없는 경우, dummy label 추가
            dummy_events = np.column_stack([np.zeros(len(events)), events])
            self.patches, _ = self.patch_generator.process_events(dummy_events)
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.patches[idx])


def create_dataloaders(
    fps: int,
    data_root: str = "/local_data/EventGuard/EventSNN/data/DVSCLEAN_FRAME",
    batch_size: int = 4096,
    num_workers: int = 4,
    max_files: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    FPS별 DataLoader 생성
    
    Args:
        fps: FPS 값 (30, 60, 90, 120)
        data_root: 데이터 루트 경로
        batch_size: 배치 크기
        num_workers: DataLoader worker 수
        max_files: 디버깅용 최대 파일 수
    
    Returns:
        train_loader, val_loader, test_50_loader, test_100_loader
    """
    # τ = 1 / FPS (프레임 기간)
    tau_seconds = 1.0 / fps
    
    print(f"\n{'='*60}")
    print(f"FPS: {fps}, τ = {tau_seconds*1000:.1f}ms")
    print(f"{'='*60}")
    
    fps_folder = os.path.join(data_root, f"fps{fps}")
    test_folder = os.path.join(fps_folder, "test")
    
    # 데이터셋 생성
    print("\n📂 Train 데이터셋 로딩...")
    train_dataset = MLPFDataset(
        os.path.join(fps_folder, "train"),
        tau_seconds=tau_seconds,
        max_files=max_files,
        verbose=True
    )
    
    print("\n📂 Validation 데이터셋 로딩...")
    val_dataset = MLPFDataset(
        os.path.join(fps_folder, "val"),
        tau_seconds=tau_seconds,
        max_files=max_files,
        verbose=True
    )
    
    print("\n📂 Test_50 데이터셋 로딩...")
    test_50_dataset = MLPFDataset(
        os.path.join(test_folder, "test_50"),
        tau_seconds=tau_seconds,
        max_files=max_files,
        verbose=True
    )
    
    print("\n📂 Test_100 데이터셋 로딩...")
    test_100_dataset = MLPFDataset(
        os.path.join(test_folder, "test_100"),
        tau_seconds=tau_seconds,
        max_files=max_files,
        verbose=True
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_50_loader = DataLoader(
        test_50_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_100_loader = DataLoader(
        test_100_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_50_loader, test_100_loader


if __name__ == "__main__":
    # 테스트
    print("TIPatchGenerator 테스트...")
    
    # 간단한 테스트 데이터
    events = np.array([
        [1, 100, 100, 0.01, 1],  # signal, ON
        [0, 101, 100, 0.02, 0],  # noise, OFF
        [1, 100, 101, 0.03, 1],  # signal, ON
    ])
    
    generator = TIPatchGenerator(patch_size=7, tau_seconds=0.1)
    patches, labels = generator.process_events(events)
    
    print(f"입력 이벤트 수: {len(events)}")
    print(f"패치 shape: {patches.shape}")
    print(f"레이블: {labels}")
    print(f"첫 번째 패치 (TI): {patches[0, :49]}")  # 7x7 = 49
    print(f"첫 번째 패치 (Pol): {patches[0, 49:]}")  # 7x7 = 49
