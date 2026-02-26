# -*- coding: utf-8 -*-
"""
MLPF(MultiLayer Perceptron Filter) PyTorch 모델 정의

MLPF 논문 구조: 98 -> 20 -> 1 (Sigmoid)
- 입력: 7x7 TI 패치 + 7x7 Polarity 패치 = 98 features
- 은닉층: 20 neurons with ReLU
- 출력: 1 (Signal probability via Sigmoid)

Reference: Guo & Delbruck, T-PAMI 2022
"""

import torch
import torch.nn as nn


class MLPF(nn.Module):
    """
    MLPF: Multilayer Perceptron Filter for Event Denoising
    
    원본 MLPF (from MLPF.ipynb):
    - Architecture: 49 -> 20 -> 1 (TI only)
    - Loss: MSE
    
    확장 버전 (TI + Polarity, from MLPF.java):
    - Architecture: 98 -> 20 -> 1
    """
    
    def __init__(
        self,
        patch_size: int = 7,
        hidden_size: int = 20,
        use_polarity: bool = True,  # Java 버전 기본값 (TI + Polarity)
        dropout: float = 0.2
    ):
        """
        Args:
            patch_size: TI 패치 크기 (default: 7x7)
            hidden_size: 은닉층 뉴런 수 (default: 20)
            use_polarity: Polarity 채널 사용 여부 (default: True, Java MLPF 기준)
            dropout: Dropout 비율 (default: 0.2)
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_polarity = use_polarity
        
        # 입력 크기 결정
        if use_polarity:
            input_size = 2 * patch_size * patch_size  # TI + Polarity = 98
        else:
            input_size = patch_size * patch_size  # TI only = 49
        
        self.input_size = input_size
        
        # MLP 구조 정의
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            # Sigmoid는 BCEWithLogitsLoss 사용시 생략
        )
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch_size, input_size) 형태의 입력 텐서
        
        Returns:
            logits: (batch_size, 1) 형태의 출력 (logit, Sigmoid 적용 전)
        """
        return self.mlp(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        예측 수행 (Signal/Noise 분류)
        
        Args:
            x: 입력 텐서
            threshold: Signal 분류 임계값 (default: 0.5)
        
        Returns:
            predictions: (batch_size,) 형태의 예측 (1=Signal, 0=Noise)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze(-1)
            return (probs > threshold).long()
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        확률 예측
        
        Args:
            x: 입력 텐서
        
        Returns:
            probs: (batch_size,) 형태의 Signal 확률
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)


class MLPFLarge(nn.Module):
    """
    더 큰 MLPF 변형 (실험용)
    
    Architecture: 98 -> 64 -> 32 -> 1
    """
    
    def __init__(self, patch_size: int = 7, use_polarity: bool = True):
        super().__init__()
        
        if use_polarity:
            input_size = 2 * patch_size * patch_size
        else:
            input_size = patch_size * patch_size
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


if __name__ == "__main__":
    # 기본 MLPF 테스트 (TI + Polarity, Java 버전)
    print("="*60)
    print("MLPF (TI + Polarity, 98 inputs) - Java 버전 기본값")
    print("="*60)
    model = MLPF(patch_size=7, hidden_size=20, use_polarity=True)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")
    
    batch_size = 32
    x = torch.randn(batch_size, 98)  # 7x7x2 = 98
    logits = model(x)
    print(f"입력 shape: {x.shape}, 출력 shape: {logits.shape}")
    
    # 예측 테스트
    preds = model.predict(x)
    print(f"예측 shape: {preds.shape}")
    print(f"Signal 예측 비율: {preds.float().mean():.2%}")
    
    # 원본 MLPF 테스트 (TI only)
    print("\n" + "="*60)
    print("원본 MLPF (TI only, 49 inputs) - 노트북 버전")
    print("="*60)
    model_ti = MLPF(patch_size=7, hidden_size=20, use_polarity=False)
    print(model_ti)
    
    total_params = sum(p.numel() for p in model_ti.parameters())
    print(f"총 파라미터 수: {total_params:,}")
    
    x_ti = torch.randn(batch_size, 49)  # 7x7 = 49
    logits_ti = model_ti(x_ti)
    print(f"입력 shape: {x_ti.shape}, 출력 shape: {logits_ti.shape}")
    
    print("\n✅ 모델 테스트 완료!")
