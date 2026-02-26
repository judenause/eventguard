# model.py
import torch
import torch.nn as nn
from snntorch import surrogate

# 위에서 완성된 custom_layers.py를 임포트
from custom_layers import BinaryConv2d, BinarizeAct, QuantLeaky

from config import cfg

class Hybrid_SNN_Pure_BNN(nn.Module):
    """
    BatchNorm이 없는 SNN-BNN 하이브리드 모델.
    - 첫 번째 블록: SNN (BinaryConv2d -> QuantLeaky)
    - 이후 블록: Pure BNN (BinaryConv2d -> BinarizeAct)
    """
    def __init__(self,
                 input_channels: int,
                 output_classes: int,
                 snn_params: dict,
                 conv_channels: list[int] = [16, 32],
                 kernel_sizes: list[int] = [3, 3, 3]):
        super().__init__()

        spike_grad = surrogate.atan()

        if len(kernel_sizes) < len(conv_channels) + 1:
            kernel_sizes.extend([kernel_sizes[-1]] * (len(conv_channels) + 1 - len(kernel_sizes)))

        self.layers = nn.ModuleList()
        current_in_channels = input_channels

        # --- Layer Construction ---
        # 1. First Block (SNN)
        first_out_channels = conv_channels[0]
        # Dilation for first layer (usually 1)
        d1 = cfg.DILATION_RATES[0] if hasattr(cfg, 'DILATION_RATES') and len(cfg.DILATION_RATES) > 0 else 1
        self.snn_conv = BinaryConv2d(current_in_channels, first_out_channels,
                                     kernel_size=kernel_sizes[0], 
                                     padding=kernel_sizes[0]//2 * d1, # Padding must adjust for dilation to keep size
                                     dilation=d1)
        # Learnable Parameters Enabled
        self.snn_act = QuantLeaky(beta=snn_params['beta'], threshold=snn_params.get('threshold', 1.0),
                                  spike_grad=spike_grad, reset_mechanism="zero",
                                  learn_beta=True, learn_threshold=True)
        current_in_channels = first_out_channels

        # 2. Intermediate Blocks (Pure BNN)
        self.bnn_layers = nn.ModuleList()
        for i in range(1, len(conv_channels)):
            num_out_channels = conv_channels[i]
            # Dilation for intermediate layers
            d_rate = cfg.DILATION_RATES[i] if hasattr(cfg, 'DILATION_RATES') and len(cfg.DILATION_RATES) > i else 1
            
            self.bnn_layers.append(
                BinaryConv2d(current_in_channels, num_out_channels,
                             kernel_size=kernel_sizes[i], 
                             padding=kernel_sizes[i]//2 * d_rate, # Adjust padding
                             dilation=d_rate)
            )
            current_in_channels = num_out_channels

        # 3. Final BNN Layer
        # Dilation for final layer
        d_final = cfg.DILATION_RATES[len(conv_channels)] if hasattr(cfg, 'DILATION_RATES') and len(cfg.DILATION_RATES) > len(conv_channels) else 1
        self.final_conv = BinaryConv2d(current_in_channels, output_classes,
                                     kernel_size=kernel_sizes[len(conv_channels)],
                                     padding=kernel_sizes[len(conv_channels)]//2 * d_final,
                                     dilation=d_final)

    def forward(self, x: torch.Tensor, mem: torch.Tensor = None, regulate=False):
        """
        Args:
            x (torch.Tensor): 입력 텐서 [Batch, Time, Channels, Height, Width].
            mem (torch.Tensor, optional): 초기 SNN 막전위 상태. Defaults to None.
            regulate (bool): 가중치 정규화 사용 여부.

        Returns:
            torch.Tensor: 출력 로짓 [Batch, Time, Output_Classes, Height, Width].
            torch.Tensor: 업데이트된 SNN 막전위 상태.
        """
        # mem이 None이면 내부에서 초기화 (Stateless 또는 첫 스텝)
        # QuantLeaky가 None을 받으면 0으로 초기화함
        
        outputs_over_time = []

        for step in range(x.size(1)): # 시간 축 루프
            x_step = x[:, step, ...]

            # SNN Layer
            cur = self.snn_conv(x_step, regulate=regulate)
            
            # DAC2026-style: Calculate scale from SNN conv layer weights
            out_scale = self.snn_conv.weight.abs().sum() / self.snn_conv.weight.numel()
            
            # Pass out_scale to QuantLeaky for proper quantization
            spk, mem = self.snn_act(cur, mem, out_scale=out_scale, mem_bit=8, thr_bit=4)
            bnn_input = spk

            # BNN Layers
            for layer in self.bnn_layers:
                # DAC2026 style: Pass {0, 1} from SNN directly to next Conv.
                # Do NOT convert to {-1, 1} here.
                cur = layer(bnn_input, regulate=regulate)
                
                # Residual Connection (Skip Connection)
                if cfg.USE_RESIDUAL and bnn_input.shape == cur.shape:
                    cur = cur + bnn_input
                
                bnn_input = BinarizeAct.apply(cur)

            # Final BNN Layer
            final_input = bnn_input
            output_logits = self.final_conv(final_input, regulate=regulate)
            outputs_over_time.append(output_logits)

        return torch.stack(outputs_over_time, dim=1), mem

    def prepare_for_inference(self, thr_bit=4):
        """
        Pre-compute quantized threshold for HW inference.
        Call this AFTER training is complete, BEFORE running inference.
        """
        scale = self.snn_conv.weight.abs().sum() / self.snn_conv.weight.numel()
        q_thr = self.snn_act.compute_quantized_threshold(scale, thr_bit)
        print(f"✅ Pre-computed quantized threshold: {q_thr.item():.0f} (scale={scale:.4f})")
        return q_thr