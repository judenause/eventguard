# custom_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn # QuantLeaky를 위해 필요


# ===================================================================
# Weight Binarization
# ===================================================================
class BinarizeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        b_weight = torch.zeros_like(weight)
        b_weight[weight >= 0] = 1
        b_weight[weight < 0] = -1
        return b_weight

    @staticmethod
    def backward(ctx, grad_out):
        weight, = ctx.saved_tensors
        grad_in = grad_out.clone()
        return grad_in

# ===================================================================
# Activation Binarization
# ===================================================================
class BinarizeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, act):
        ctx.save_for_backward(act)
        b_act = torch.zeros_like(act)
        b_act[act >= 0] = 1
        b_act[act < 0] = -1
        return b_act

    @staticmethod
    def backward(ctx, grad_out):
        act, = ctx.saved_tensors
        grad_in = grad_out.clone()
        return grad_in

# ===================================================================
# Simplified BinaryConv2d
# ===================================================================
class BinaryConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinaryConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input, regulate = False):
        if (regulate):
            # 정규화된 가중치를 이진화
            binarized_weights = (self.weight - self.weight.mean()) / self.weight.std()
            binarized_weights = BinarizeWeight.apply(binarized_weights)
        else:
            # 원본 가중치를 바로 이진화
            binarized_weights = BinarizeWeight.apply(self.weight)

        # DAC2026 style: Scale only during TRAINING
        # Inference outputs INTEGER for proper quantized comparison
        if self.training:
            scaling_factor = self.weight.abs().sum() / self.weight.numel()
            binarized_weights = binarized_weights * scaling_factor

        return F.conv2d(input, binarized_weights, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None)

    def reset_parameters(self):
        # 가중치 초기화
        nn.init.xavier_normal_(self.weight)

# ===================================================================
# DAC2026-style Membrane Quantization (STE)
# ===================================================================
class QuantizeMem(torch.autograd.Function):
    """
    DAC2026-style membrane potential quantization.
    q_x = scale * trunc(beta * mem / scale) + input
    STE for backward pass.
    """
    @staticmethod
    def forward(ctx, mem, input_, beta, scale):
        # Quantize membrane: scale * trunc(beta * mem / scale)
        q_mem = scale * torch.trunc(beta.clamp(0, 1) * mem / scale)
        # Add input (float)
        q_x = q_mem + input_
        return q_x
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient to mem, input, beta (not scale)
        return grad_output, grad_output, grad_output, None

# ===================================================================
# DAC2026-style Threshold Quantization (STE)
# ===================================================================
class QuantizeThr(torch.autograd.Function):
    """
    DAC2026-style threshold quantization.
    q_thr = scale * floor(threshold / scale)
    STE for backward pass.
    """
    @staticmethod
    def forward(ctx, threshold, scale):
        q_thr = scale * torch.floor(threshold / scale)
        return q_thr
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient to threshold (not scale)
        return grad_output, None

# ===================================================================
# Power-of-2 Quantization (for Beta)
# ===================================================================
class PowerOfTwo(torch.autograd.Function):
    """
    Quantizes input to nearest power-of-2 (2^-n form).
    Used for Beta parameter quantization.
    STE for backward pass.
    """
    @staticmethod
    def forward(ctx, input):
        # 2^-n 형태로 양자화 (0 < input <= 1 가정)
        ctx.save_for_backward(input)
        
        # 0 이하인 경우 처리 (Beta는 양수여야 함)
        input_clamped = input.clamp(min=1e-6, max=1.0)
        
        log_val = torch.log2(input_clamped)
        round_log = torch.round(log_val)
        quantized = torch.pow(2, round_log)
        
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient unchanged
        return grad_output

# ===================================================================
# Quantized Leaky Neuron (DAC2026-aligned)
# ===================================================================
class QuantLeaky(snn.LIF):
    def __init__(self, beta, threshold=1.0, spike_grad=None, reset_mechanism="zero", 
                 learn_beta=False, learn_threshold=False, quantize_beta_power2=True, **kwargs):
        super().__init__(beta=beta, threshold=threshold, spike_grad=spike_grad, reset_mechanism=reset_mechanism, **kwargs)

        self.quantize_beta_power2 = quantize_beta_power2
        
        # Pre-computed quantized threshold (set after training via compute_quantized_threshold())
        self.register_buffer('quantized_threshold', None)

        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        if learn_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))

    def compute_quantized_threshold(self, scale, thr_bit=4):
        """
        Pre-compute and cache the quantized threshold for HW inference.
        Call this AFTER training is complete, BEFORE inference.
        
        Args:
            scale: Scaling factor from conv layer (weight.abs().sum()/numel())
            thr_bit: Bit width for threshold (default 4)
        """
        thr_val = self.threshold.abs() if isinstance(self.threshold, torch.Tensor) else self.threshold
        thr_max = (2**(thr_bit-1)-1)  # For 4bit: 7
        thr_min = -(2**(thr_bit-1))   # For 4bit: -8
        
        # Compute and cache quantized threshold
        q_thr = torch.clamp(torch.floor(thr_val / scale), min=thr_min, max=thr_max)
        self.quantized_threshold = q_thr.detach()
        
        return self.quantized_threshold

    def forward(self, input_, mem=None, out_scale=1.0, mem_bit=8, thr_bit=4):
        """
        Forward pass with DAC2026-style scale-based quantization.
        
        Args:
            input_: Input tensor from conv layer
            mem: Previous membrane potential (None = initialize to zeros)
            out_scale: Scaling factor from conv layer weights (weight.abs().sum()/numel())
            mem_bit: Bit width for membrane potential (default 8)
            thr_bit: Bit width for threshold (default 4)
        """
        if mem is None:
            mem = torch.zeros_like(input_, device=input_.device)

        self.reset = super().fire(mem)
        
        # Dispatch based on reset_mechanism_val directly
        if self.reset_mechanism_val == 0:
            mem = self._base_sub(input_, mem, out_scale, mem_bit)
        elif self.reset_mechanism_val == 1:
            mem = self._base_zero(input_, mem, out_scale, mem_bit)
        else:
            mem = self._base_int(input_, mem, out_scale, mem_bit)
            
        spk = self.fire(mem, out_scale, thr_bit)
        return spk, mem

    def _base_state_function(self, input_, mem, scale, q_bit=8):
        """
        DAC2026-style membrane calculation:
        Training: Float (beta * mem + input)
        Inference: Quantized (scale * trunc(beta * mem / scale) + input)
        """
        # Clamp beta to [0, 1] for stability
        beta_val = self.beta.clamp(0, 1)
        
        # --- Power-of-2 Quantization for Beta (always applied) ---
        if self.quantize_beta_power2:
            beta_val = PowerOfTwo.apply(beta_val)
        
        # --- DAC2026-style: Float for Training, Quant for Inference ---
        if self.training:
            # Training: Use float arithmetic
            base_fn = beta_val * mem + input_
        else:
            # Inference: Use quantized arithmetic
            mem_max = (2**(q_bit-1)-1)  # For 8bit: 127
            mem_min = -(2**(q_bit-1))   # For 8bit: -128
            base_fn = torch.clamp(torch.trunc(beta_val * mem) + input_, min=mem_min, max=mem_max)
        
        return base_fn

    def _base_sub(self, input_, mem, scale, q_bit):
        # Clamp threshold to be positive
        thr_val = self.threshold.abs() if isinstance(self.threshold, torch.Tensor) else self.threshold
        return self._base_state_function(input_, mem, scale, q_bit) - self.reset * thr_val

    def _base_zero(self, input_, mem, scale, q_bit):
        mem = (1 - self.reset) * mem
        return self._base_state_function(input_, mem, scale, q_bit)

    def _base_int(self, input_, mem, scale, q_bit):
        return self._base_state_function(input_, mem, scale, q_bit)

    def fire(self, mem, scale, q_bit=4):
        """
        Spike generation with threshold comparison.
        Training: Float threshold
        Inference: Use pre-computed quantized threshold (integer comparison)
        """
        if self.training:
            # Training: Use float threshold for gradient flow
            if isinstance(self.threshold, torch.Tensor):
                thr_val = self.threshold.abs()
            else:
                thr_val = self.threshold
            q_thr = thr_val
        else:
            # Inference: Use pre-computed quantized threshold (integer)
            if self.quantized_threshold is not None:
                q_thr = self.quantized_threshold
            else:
                # Fallback: compute on-the-fly if not pre-computed
                thr_val = self.threshold.abs() if isinstance(self.threshold, torch.Tensor) else self.threshold
                thr_max = (2**(q_bit-1)-1)
                thr_min = -(2**(q_bit-1))
                q_thr = torch.clamp(torch.floor(thr_val / scale), min=thr_min, max=thr_max)

        mem_shift = mem - q_thr
        spk = self.spike_grad(mem_shift)
        return spk