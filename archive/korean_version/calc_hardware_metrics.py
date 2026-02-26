import torch
import numpy as np
from config import cfg

def estimate_hardware_metrics():
    print("==================================================")
    print(" 🖥️  v8_bconvsnn Hardware Metrics Estimation")
    print("==================================================")
    
    # --- 1. Architecture Definitions ---
    H, W = cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH
    T = cfg.WINDOW_SIZE
    
    # Layer 1: SNN (BinaryConv2d)
    # Input: 1 channel (Events)
    # Output: 16 channels
    # Kernel: 3x3
    L1_in_c = 1
    L1_out_c = 16
    L1_k = 3
    
    # Layer 2: BNN (BinaryConv2d)
    # Input: 16 channels
    # Output: 32 channels
    # Kernel: 3x3
    L2_in_c = 16
    L2_out_c = 32
    L2_k = 3
    
    # Layer 3: BNN (BinaryConv2d) - Final
    # Input: 32 channels
    # Output: 2 channels
    # Kernel: 3x3
    L3_in_c = 32
    L3_out_c = 2
    L3_k = 3
    
    print(f"Resolution: {W}x{H}")
    print(f"Time Steps (T): {T}")
    print(f"Architecture: {L1_in_c}->{L1_out_c} (SNN) -> {L2_out_c} (BNN) -> {L3_out_c} (BNN)")
    print("-" * 50)

    # --- 2. Memory Footprint (Weights) ---
    # Binary Weights = 1 bit per weight
    # SNN Layer (L1)
    w1_bits = L1_in_c * L1_out_c * L1_k * L1_k
    # BNN Layer (L2)
    w2_bits = L2_in_c * L2_out_c * L2_k * L2_k
    # Final Layer (L3)
    w3_bits = L3_in_c * L3_out_c * L3_k * L3_k
    
    total_weight_bits = w1_bits + w2_bits + w3_bits
    total_weight_kb = total_weight_bits / 8 / 1024
    
    print(f"💾 Model Size (Weights):")
    print(f"  - Total Bits: {total_weight_bits:,}")
    print(f"  - Total Size: {total_weight_kb:.2f} KB (Extremely Small!)")
    
    # --- 3. State Memory (Activations/Membrane) ---
    # SNN Layer: Needs Membrane Potential (Int8 or Int16) per pixel per channel
    # Let's assume Int8 (8 bits) for membrane potential
    mem_bits_l1 = H * W * L1_out_c * 8 
    
    # BNN Layers: Need Input Feature Maps (Binary - 1 bit) or Accumulators?
    # BNNs usually need to store the input binary map for the next layer
    mem_bits_l2_in = H * W * L2_in_c * 1
    mem_bits_l3_in = H * W * L3_in_c * 1
    
    total_state_bits = mem_bits_l1 + mem_bits_l2_in + mem_bits_l3_in
    total_state_mb = total_state_bits / 8 / 1024 / 1024
    
    print(f"🧠 State Memory (RAM):")
    print(f"  - Total Size: {total_state_mb:.2f} MB")
    print("-" * 50)

    # --- 4. Operations Count (Per Window) ---
    # Scenario: 5000 events in a window (Typical for test_50)
    num_events = 5000
    
    # L1 (SNN): Event-Driven
    # Ops = Input Events * Kernel_Size^2 * Out_Channels
    # These are Accumulations (AC), not MACs
    ops_l1 = num_events * (L1_k * L1_k) * L1_out_c
    
    # L2 (BNN): Frame-Based (Dense)
    # Ops = H * W * In_Channels * Out_Channels * Kernel_Size^2 * Time_Steps
    # These are XNOR-Popcount Ops (BOPs)
    ops_l2 = (H * W) * L2_in_c * L2_out_c * (L2_k * L2_k) * T
    
    # L3 (BNN): Frame-Based (Dense)
    ops_l3 = (H * W) * L3_in_c * L3_out_c * (L3_k * L3_k) * T
    
    total_ops = ops_l1 + ops_l2 + ops_l3
    ops_per_event = total_ops / num_events
    
    print(f"⚡ Operations (Assuming {num_events} events/window):")
    print(f"  - L1 (SNN, Event-Driven): {ops_l1:,} SOPs (Accumulations)")
    print(f"  - L2 (BNN, Dense):        {ops_l2:,} BOPs (XNORs)")
    print(f"  - L3 (BNN, Dense):        {ops_l3:,} BOPs (XNORs)")
    print(f"  - Total Ops:              {total_ops:,}")
    print(f"  - Effective Ops/Event:    {ops_per_event:,.0f} Ops/Event")
    
    print("\n⚠️  Note: BNN layers are dense, so Ops/Event is very high.")
    print("    However, XNOR ops are ~58x cheaper than FP32 MACs in hardware.")
    
if __name__ == "__main__":
    estimate_hardware_metrics()
