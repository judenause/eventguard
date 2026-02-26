#!/usr/bin/env python3
"""
Hardware Efficiency Comparison - Paper-Verified Values
Based on Table 1 from "Low Cost and Latency Event Camera BA Denoising"
"""

# ============================================
# Resolution
# ============================================
N = 1280 * 720  # 921,600 pixels (HD)
N_squared_KB = N / 1024  # ~900 KB

print("=" * 80)
print(" 🔬 Hardware Efficiency Comparison (Paper Table 1 Values)")
print("=" * 80)
print(f"Resolution: 1280x720 = {N:,} pixels")
print("=" * 80)

# ============================================
# Paper-Verified Values (Table 1)
# ============================================
paper_models = {
    # (Ops/Event, Memory Description, Memory Size)
    "BAF": (11, "N²", f"{N_squared_KB:.0f} KB"),
    "STCF": (25, "N²", f"{N_squared_KB:.0f} KB"),
    "ONF (NN Filter)": (40, "4N", f"{4*N/1024:.0f} KB"),
    "HashHeat": (40, "Lʰ", "Variable"),
    "MLPF (98-20-1)": (2000, "N² + MLP", f"{N_squared_KB + 2:.0f} KB"),
    "EDnCNN": (167_000_000, ">48M", ">48 MB"),
}

# ============================================
# v8_bconvsnn Calculation
# ============================================
print("\n📌 v8_bconvsnn Calculation")
print("-" * 60)

# Architecture: BinaryConv 1->16 (3x3) -> 16->32 (3x3) -> 32->2 (3x3)
# All stride 1, maintaining resolution

# --- Weights Memory ---
# L1: 1 * 16 * 3 * 3 = 144 bits
# L2: 16 * 32 * 3 * 3 = 4608 bits
# L3: 32 * 2 * 3 * 3 = 576 bits
# Total: 5328 bits = 666 bytes ≈ 0.65 KB
weights_bits = (1*16*9) + (16*32*9) + (32*2*9)
weights_kb = weights_bits / 8 / 1024
print(f"Weights (Binary): {weights_bits:,} bits = {weights_kb:.2f} KB")

# --- State Memory (Membrane Potential) ---
# Need to store membrane potential for SNN layer
# L1 output: N × 16 channels × 8-bit = 14.7 MB (Dense)
# For Sparse: only store active neurons
membrane_dense_mb = N * 16 * 1 / 1024 / 1024  # 8-bit per neuron
print(f"Membrane (Dense): {membrane_dense_mb:.1f} MB")

# --- Ops/Event ---
# Dense (Current): Process entire frame, amortize over events
event_density = 0.05
events_per_frame = int(N * event_density)
sparse_rate = 0.10

# Dense mode
l1_ops_dense = N * 1 * 16 * 9  # 132.7M
l2_ops_dense = int(N * 16 * 32 * 9 * sparse_rate)  # 424.7M * 0.1
l3_ops_dense = int(N * 32 * 2 * 9 * sparse_rate)   # 53.1M * 0.1
total_dense = l1_ops_dense + l2_ops_dense + l3_ops_dense
ops_per_event_dense = total_dense / events_per_frame

print(f"\nDense Mode (Current):")
print(f"  Ops/Frame: {total_dense:,} ({total_dense/1e6:.0f}M)")
print(f"  Events/Frame: {events_per_frame:,}")
print(f"  Ops/Event: {ops_per_event_dense:,.0f}")

# Sparse mode (Future - event-driven)
# Only process 3x3 neighborhood at event location
l1_ops_sparse = 1 * 16 * 9  # 144
l2_ops_sparse = int(16 * 32 * 9 * sparse_rate)  # 461
l3_ops_sparse = int(32 * 2 * 9 * sparse_rate)   # 58
ops_per_event_sparse = l1_ops_sparse + l2_ops_sparse + l3_ops_sparse

print(f"\nSparse Mode (Future):")
print(f"  L1: {l1_ops_sparse} Ops")
print(f"  L2: {l2_ops_sparse} Ops (10% sparse)")
print(f"  L3: {l3_ops_sparse} Ops (10% sparse)")
print(f"  Ops/Event: {ops_per_event_sparse}")

# Memory for sparse mode
# N² for timestamp (like BAF) + weights
sparse_mem_kb = N_squared_KB + weights_kb
print(f"\nSparse Memory: N² + weights = {sparse_mem_kb:.0f} KB")

# ============================================
# Full Comparison Table
# ============================================
print("\n" + "=" * 80)
print(" 📊 COMPARISON TABLE (Paper Format)")
print("=" * 80)

all_models = [
    ("BAF [13]", "N²", 11, f"{N_squared_KB:.0f} KB"),
    ("STCF", "N²", 25, f"{N_squared_KB:.0f} KB"),
    ("ONF (NN)", "4N", 40, f"{4*N/1024/1024:.1f} MB"),
    ("HashHeat", "Lʰ", 40, "Variable"),
    ("MLPF (98-20-1)", "N² + MLP", 2000, f"{N_squared_KB + 2:.0f} KB"),
    ("v8 (Sparse)", "N² + 0.7KB", ops_per_event_sparse, f"{sparse_mem_kb:.0f} KB"),
    ("v8 (Dense)", f"{membrane_dense_mb:.0f}MB", int(ops_per_event_dense), f"{membrane_dense_mb:.0f} MB"),
    ("EDnCNN [30]", ">48M", 167_000_000, ">48 MB"),
]

# Sort by Ops/Event
all_models.sort(key=lambda x: x[2])

def fmt_ops(n):
    if n < 1000: return str(n)
    elif n < 1e6: return f"≈{n/1000:.0f}k"
    else: return f"{n/1e6:.0f}M"

print(f"\n{'Filter':<20} {'Mem(#)':>12} {'Op/event':>12}")
print("-" * 50)

for name, mem_formula, ops, mem_size in all_models:
    marker = " ⭐" if "v8" in name else ""
    print(f"{name:<20} {mem_formula:>12} {fmt_ops(ops):>12}{marker}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 80)
print(" 🏆 Key Results")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  v8_bconvsnn Sparse vs Others                                       │
├─────────────────────────────────────────────────────────────────────┤
│  vs BAF:      {ops_per_event_sparse / 11:.1f}x more Ops (but learned denoising!)      │
│  vs STCF:     {ops_per_event_sparse / 25:.1f}x more Ops (but HD + learning!)          │
│  vs MLPF:     {ops_per_event_sparse / 2000:.2f}x fewer Ops 🔥 (with Full HD!)           │
│  vs EDnCNN:   {167_000_000 / ops_per_event_sparse:,.0f}x fewer Ops 🚀                          │
├─────────────────────────────────────────────────────────────────────┤
│  Memory:      N² + 0.7KB (same as BAF/STCF + tiny weights)          │
└─────────────────────────────────────────────────────────────────────┘
""")
