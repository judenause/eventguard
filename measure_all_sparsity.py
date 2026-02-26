"""
DVSCLEAN Dataset 전체 Sparsity 측정 스크립트 (L1, L2 포함)
- Event Rate: 입력 프레임의 Non-zero 비율
- L1 Spike Rate: SNN 출력의 Spike 비율
- L2 Effective Sparsity: BNN 출력에서 baseline과 다른 비율

Usage: python measure_all_sparsity.py --dataset test_50
"""

import torch
import numpy as np
import argparse
import sys
import os
import glob
from collections import OrderedDict
from tqdm import tqdm

sys.path.insert(0, '/local_data/EventGuard/EventSNN/code/v8_bconvsnn')
from config import cfg
from model import Hybrid_SNN_Pure_BNN
from dataset import EventFrameLazyDataset
from data_processing import process_folder_to_frame_lists

# ===================================================================
# Hooks to capture intermediate outputs
# ===================================================================
l1_outputs = []
l2_outputs = []

def l1_spike_hook(module, input, output):
    """L1 SNN 출력(spike)을 캡처"""
    spike = output[0]  # (spike, mem) tuple
    rate = (spike != 0).float().mean().item()
    l1_outputs.append(rate)

def l2_output_hook(module, input, output):
    """L2 BNN 출력을 캡처 (BinarizeAct 후)"""
    # output은 BinarizeAct.apply(cur) 결과: {-1, +1}
    # Effective Sparsity: 가장 많은 값(baseline)과 다른 비율
    out = output.detach()
    
    # Mode (가장 빈번한 값) 찾기
    unique, counts = torch.unique(out, return_counts=True)
    mode_val = unique[counts.argmax()].item()
    
    # Baseline과 다른 비율
    diff_rate = (out != mode_val).float().mean().item()
    l2_outputs.append(diff_rate)

# ===================================================================
# Raw Data Event Rate 확인 (캐시 전 데이터)
# ===================================================================
def check_raw_event_rate(data_folder, num_files=5):
    """원본 .npy 파일에서 직접 Event Rate 확인"""
    files = sorted(glob.glob(os.path.join(data_folder, '*.npy')))[:num_files]
    
    event_counts = []
    total_events = 0
    total_time = 0
    
    print(f"\n🔍 Raw Data Event Rate Check ({len(files)} files)")
    print("-" * 50)
    
    for f in files:
        data = np.load(f, allow_pickle=True)
        
        # 데이터 구조 확인
        if len(data) > 0:
            # 첫 번째 아이템 체크
            sample = data[0]
            total_events += len(data)
            
            # 시간 범위 계산
            if hasattr(sample, '__len__') and len(sample) >= 4:
                # [x, y, t, p] 형태
                timestamps = [d[2] for d in data if len(d) >= 4]
                if timestamps:
                    duration = (max(timestamps) - min(timestamps)) / 1e6  # μs -> s
                    total_time += duration
            
            print(f"  {os.path.basename(f)}: {len(data):,} events")
    
    if total_time > 0:
        events_per_sec = total_events / total_time
        print(f"\n  Total: {total_events:,} events over {total_time:.2f}s")
        print(f"  Rate: {events_per_sec:,.0f} events/sec")
    else:
        print(f"\n  Total: {total_events:,} events")
    
    return total_events

# ===================================================================
# Main Measurement
# ===================================================================
def measure_all_sparsity(dataset_name, model_path, fps=30, num_samples=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cfg.FPS = fps
    if dataset_name == 'test_50':
        data_folder = '/local_data/EventGuard/EventSNN/data/esd/total/test_50/'
    elif dataset_name == 'test_100':
        data_folder = '/local_data/EventGuard/EventSNN/data/esd/total/test_100/'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print("=" * 60)
    print(f" Dataset: {dataset_name} | FPS: {fps}")
    print("=" * 60)
    
    # 0. Raw Data 확인
    check_raw_event_rate(data_folder, num_files=3)
    
    # 1. Load Data
    print("\n[1/3] Loading processed frame data...")
    frame_lists = process_folder_to_frame_lists(data_folder, '*.npy', dataset_name, cfg)
    dataset = EventFrameLazyDataset(frame_lists[:5], cfg)
    
    # 직접 프레임 데이터 확인 (processed cache에서)
    cache_dir = f'./processed_cache/{dataset_name}'
    cache_files = sorted(glob.glob(os.path.join(cache_dir, '*.npy')))
    
    if cache_files:
        print("\n📊 Frame Data Statistics:")
        sample_data = np.load(cache_files[0])
        print(f"   Sample shape: {sample_data.shape}")  # [Frames, Channels, H, W]
        
        for i in range(min(3, len(sample_data))):
            frame = sample_data[i]
            nonzero = np.count_nonzero(frame)
            total = frame.size
            rate = nonzero / total * 100
            print(f"   Frame {i}: {nonzero:,} / {total:,} = {rate:.4f}%")
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Load Model & Register Hooks
    print("\n[2/3] Loading model and registering hooks...")
    snn_params = {'beta': cfg.SNN_BETA, 'threshold': cfg.SNN_THRESHOLD}
    model = Hybrid_SNN_Pure_BNN(input_channels=1, output_classes=2, snn_params=snn_params).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("   Model loaded.")
    
    # Hooks
    handle1 = model.snn_act.register_forward_hook(l1_spike_hook)
    
    # L2 출력 캡처를 위해 BNN layer 후에 hook 추가
    # model.bnn_layers[0] 이후의 출력을 캡처하려면 forward 수정 필요
    # 간단히 forward pass 중에 직접 측정
    
    model.eval()
    
    # 3. Measure
    print(f"\n[3/3] Measuring sparsity (samples: {num_samples})...")
    
    global l1_outputs, l2_outputs
    l1_outputs = []
    l2_outputs = []
    event_rates = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(loader, total=min(num_samples, len(loader)))):
            if batch_idx >= num_samples:
                break
            
            inputs = batch_data[0].to(device)
            
            # Event Rate
            for t in range(inputs.size(1)):
                frame = inputs[:, t, :, :, :]
                event_rates.append((frame != 0).float().mean().item())
            
            # Forward with manual L2 capture
            mem = None
            for step in range(inputs.size(1)):
                x_step = inputs[:, step, ...]
                
                # L1 (SNN)
                cur = model.snn_conv(x_step, regulate=True)
                out_scale = model.snn_conv.weight.abs().sum() / model.snn_conv.weight.numel()
                spk, mem = model.snn_act(cur, mem, out_scale=out_scale)
                l1_outputs.append((spk != 0).float().mean().item())
                
                # L2 (BNN)
                bnn_input = spk
                for layer in model.bnn_layers:
                    cur = layer(bnn_input, regulate=True)
                    if cfg.USE_RESIDUAL and bnn_input.shape == cur.shape:
                        cur = cur + bnn_input
                    from custom_layers import BinarizeAct
                    bnn_output = BinarizeAct.apply(cur)
                    
                    # L2 Effective Sparsity (첫 번째 BNN layer만)
                    unique, counts = torch.unique(bnn_output, return_counts=True)
                    mode_val = unique[counts.argmax()].item()
                    diff_rate = (bnn_output != mode_val).float().mean().item()
                    l2_outputs.append(diff_rate)
                    
                    bnn_input = bnn_output
    
    handle1.remove()
    
    # Results
    print("\n" + "=" * 60)
    print(f" RESULTS: {dataset_name} (FPS={fps})")
    print("=" * 60)
    
    avg_event = np.mean(event_rates) * 100
    avg_l1 = np.mean(l1_outputs) * 100
    avg_l2 = np.mean(l2_outputs) * 100 if l2_outputs else 0
    
    print(f"\n📥 Event Rate (Input): {avg_event:.4f}%")
    print(f"   Min: {min(event_rates)*100:.4f}%, Max: {max(event_rates)*100:.4f}%")
    
    print(f"\n⚡ L1 Spike Rate: {avg_l1:.4f}%")
    print(f"   Min: {min(l1_outputs)*100:.4f}%, Max: {max(l1_outputs)*100:.4f}%")
    
    print(f"\n🔲 L2 Effective Sparsity: {avg_l2:.4f}%")
    if l2_outputs:
        print(f"   Min: {min(l2_outputs)*100:.4f}%, Max: {max(l2_outputs)*100:.4f}%")
    
    # Energy 재계산 (L3도 Sparse 가정)
    events_per_frame = int(1280 * 720 * avg_event / 100)
    l1_rate = avg_l1 / 100
    l2_rate = avg_l2 / 100
    
    L1_ops = 1 * 16 * 9  # 144
    L2_ops = 16 * 32 * 9 * l1_rate
    L3_ops_dense = (921600 * 32 * 2 * 9) / events_per_frame if events_per_frame > 0 else 0
    L3_ops_sparse = L3_ops_dense * l2_rate  # L2 sparsity 적용
    
    total_dense = L1_ops + L2_ops + L3_ops_dense
    total_sparse = L1_ops + L2_ops + L3_ops_sparse
    
    print(f"\n💡 Energy Comparison:")
    print(f"   L1: {L1_ops:.1f} Ops/Event")
    print(f"   L2: {L2_ops:.1f} Ops/Event (L1 sparsity: {l1_rate*100:.4f}%)")
    print(f"   L3 (Dense):  {L3_ops_dense:.1f} Ops/Event")
    print(f"   L3 (Sparse): {L3_ops_sparse:.1f} Ops/Event (L2 sparsity: {l2_rate*100:.4f}%)")
    print(f"\n   Total (Dense):  {total_dense:.1f} Ops → {total_dense*0.03:.2f} pJ")
    print(f"   Total (Sparse): {total_sparse:.1f} Ops → {total_sparse*0.03:.2f} pJ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='test_50')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--model', type=str, 
                        default='/local_data/EventGuard/EventSNN/code/v8_bconvsnn/results/reproduction_v5_multigpu_fixedlr/latest_checkpoint.pth')
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    
    measure_all_sparsity(args.dataset, args.model, args.fps, args.samples)
