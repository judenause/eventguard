#!/bin/bash
# ==============================================================================
# Hotelbar 데이터셋에 대한 Classical Filter 평가
# 결과는 dataset별로 분류하여 저장
# ==============================================================================

cd /local_data/EventGuard/EventSNN/code/classical_filters

# Hotelbar 데이터셋 경로
HOTELBAR_TEST="/local_data/EventGuard/EventSNN/data/hotelbar/test"
HOTELBAR_VAL="/local_data/EventGuard/EventSNN/data/hotelbar/val"
HOTELBAR_TRAIN="/local_data/EventGuard/EventSNN/data/hotelbar/train"

# 결과 저장 경로
OUTPUT_DIR="./results/hotelbar"
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Classical Filter Evaluation on Hotelbar Dataset"
echo "============================================"

# Python 스크립트로 평가 수행
/opt/miniconda/envs/Event/bin/python << 'EOF'
import numpy as np
import os
import sys
import glob
import pandas as pd
from tqdm import tqdm

# Add classical_filters to path
sys.path.insert(0, '/local_data/EventGuard/EventSNN/code/classical_filters')

from config import cfg
from filters import create_filter, get_all_filter_names
from utils import (
    compute_event_stream_metrics,
    compute_frame_level_metrics,
    events_to_frame_predictions,
    format_hw_ops_summary
)

# Hotelbar용 설정 (DAVIS346: 346x260)
cfg.FRAME_WIDTH = 346
cfg.FRAME_HEIGHT = 260
cfg.FPS = 30  # 기본 FPS

# 평가할 데이터셋
datasets = [
    ('hotelbar_test', '/local_data/EventGuard/EventSNN/data/hotelbar/test'),
    ('hotelbar_val', '/local_data/EventGuard/EventSNN/data/hotelbar/val'),
    ('hotelbar_train', '/local_data/EventGuard/EventSNN/data/hotelbar/train'),
]

output_dir = './results/hotelbar'
os.makedirs(output_dir, exist_ok=True)

def evaluate_filter_on_file(filter_obj, events, filename):
    """단일 파일에 대해 필터 평가"""
    if len(events) == 0:
        return {'error': 'empty_file', 'filename': filename}
    
    # Apply filter
    predictions, hw_ops = filter_obj.filter_events(events)
    
    # Compute event-stream metrics
    stream_metrics = compute_event_stream_metrics(events, predictions)
    
    # Convert to frames for frame-level metrics
    pred_frames, gt_signal_frames, gt_noise_frames = events_to_frame_predictions(
        events, predictions, cfg.FPS, cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT
    )
    
    # Compute frame-level metrics
    frame_metrics = compute_frame_level_metrics(pred_frames, gt_signal_frames, gt_noise_frames)
    
    # Format hardware operations
    hw_summary = format_hw_ops_summary(hw_ops, len(events))
    
    # Combine all metrics
    result = {
        'filename': filename,
        'filter_name': filter_obj.name,
        'num_events': len(events),
        'num_frames': pred_frames.shape[0] if len(pred_frames) > 0 else 0,
        **stream_metrics,
        **frame_metrics,
        **hw_summary
    }
    
    return result

def evaluate_all_filters_on_dataset(test_folder, filter_names=None):
    """모든 필터를 데이터셋 전체 파일에 대해 평가"""
    if filter_names is None:
        filter_names = get_all_filter_names()
    
    file_pattern = os.path.join(test_folder, '*.npy')
    file_list = sorted(glob.glob(file_pattern))
    
    if not file_list:
        print(f"❌ No files found in {test_folder}")
        return pd.DataFrame()
    
    print(f"\n{'='*70}")
    print(f"Evaluating {len(filter_names)} filters on {len(file_list)} files")
    print(f"Test folder: {test_folder}")
    print(f"Frame size: {cfg.FRAME_WIDTH}x{cfg.FRAME_HEIGHT}, FPS: {cfg.FPS}")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for filter_name in filter_names:
        print(f"\n🔍 Evaluating filter: {filter_name}")
        filter_obj = create_filter(filter_name)
        
        for file_path in tqdm(file_list, desc=f"  Processing files", unit="file"):
            filename = os.path.basename(file_path)
            
            try:
                events = np.load(file_path)
                
                if not isinstance(events, np.ndarray) or events.ndim != 2 or events.shape[1] != 5:
                    print(f"    ⚠️  Skipping {filename}: Invalid format")
                    continue
                
                if events.shape[0] == 0:
                    print(f"    ⚠️  Skipping {filename}: Empty file")
                    continue
                
                result = evaluate_filter_on_file(filter_obj, events, filename)
                all_results.append(result)
                
            except Exception as e:
                print(f"    ❌ Error processing {filename}: {e}")
                continue
    
    return pd.DataFrame(all_results)

# 메인 평가 루프
all_dataset_results = {}

for dataset_name, dataset_path in datasets:
    print(f"\n\n{'#'*70}")
    print(f"# Processing Dataset: {dataset_name}")
    print(f"{'#'*70}\n")
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset path not found: {dataset_path}")
        continue
    
    df_results = evaluate_all_filters_on_dataset(dataset_path)
    
    if df_results.empty:
        print(f"⚠️  No results for {dataset_name}")
        continue
    
    # 필터별 집계
    aggregated = []
    for filter_name in df_results['filter_name'].unique():
        filter_df = df_results[df_results['filter_name'] == filter_name]
        total_events = filter_df['num_events'].sum()
        
        agg = {
            'filter_name': filter_name,
            'dataset': dataset_name,
            'num_files': len(filter_df),
            'total_events': int(total_events),
            'stream_da': (filter_df['stream_da'] * filter_df['num_events']).sum() / total_events,
            'stream_f1': (filter_df['stream_f1'] * filter_df['num_events']).sum() / total_events,
            'stream_precision': (filter_df['stream_precision'] * filter_df['num_events']).sum() / total_events,
            'stream_recall': (filter_df['stream_recall'] * filter_df['num_events']).sum() / total_events,
            'stream_snr_db': filter_df['stream_snr_db'].replace([np.inf, -np.inf], np.nan).mean(),
            'stream_auc': filter_df['stream_auc'].mean() if 'stream_auc' in filter_df else np.nan,
            'ops_per_event': filter_df['ops_per_event'].mean(),
        }
        aggregated.append(agg)
    
    df_aggregated = pd.DataFrame(aggregated)
    df_aggregated = df_aggregated.sort_values('stream_f1', ascending=False)
    
    # 결과 저장
    detailed_path = os.path.join(output_dir, f'{dataset_name}_detailed.csv')
    aggregated_path = os.path.join(output_dir, f'{dataset_name}_aggregated.csv')
    
    df_results.to_csv(detailed_path, index=False)
    df_aggregated.to_csv(aggregated_path, index=False)
    
    print(f"\n✅ Results saved to:")
    print(f"   - {detailed_path}")
    print(f"   - {aggregated_path}")
    
    # 요약 출력
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Filter':<15} {'F1':<10} {'DA':<10} {'Precision':<10} {'Recall':<10} {'SNR(dB)':<10}")
    print(f"{'-'*70}")
    for _, row in df_aggregated.iterrows():
        snr_str = f"{row['stream_snr_db']:.2f}" if pd.notna(row['stream_snr_db']) else "N/A"
        print(f"{row['filter_name']:<15} {row['stream_f1']:<10.4f} {row['stream_da']:<10.4f} "
              f"{row['stream_precision']:<10.4f} {row['stream_recall']:<10.4f} {snr_str:<10}")
    
    all_dataset_results[dataset_name] = df_aggregated

print(f"\n\n{'='*70}")
print("All evaluations completed!")
print(f"Results saved in: {output_dir}")
print(f"{'='*70}\n")
EOF

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Results saved in: ./results/hotelbar/"
