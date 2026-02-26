import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
from filters import BAF, NNFilter
from utils import compute_event_stream_metrics
from config import cfg

def tune_baf(file_path, windows=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]):
    print(f"\n🔍 Tuning BAF on {os.path.basename(file_path)}")
    events = np.load(file_path)
    
    results = []
    for w in windows:
        filter_obj = BAF(time_window=w)
        preds, _ = filter_obj.filter_events(events)
        metrics = compute_event_stream_metrics(events, preds)
        metrics['time_window'] = w
        results.append(metrics)
        print(f"  Window={w*1000:.1f}ms -> SNR={metrics['stream_snr_db']:.2f} dB, F1={metrics['stream_f1']:.4f}, Recall={metrics['stream_recall']:.4f}")
    
    return pd.DataFrame(results)

def tune_nn(file_path, neighbors=[1, 2, 3, 4, 5], radius=1, window=0.01):
    print(f"\n🔍 Tuning NN (r={radius}, w={window*1000:.0f}ms) on {os.path.basename(file_path)}")
    events = np.load(file_path)
    
    results = []
    for n in neighbors:
        filter_obj = NNFilter(spatial_radius=radius, temporal_window=window, min_neighbors=n)
        preds, _ = filter_obj.filter_events(events)
        metrics = compute_event_stream_metrics(events, preds)
        metrics['min_neighbors'] = n
        results.append(metrics)
        print(f"  Neighbors={n} -> SNR={metrics['stream_snr_db']:.2f} dB, F1={metrics['stream_f1']:.4f}, Recall={metrics['stream_recall']:.4f}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Use the first file in test_50 for tuning
    files = sorted(glob.glob(os.path.join(cfg.TEST_DATA_FOLDER_50, cfg.DATA_FILE_PATTERN)))
    if not files:
        print("No files found.")
        exit()
        
    test_file = files[0]
    
    print("==================================================")
    print(" Parameter Tuning to Match Literature Results")
    print(" Target BAF SNR: ~23.54 dB")
    print(" Target NN SNR:  ~23.80 dB")
    print("==================================================")
    
    tune_baf(test_file)
    tune_nn(test_file)
