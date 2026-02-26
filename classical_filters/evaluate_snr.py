#!/usr/bin/env python3
import numpy as np
import os
import argparse
import glob
from tqdm import tqdm

# --- Filter Implementations ---

def apply_baf_filter(events, tau, width=1280, height=720):
    """BAF (Bilateral Accumulation Filter) using 2D map"""
    predictions = np.zeros(len(events), dtype=np.int32)
    last_pixel_timestamp = {} 
    
    # 3x3 neighbor check requires more complex map than dict usually, 
    # but let's use the optimized logic from evaluate_roc_auc.py (if available)
    # Re-implementing based on Java BAF (3x3 neighbor check)
    
    # Using dense array for speed if possible, but 1280x720 is small enough
    ts_map = np.full((width, height), -1e9, dtype=np.float64) 
    
    # However, Python loops are slow. We assume this script runs offline.
    # To match 'evaluate_roc_auc.py' exactly, we should copy its logic.
    # Since I cannot import 'evaluate_roc_auc' easily if it has global code, 
    # I'll implement a simplified version or the correct one.
    
    # Correct BAF Logic (Java): Check 3x3 neighbors. If any has t within tau, pass.
    # Update SELF pixel timestamp.
    
    for i, event in enumerate(events):
        x, y, t = int(event[1]), int(event[2]), event[3]
        if not (0 <= x < width and 0 <= y < height): continue
        
        # Check neighbors
        found_correlation = False
        x_min, x_max = max(0, x-1), min(width-1, x+1)
        y_min, y_max = max(0, y-1), min(height-1, y+1)
        
        # Optimization: check 3x3 region in ts_map
        # (This logic is slow in pure Python, but correct)
        region = ts_map[x_min:x_max+1, y_min:y_max+1]
        if np.any(t - region <= tau):
             predictions[i] = 1 # Signal
        
        ts_map[x, y] = t # Update self
        
    return predictions


def apply_stcf_filter(events, tau, k=4, width=1280, height=720):
    """STCF Filter"""
    predictions = np.zeros(len(events), dtype=np.int32)
    
    # Store timestamp and polarity
    ts_map = np.full((width, height), -1e9, dtype=np.float64)
    pol_map = np.full((width, height), -1, dtype=np.int32)
    
    for i, event in enumerate(events):
        x, y, t, p = int(event[1]), int(event[2]), event[3], int(event[4])
        if not (0 <= x < width and 0 <= y < height): continue
        
        x_min, x_max = max(0, x-1), min(width-1, x+1)
        y_min, y_max = max(0, y-1), min(height-1, y+1)
        
        count = 0
        for nx in range(x_min, x_max+1):
            for ny in range(y_min, y_max+1):
                if nx == x and ny == y: continue # Exclude self
                
                dt = t - ts_map[nx, ny]
                if dt <= tau:
                    if pol_map[nx, ny] == p: # Polarity match
                        count += 1
        
        if count >= k:
            predictions[i] = 1
            
        ts_map[x, y] = t
        pol_map[x, y] = p
        
    return predictions

def apply_onf_filter(events, tau, width=1280, height=720):
    """ONF Filter (Corrected O(N) Implementation)"""
    predictions = np.zeros(len(events), dtype=np.int32)
    
    row_ts = np.full(height, -np.inf)
    col_ts = np.full(width, -np.inf)
    
    for i, event in enumerate(events):
        x, y, t = int(event[1]), int(event[2]), event[3]
        if not (0 <= x < width and 0 <= y < height): continue
        
        # Check Row OR Column connectivity
        if (t - row_ts[y] <= tau) or (t - col_ts[x] <= tau):
            predictions[i] = 1
            
        row_ts[y] = t
        col_ts[x] = t
        
    return predictions

def apply_stcf_sub_filter(events, tau, block_size=2, k=1, width=1280, height=720):
    """STCF Subsampled"""
    predictions = np.zeros(len(events), dtype=np.int32)
    
    w_sub = width // block_size
    h_sub = height // block_size
    
    ts_map = np.full((w_sub, h_sub), -1e9, dtype=np.float64)
    
    for i, event in enumerate(events):
        x, y, t = int(event[1]), int(event[2]), event[3]
        if not (0 <= x < width and 0 <= y < height): continue
        
        sx, sy = x // block_size, y // block_size
        
        # Check neighbors in subsampled space
        sx_min, sx_max = max(0, sx-1), min(w_sub-1, sx+1)
        sy_min, sy_max = max(0, sy-1), min(h_sub-1, sy+1)
        
        count = 0
        # Check 3x3 in subsampled grid
        region = ts_map[sx_min:sx_max+1, sy_min:sy_max+1]
        
        # Exclude self? Java STCF_Sub applies STCF on subsampled stream.
        # Usually it excludes self.
        # Simple check: count valid neighbors
        count = np.sum((t - region) <= tau)
        
        # If region includes self, we should correct. 
        # But 'region' contains the timestamp of the LAST event in that block.
        # If the last event was indeed 'self' (same block), it would be < tau (dt=0).
        # But we haven't updated ts_map yet. So region contains OLD timestamps. 
        # So we don't need to explicitly exclude self unless we update before check.
        
        if count >= k:
             predictions[i] = 1
             
        ts_map[sx, sy] = t
        
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Calculate SNR for Classical Filters")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--fps', type=int, required=True, help='FPS of the dataset (to determine tau)')
    parser.add_argument('--filters', nargs='+', default=['BAF', 'STCF', 'ONF', 'STCF_Sub'])
    
    args = parser.parse_args()
    
    tau = 1.0 / args.fps
    print(f"Applying Filters with FPS={args.fps} -> tau={tau:.4f}s")
    
    filepaths = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))
    if not filepaths:
        print(f"No .npy files found in {args.data_dir}")
        return
        
    # Stats: [Signal_Pass, Noise_Pass]
    # GT: 0=Signal, 1=Noise (Raw Data)
    stats = {f: {'sig_pass': 0, 'noise_pass': 0, 'total_sig': 0, 'total_noise': 0} for f in args.filters}
    
    for fp in tqdm(filepaths, desc="Processing files"):
        try:
            events = np.load(fp)
            if events.ndim != 2 or events.shape[1] < 4: continue
            
            # Normalize timestamp if needed (microseconds -> seconds)
            if events[:, 3].max() > 10000:
                events[:, 3] = (events[:, 3] - events[:, 3].min()) / 1_000_000.0
                
            # GT Labels: 0=Signal, 1=Noise
            gt_labels = events[:, 0].astype(np.int32)
            
            stats[args.filters[0]]['total_sig'] += np.sum(gt_labels == 0)
            stats[args.filters[0]]['total_noise'] += np.sum(gt_labels == 1)
            
            for filter_name in args.filters:
                if filter_name == 'BAF':
                    preds = apply_baf_filter(events, tau)
                elif filter_name == 'STCF':
                    preds = apply_stcf_filter(events, tau)
                elif filter_name == 'ONF':
                    preds = apply_onf_filter(events, tau)
                elif filter_name == 'STCF_Sub':
                    preds = apply_stcf_sub_filter(events, tau)
                else:
                    continue
                
                # preds=1 (Signal), preds=0 (Noise)
                # Signal Pass: pred=1 & gt=0 (GT Signal) -> TP
                # Noise Pass: pred=1 & gt=1 (GT Noise) -> FP
                
                sig_pass = np.sum((preds == 1) & (gt_labels == 0))
                noise_pass = np.sum((preds == 1) & (gt_labels == 1))
                
                stats[filter_name]['sig_pass'] += sig_pass
                stats[filter_name]['noise_pass'] += noise_pass
                
                # Other filters share total counts
                stats[filter_name]['total_sig'] = stats[args.filters[0]]['total_sig']
                stats[filter_name]['total_noise'] = stats[args.filters[0]]['total_noise']

        except Exception as e:
            print(f"Error {fp}: {e}")
            
    print("\n=== SNR Results ===")
    print(f"Dataset: {args.data_dir}")
    print(f"FPS: {args.fps} (tau={tau:.4f}s)")
    print(f"{'Filter':<10} {'SNR (dB)':<10} {'SNR (Ratio)':<15} {'Sig Pass %':<12} {'Noise Pass % (FPR)':<15}")
    print("-" * 70)
    
    for f in args.filters:
        s = stats[f]
        sig_pass = s['sig_pass']
        noise_pass = s['noise_pass']
        
        # SNR Calculation
        # SNR = 20 * log10(Signal_Power / Noise_Power)
        # Here power ~ event count? Or just ratio of counts?
        # Usually SNR in DVS filter context: 20 log10 (Number of Signal Events / Number of Noise Events) in the output.
        
        if noise_pass > 0:
            ratio = sig_pass / noise_pass
            snr_db = 20 * np.log10(ratio)
        else:
            ratio = float('inf')
            snr_db = float('inf')
            
        sig_pass_rate = (sig_pass / s['total_sig'] * 100) if s['total_sig'] > 0 else 0
        noise_pass_rate = (noise_pass / s['total_noise'] * 100) if s['total_noise'] > 0 else 0 # This is FPR
        
        print(f"{f:<10} {snr_db:<10.2f} {ratio:<15.2f} {sig_pass_rate:<11.1f}% {noise_pass_rate:<14.2f}%")

if __name__ == "__main__":
    main()
