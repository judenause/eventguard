import pandas as pd
import os
import argparse
from evaluate_filters import compute_aggregated_metrics, save_results

def regenerate_summary(dataset_name, results_dir):
    print(f"Regenerating summary for {dataset_name}...")
    
    detailed_path = os.path.join(results_dir, f'{dataset_name}_detailed_results.csv')
    if not os.path.exists(detailed_path):
        print(f"❌ File not found: {detailed_path}")
        return

    # Load detailed results
    df_results = pd.read_csv(detailed_path)
    
    # Check if frame SNR columns exist, if not compute them
    if 'frame_snr_db' not in df_results.columns and 'frame_tp' in df_results.columns:
        print("Computing missing frame SNR metrics...")
        import math
        import numpy as np
        epsilon = 1e-10
        
        def calc_snr(row):
            t, f = row['frame_tp'], row['frame_fp']
            if f + epsilon == 0: return float('inf') if t > 0 else 0.0
            elif t + epsilon == 0: return float('-inf')
            try:
                return 10 * math.log10((t + epsilon) / (f + epsilon))
            except:
                return 0.0
            
        def calc_esnr(row):
            t, f = row['frame_tp'], row['frame_fp']
            if f + epsilon == 0: return float('inf') if t > 0 else 0.0
            elif t + epsilon == 0: return float('-inf')
            try:
                return 20 * math.log10((t + epsilon) / (f + epsilon))
            except:
                return 0.0

        df_results['frame_snr_db'] = df_results.apply(calc_snr, axis=1)
        df_results['frame_esnr_db'] = df_results.apply(calc_esnr, axis=1)
        
        # Compute other missing metrics (NRR, SR, NR, EDP)
        # Stream metrics
        df_results['stream_nrr'] = df_results['stream_tn'] / (df_results['stream_tn'] + df_results['stream_fp'] + epsilon)
        df_results['stream_sr'] = df_results['stream_recall']
        df_results['stream_nr'] = df_results['stream_nrr']
        df_results['stream_edp'] = df_results['stream_precision']
        
        # Frame metrics
        df_results['frame_nrr'] = df_results['frame_tn'] / (df_results['frame_tn'] + df_results['frame_fp'] + epsilon)
        df_results['frame_sr'] = df_results['frame_recall']
        df_results['frame_nr'] = df_results['frame_nrr']
        df_results['frame_edp'] = df_results['frame_precision']
        
        # Frame AUC (approximate from binary predictions as balanced accuracy)
        # AUC = 0.5 * (TPR + TNR) for binary classifier
        tpr = df_results['frame_recall']
        tnr = df_results['frame_nrr']
        df_results['frame_auc'] = 0.5 * (tpr + tnr)
    
    # Compute aggregated metrics (using updated function with SNR)
    df_aggregated = compute_aggregated_metrics(df_results)
    
    # Save results (overwriting summary.txt with new format)
    save_results(df_results, df_aggregated, results_dir, dataset_name)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()
    
    regenerate_summary('test_50', args.results_dir)
    # Check if test_100 exists
    if os.path.exists(os.path.join(args.results_dir, 'test_100_detailed_results.csv')):
        regenerate_summary('test_100', args.results_dir)
