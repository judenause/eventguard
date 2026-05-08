"""
Main evaluation script for classical event noise filters.

This script evaluates all classical filters on specified test datasets
and generates comprehensive comparison reports including:
- Frame-level metrics
- Event-stream level metrics  
- Hardware operation counts
- Per-filter and aggregated results
"""

import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Dict, List
import json

from config import cfg
from filters import create_filter, get_all_filter_names
from utils import (
    compute_event_stream_metrics,
    compute_frame_level_metrics,
    events_to_frame_predictions,
    format_hw_ops_summary
)


def evaluate_filter_on_file(filter_obj, events: np.ndarray, 
                            filename: str) -> Dict:
    """
    Evaluate a single filter on a single file.
    
    Args:
        filter_obj: Filter instance
        events: (N, 5) array [label, x, y, t, polarity]
        filename: Name of the file being processed
    
    Returns:
        Dictionary containing all metrics for this file
    """
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


def evaluate_all_filters_on_dataset(test_folder: str, 
                                    filter_names: List[str] = None) -> pd.DataFrame:
    """
    Evaluate all filters on all files in a test dataset.
    
    Args:
        test_folder: Path to test data folder
        filter_names: List of filter names to evaluate (None = all filters)
    
    Returns:
        DataFrame containing all results
    """
    if filter_names is None:
        filter_names = get_all_filter_names()
    
    # Get list of files
    file_pattern = os.path.join(test_folder, cfg.DATA_FILE_PATTERN)
    file_list = sorted(glob.glob(file_pattern))
    
    if not file_list:
        print(f"❌ No files found in {test_folder}")
        return pd.DataFrame()
    
    print(f"\n{'='*70}")
    print(f"Evaluating {len(filter_names)} filters on {len(file_list)} files")
    print(f"Test folder: {test_folder}")
    print(f"Filters: {', '.join(filter_names)}")
    print(f"{'='*70}\n")
    
    all_results = []
    
    # Evaluate each filter
    for filter_name in filter_names:
        print(f"\n🔍 Evaluating filter: {filter_name}")
        print(f"{'-'*70}")
        
        filter_obj = create_filter(filter_name)
        
        # Process each file
        for file_path in tqdm(file_list, desc=f"  Processing files", unit="file"):
            filename = os.path.basename(file_path)
            
            try:
                # Load events
                events = np.load(file_path)
                
                # Validate data
                if not isinstance(events, np.ndarray) or events.ndim != 2 or events.shape[1] != 5:
                    print(f"    ⚠️  Skipping {filename}: Invalid format")
                    continue
                
                if events.shape[0] == 0:
                    print(f"    ⚠️  Skipping {filename}: Empty file")
                    continue
                
                # Check timestamp scale and normalize to seconds if needed
                # ESD test_50 is in seconds (e.g., 0.04 to 0.7)
                # driving and hotelbar are typically in microseconds (e.g., 2e6 or 1.6e9)
                t_max = events[:, 3].max()
                if t_max > 10000:  # Threshold to detect microseconds or similar large scales
                    # We assume it's in microseconds if it's very large
                    # This handles both 2e6 (relative) and 1.6e9 (absolute) scales
                    print(f"    ℹ️  Normalizing timestamps for {filename} (t_max={t_max:.2f})")
                    events[:, 3] = (events[:, 3] - events[:, 3].min()) / 1_000_000.0
                
                # Evaluate filter
                result = evaluate_filter_on_file(filter_obj, events, filename)
                all_results.append(result)
                
            except Exception as e:
                print(f"    ❌ Error processing {filename}: {e}")
                continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    return df_results


def compute_aggregated_metrics(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated metrics per filter across all files.
    
    Args:
        df_results: DataFrame with per-file results
    
    Returns:
        DataFrame with aggregated metrics per filter
    """
    if df_results.empty:
        return pd.DataFrame()
    
    # Group by filter
    aggregated = []
    
    for filter_name in df_results['filter_name'].unique():
        filter_df = df_results[df_results['filter_name'] == filter_name]
        
        # Compute weighted averages for metrics
        total_events = filter_df['num_events'].sum()
        total_frames = filter_df['num_frames'].sum()
        
        agg_metrics = {
            'filter_name': filter_name,
            'num_files': len(filter_df),
            'total_events': int(total_events),
            'total_frames': int(total_frames),
            
            # Event-stream metrics (weighted by num_events)
            'avg_stream_da': (filter_df['stream_da'] * filter_df['num_events']).sum() / total_events,
            'avg_stream_f1': (filter_df['stream_f1'] * filter_df['num_events']).sum() / total_events,
            'avg_stream_precision': (filter_df['stream_precision'] * filter_df['num_events']).sum() / total_events,
            'avg_stream_recall': (filter_df['stream_recall'] * filter_df['num_events']).sum() / total_events,
            
            # SNR metrics
            'avg_stream_snr_db': filter_df['stream_snr_db'].replace([np.inf, -np.inf], np.nan).mean(),
            'avg_stream_esnr_db': filter_df['stream_esnr_db'].replace([np.inf, -np.inf], np.nan).mean(),
            
            # Additional v8_bconvsnn metrics
            'avg_stream_nrr': (filter_df['stream_nrr'] * filter_df['num_events']).sum() / total_events,
            'avg_stream_sr': (filter_df['stream_sr'] * filter_df['num_events']).sum() / total_events,
            'avg_stream_nr': (filter_df['stream_nr'] * filter_df['num_events']).sum() / total_events,
            'avg_stream_edp': (filter_df['stream_edp'] * filter_df['num_events']).sum() / total_events,
            
            # Frame-level metrics (weighted by num_frames)
            'avg_frame_da': (filter_df['frame_da'] * filter_df['num_frames']).sum() / total_frames if total_frames > 0 else 0,
            'avg_frame_f1': (filter_df['frame_f1'] * filter_df['num_frames']).sum() / total_frames if total_frames > 0 else 0,
            'avg_frame_precision': (filter_df['frame_precision'] * filter_df['num_frames']).sum() / total_frames if total_frames > 0 else 0,
            'avg_frame_recall': (filter_df['frame_recall'] * filter_df['num_frames']).sum() / total_frames if total_frames > 0 else 0,
            
            # Frame SNR metrics
            'avg_frame_snr_db': filter_df['frame_snr_db'].replace([np.inf, -np.inf], np.nan).mean() if 'frame_snr_db' in filter_df else np.nan,
            'avg_frame_esnr_db': filter_df['frame_esnr_db'].replace([np.inf, -np.inf], np.nan).mean() if 'frame_esnr_db' in filter_df else np.nan,
            
            # Additional Frame metrics
            'avg_frame_nrr': (filter_df['frame_nrr'] * filter_df['num_frames']).sum() / total_frames if total_frames > 0 and 'frame_nrr' in filter_df else 0,
            'avg_frame_auc': (filter_df['frame_auc'] * filter_df['num_frames']).sum() / total_frames if total_frames > 0 and 'frame_auc' in filter_df else 0,
            
            # Hardware operations (average per event)
            'avg_ops_per_event': filter_df['ops_per_event'].mean(),
            'total_hw_ops': int(filter_df['total_hw_ops'].sum()),
            'avg_comparisons': filter_df['comparisons'].mean(),
            'avg_additions': filter_df['additions'].mean(),
            'avg_multiplications': filter_df['multiplications'].mean(),
            'avg_divisions': filter_df['divisions'].mean(),
            'avg_exp_ops': filter_df['exp_ops'].mean(),
            'avg_memory_accesses': filter_df['memory_accesses'].mean(),
        }
        
        aggregated.append(agg_metrics)
    
    df_aggregated = pd.DataFrame(aggregated)
    
    # Sort by stream F1 score (descending)
    df_aggregated = df_aggregated.sort_values('avg_stream_f1', ascending=False)
    
    return df_aggregated


def save_results(df_results: pd.DataFrame, df_aggregated: pd.DataFrame,
                output_dir: str, dataset_name: str):
    """
    Save results to CSV files.
    
    Args:
        df_results: Per-file results
        df_aggregated: Aggregated results per filter
        output_dir: Output directory
        dataset_name: Name of the dataset (e.g., 'test_50')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed per-file results
    detailed_path = os.path.join(output_dir, f'{dataset_name}_detailed_results.csv')
    df_results.to_csv(detailed_path, index=False)
    print(f"\n✅ Detailed results saved to: {detailed_path}")
    
    # Save aggregated results
    aggregated_path = os.path.join(output_dir, f'{dataset_name}_aggregated_results.csv')
    df_aggregated.to_csv(aggregated_path, index=False)
    print(f"✅ Aggregated results saved to: {aggregated_path}")
    
    # Save summary report (text file)
    summary_path = os.path.join(output_dir, f'{dataset_name}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Classical Event Noise Filters - Evaluation Summary\n")
        f.write(f"{'='*70}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of files: {df_aggregated['num_files'].iloc[0]}\n")
        f.write(f"Total events: {df_aggregated['total_events'].iloc[0]:,}\n\n")
        
        # Print filter configuration parameters
        f.write(f"Filter Configuration\n")
        f.write(f"{'-'*70}\n")
        for filter_name, filter_params in cfg.FILTER_CONFIGS.items():
            params_str = ', '.join(f"{k}={v}" for k, v in filter_params.items())
            f.write(f"  {filter_name}: {params_str}\n")
        f.write(f"\n")
        
        f.write(f"Filter Performance Ranking (by Stream F1 Score)\n")
        f.write(f"{'-'*70}\n")
        
        for idx, row in df_aggregated.iterrows():
            f.write(f"\n{idx+1}. {row['filter_name']}\n")
            f.write(f"   Stream Metrics:\n")
            f.write(f"     - DA (Denoising Accuracy): {row['avg_stream_da']:.4f}\n")
            f.write(f"     - F1 Score:                {row['avg_stream_f1']:.4f}\n")
            f.write(f"     - Precision:               {row['avg_stream_precision']:.4f}\n")
            f.write(f"     - Recall:                  {row['avg_stream_recall']:.4f}\n")
            f.write(f"     - SNR (dB):                {row['avg_stream_snr_db']:.2f}\n")
            f.write(f"     - ESNR (dB):               {row['avg_stream_esnr_db']:.2f}\n")
            if pd.notna(row.get('avg_stream_nrr')):
                f.write(f"     - NRR (Noise Rejection):   {row['avg_stream_nrr']:.4f}\n")
            
            f.write(f"   Frame Metrics:\n")
            f.write(f"     - DA:                      {row['avg_frame_da']:.4f}\n")
            f.write(f"     - F1 Score:                {row['avg_frame_f1']:.4f}\n")
            if pd.notna(row.get('avg_frame_snr_db')):
                f.write(f"     - SNR (dB):                {row['avg_frame_snr_db']:.2f}\n")
            if pd.notna(row.get('avg_frame_esnr_db')):
                f.write(f"     - ESNR (dB):               {row['avg_frame_esnr_db']:.2f}\n")
            if pd.notna(row.get('avg_frame_auc')):
                f.write(f"     - AUC:                     {row['avg_frame_auc']:.4f}\n")
            f.write(f"   Hardware Complexity:\n")
            f.write(f"     - Ops per event:           {row['avg_ops_per_event']:.2f}\n")
            f.write(f"     - Total operations:        {row['total_hw_ops']:,}\n")
    
    print(f"✅ Summary report saved to: {summary_path}")


def print_summary_table(df_aggregated: pd.DataFrame):
    """Print a formatted summary table to console."""
    print(f"\n{'='*70}")
    print(f"AGGREGATED RESULTS - FILTER COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Filter':<12} {'Stream F1':<10} {'Stream DA':<10} {'Stream SNR':<12} {'Frame F1':<10} {'Ops/Event':<12}")
    print(f"{'-'*75}")
    
    for _, row in df_aggregated.iterrows():
        print(f"{row['filter_name']:<12} "
              f"{row['avg_stream_f1']:<10.4f} "
              f"{row['avg_stream_da']:<10.4f} "
              f"{row['avg_stream_snr_db']:<12.2f} "
              f"{row['avg_frame_f1']:<10.4f} "
              f"{row['avg_ops_per_event']:<12.2f}")
    
    print(f"{'-'*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classical event noise filters on test datasets"
    )
    parser.add_argument(
        '--test_folder',
        type=str,
        default='test_50',
        choices=['test_50', 'test_100', 'both', 'custom'],
        help='Which test dataset to evaluate (default: test_50)'
    )
    parser.add_argument(
        '--custom_path',
        type=str,
        default=None,
        help='Path to custom test dataset (required if test_folder is custom)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='Override FPS setting'
    )
    parser.add_argument(
        '--filters',
        type=str,
        nargs='+',
        default=None,
        help='Specific filters to evaluate (default: all filters)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: ./results/)'
    )
    
    args = parser.parse_args()

    # Override FPS if provided
    if args.fps:
        cfg.FPS = args.fps
        print(f"ℹ️  Overriding FPS to {cfg.FPS}")
    
    # Determine output directory
    output_dir = args.output_dir if args.output_dir else cfg.RESULTS_DIR
    
    # Determine which datasets to evaluate
    if args.test_folder == 'both':
        datasets = [
            ('test_50', cfg.TEST_DATA_FOLDER_50),
            ('test_100', cfg.TEST_DATA_FOLDER_100)
        ]
    elif args.test_folder == 'test_50':
        datasets = [('test_50', cfg.TEST_DATA_FOLDER_50)]
    elif args.test_folder == 'test_100':
        datasets = [('test_100', cfg.TEST_DATA_FOLDER_100)]
    elif args.test_folder == 'custom':
        if not args.custom_path:
            parser.error("--custom_path is required when --test_folder is 'custom'")
        dataset_name = os.path.basename(os.path.normpath(args.custom_path))
        datasets = [(dataset_name, args.custom_path)]
    
    # Evaluate each dataset
    for dataset_name, dataset_path in datasets:
        print(f"\n\n{'#'*70}")
        print(f"# Processing Dataset: {dataset_name}")
        print(f"{'#'*70}\n")
        
        # Run evaluation
        df_results = evaluate_all_filters_on_dataset(dataset_path, args.filters)
        
        if df_results.empty:
            print(f"⚠️  No results for {dataset_name}, skipping...")
            continue
        
        # Compute aggregated metrics
        df_aggregated = compute_aggregated_metrics(df_results)
        
        # Print summary
        print_summary_table(df_aggregated)
        
        # Save results
        save_results(df_results, df_aggregated, output_dir, dataset_name)
    
    print(f"\n{'='*70}")
    print(f"✅ All evaluations completed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
