import torch
import torch.nn as nn
import os
import time
import argparse
import pandas as pd
from collections import OrderedDict
import numpy as np


# --- Project module imports ---
from config import cfg
from utils import save_metrics_to_csv, setup_device_and_batch_size, create_save_directories, create_evaluation_gif, visualize_test_set_results
from data_processing import process_folder_to_frame_lists
from model import Hybrid_SNN_Pure_BNN
from evaluation_engine import evaluate_model_on_dataset

# def save_metrics_to_csv(filename, results_dir, aggregated_metrics, per_file_metrics_list, summary_title):
#     """Utility function to save summary and detailed metrics to a CSV file"""
#     valid_metrics_list = [m for m in per_file_metrics_list if m and 'error' not in m]
#     if not valid_metrics_list:
#         print(f"No valid per-file metrics to save for {filename}.")
#         return

#     csv_path = os.path.join(results_dir, filename)
#     summary_lines = [f"--- {summary_title} ---"]
#     for key, value in aggregated_metrics.items():
#         val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
#         summary_lines.append(f'"{key.replace("_", " ").title()}","{val_str}"')
    
#     summary_lines.append("\n--- Detailed Per-File Metrics ---")
#     summary_header = "\n".join(summary_lines) + "\n"

#     try:
#         df = pd.DataFrame(valid_metrics_list)
#         with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
#             f.write(summary_header)
#             df.to_csv(f, index=False)
#         print(f"✅ Metrics report saved to: {csv_path}")
#     except Exception as e:
#         print(f"❌ Error saving metrics to CSV: {e}")

def load_model_for_evaluation(model_path: str, config: cfg, device: torch.device) -> nn.Module:
    """
    Final function to load the trained Hybrid_SNN_Pure_BNN model for evaluation.
    """
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Evaluation model not found at {model_path}")
        return None
    
    print(f"Loading trained model for evaluation: {model_path}")
    snn_params = {'beta': config.SNN_BETA, 'threshold': config.SNN_THRESHOLD}
    
    # Create model architecture using the correct model class.
    model = Hybrid_SNN_Pure_BNN(
        snn_params=snn_params,
        output_classes=config.OUTPUT_CLASSES,
        input_channels=config.INPUT_CHANNELS
    ).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Handle 'module.' prefix for models trained with DataParallel
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
            
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully for evaluation.")
        return model
    except Exception as e:
        print(f"❌ ERROR: Failed to load model state_dict. Exception: {e}")
        return None

def main_evaluation_pipeline(args):
    """
    Full pipeline for independent evaluation.
    """
    start_time = time.time()
    
    # --- 0. Update config with command line arguments ---
    for key, value in vars(args).items():
        config_key = key.upper()
        if value is not None and hasattr(cfg, config_key):
            setattr(cfg, config_key, value)
            print(f"Config override from command line: {config_key} = {value}")

    # --- 1. Initial settings ---
    print("\n--- 1. Initializing for Evaluation ---")
    setup_device_and_batch_size(cfg)
    create_save_directories(cfg)

    # --- 2. Load evaluation data ---
    print("\n--- 2. Loading Test Data ---")
    test_data_list = process_folder_to_frame_lists(cfg.TEST_DATA_FOLDER, cfg.DATA_FILE_PATTERN, "Test (Final Eval)", cfg)
    if not test_data_list:
        print("CRITICAL: No test data found. Aborting.")
        return

    # --- 3. Load model ---
    print("\n--- 3. Loading Model ---")
    model_path = os.path.join(cfg.SAVE_DIR, args.model_file)
    model = load_model_for_evaluation(model_path, cfg, cfg.DEVICE)
    if model is None: return
    model.eval()

    # --- 4. Run evaluation ---
    print("\n--- 4. Running Evaluation on Test Dataset ---")
    
    # (aggregated_frame_metrics, per_file_frame_metrics, 
    #  preds_seqs, _, _, 
    #  per_file_stream_metrics, aggregated_stream_metrics) = evaluate_model_on_dataset(model, test_data_list, cfg, cfg.DEVICE)

    # <<< [Change] Add eval_masks_seqs to the return value list
    (aggregated_frame_metrics, per_file_frame_metrics, 
     preds_seqs, probs_seqs, targets_seqs, eval_masks_seqs,
     per_file_stream_metrics, aggregated_stream_metrics) = evaluate_model_on_dataset(model, test_data_list, cfg, cfg.DEVICE)

    # ✅ Save results to CSV
    print("\n" + "="*50 + "\n=== FINAL EVALUATION RESULTS ===\n" + "="*50)
    
    if aggregated_frame_metrics:
        print("\n### Aggregated Frame-Level Metrics ###")
        for key, value in aggregated_frame_metrics.items():
            print(f"  {key:<30}: {value:.4f}" if isinstance(value, float) else f"  {key:<30}: {value}")
        save_metrics_to_csv(
            filename=f"{cfg.CSV_NAME}_frame_metrics.csv",
            results_dir=cfg.SAVE_DIR,
            aggregated_metrics=aggregated_frame_metrics,
            per_file_metrics_list=per_file_frame_metrics,
            summary_title="Overall Aggregated Frame-Level Metrics"
        )

    if aggregated_stream_metrics:
        print("\n### Aggregated Event-Stream Level Metrics ###")
        for key, value in aggregated_stream_metrics.items():
            print(f"  {key:<30}: {value:.4f}" if isinstance(value, float) else f"  {key:<30}: {value}")
        save_metrics_to_csv(
            filename=f"{cfg.CSV_NAME}_stream_metrics.csv",
            results_dir=cfg.SAVE_DIR,
            aggregated_metrics=aggregated_stream_metrics,
            per_file_metrics_list=per_file_stream_metrics,
            summary_title="Overall Aggregated Event-Stream Level Metrics"
        )
    
    # # --- 5. Save visualization and text summary ---
    # print("\n--- 5. Visualizing Final Results and Saving Summary ---")
    # if aggregated_frame_metrics:
    #     # <<< [Fully Fixed] Filter visualization data using evaluation masks
    #     masked_probs_flat = []
    #     masked_targets_flat = []
    #     for i in range(len(probs_seqs)):
    #         active_indices = eval_masks_seqs[i] > 0
    #         masked_probs_flat.append(probs_seqs[i][active_indices])
    #         masked_targets_flat.append(targets_seqs[i][active_indices])

    #     all_probs_masked = np.concatenate(masked_probs_flat)
    #     all_targets_masked = np.concatenate(masked_targets_flat)
        
    #     visualize_test_set_results(
    #         aggregated_final_metrics=aggregated_frame_metrics,
    #         aggregated_stream_metrics=aggregated_stream_metrics,
    #         config_obj=cfg,
    #         all_pred_probs_numpy=all_probs_masked,
    #         all_targets_numpy=all_targets_masked
    #     )

    # --- 6. Visualization ---
    if args.create_eval_gif and preds_seqs:
        print("\n--- 6. Generating Visualization GIF ---")
        
        # Lazy Loading support: Load frames if necessary
        first_data = test_data_list[0]
        if 'input_frames' not in first_data and 'processed_path' in first_data:
            stacked_data = np.load(first_data['processed_path'], mmap_mode='r')
            input_frames = stacked_data[:, 0]
            real_gt = stacked_data[:, 1]
            noise_gt = stacked_data[:, 2]
            eval_mask = stacked_data[:, 3]
        else:
            input_frames = first_data.get('input_frames')
            real_gt = first_data.get('real_event_gt')
            noise_gt = first_data.get('noise_event_gt')
            eval_mask = first_data.get('evaluation_mask')

        if input_frames is not None:
            create_evaluation_gif(
                input_frames_seq=input_frames,
                real_gt_frames_seq=real_gt,
                noise_gt_frames_seq=noise_gt,
                predicted_frames_seq=preds_seqs[0],
                eval_mask_frames_seq=eval_mask,
                config_obj=cfg,
                output_gif_filename_base=f"eval_{os.path.splitext(args.model_file)[0]}"
            )
        else:
            print("  - Warning: Could not load frames for visualization.")

    print(f"\n--- Evaluation Finished. Total time: {(time.time() - start_time)/60:.2f} minutes ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Hybrid SNN-BNN model.")
    
    # --- Required arguments ---
    parser.add_argument('--model_file', type=str, required=True, help='Filename of the model to evaluate (e.g., best_model.pth).')
    
    # --- Path and Environment settings ---
    parser.add_argument('--save_dir', type=str, default=None, help=f'Directory where the model is saved (default from config: {cfg.SAVE_DIR}).')
    parser.add_argument('--test_data_folder', type=str, default=None, help=f'Folder containing test data (default from config: {cfg.TEST_DATA_FOLDER}).')
    parser.add_argument('--specific_gpu_id', type=int, default=None, help=f'Specify GPU ID to use (default: {cfg.SPECIFIC_GPU_ID}).')
    
    # --- Data processing parameters ---
    parser.add_argument('--fps', type=int, default=None, help=f'FPS for data processing (default: {cfg.FPS}).')
    parser.add_argument('--data_file_pattern', type=str, default=None, help=f'Pattern for data files (default: {cfg.DATA_FILE_PATTERN}).')
    
    # --- Evaluation and Visualization parameters ---
    parser.add_argument('--evaluation_threshold', type=float, default=None, help=f'Threshold for binary prediction (default: {cfg.EVALUATION_THRESHOLD}).')
    parser.add_argument('--create_eval_gif', action='store_true', help='Flag to create a GIF of the first test file result.')
    parser.add_argument('--csv_name', type=str, default=None, help=f'Base name for result CSV files (default: {cfg.CSV_NAME}).')

    # --- SNN parameters for model reconstruction ---
    parser.add_argument('--snn_beta', type=float, default=None, help=f'SNN beta parameter used during training (default: {cfg.SNN_BETA}).')
    parser.add_argument('--snn_threshold', type=float, default=None, help=f'SNN threshold used during training (default: {cfg.SNN_THRESHOLD}).')
    
    args = parser.parse_args()
    
    main_evaluation_pipeline(args)
