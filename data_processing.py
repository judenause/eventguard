# data_processing.py
import numpy as np
import os
import glob
from tqdm import tqdm # Progress bar for file processing
import math
# --- 1. Modified events_to_frames function ---
# Now returns min_timestamp as well.
def events_to_frames(events: np.ndarray,
                     fps: int,
                     frame_width: int,
                     frame_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: # Added float to return type
    """
    Converts raw event data to various frame representations.
    """
    if not isinstance(events, np.ndarray) or events.ndim != 2 or events.shape[1] != 5 or len(events) == 0:
        empty_frames = np.empty((0, frame_height, frame_width), dtype=np.float32)
        # Return 0.0 for min_timestamp
        return empty_frames, empty_frames, empty_frames, empty_frames, 0.0

    # --- 1. Set time reference point and calculate frame indices (vectorized method) ---
    events = events[np.argsort(events[:, 3])]
    min_timestamp = events[0, 3] # This value needs to be returned.
    time_window_duration = 1.0 / fps

    # ... (Rest of the function logic) ...
    frame_indices = np.floor((events[:, 3] - min_timestamp) / time_window_duration).astype(int)
    num_frames = frame_indices.max() + 1
    
    all_input_frames = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
    all_real_event_gt = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
    all_noise_event_gt = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
    
    x_coords = events[:, 1].astype(int)
    y_coords = events[:, 2].astype(int)
    valid_mask = (x_coords >= 0) & (x_coords < frame_width) & \
                 (y_coords >= 0) & (y_coords < frame_height)

    valid_indices = frame_indices[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_x = x_coords[valid_mask]
    valid_labels = events[valid_mask, 0]

    all_input_frames[valid_indices, valid_y, valid_x] = 1.0
    
    real_mask = valid_labels == 0
    all_real_event_gt[valid_indices[real_mask], valid_y[real_mask], valid_x[real_mask]] = 1.0
    
    noise_mask = valid_labels != 0
    all_noise_event_gt[valid_indices[noise_mask], valid_y[noise_mask], valid_x[noise_mask]] = 1.0

    all_evaluation_mask = all_input_frames

    # Add min_timestamp to return values
    return all_input_frames, all_real_event_gt, all_noise_event_gt, all_evaluation_mask, min_timestamp
# def events_to_frames(events: np.ndarray,
#                      fps: int,
#                      frame_width: int,
#                      frame_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Converts raw event data to various frame representations. (Modified version)
#     """
#     if not isinstance(events, np.ndarray) or events.ndim != 2 or events.shape[1] != 5 or len(events) == 0:
#         empty_frames = np.empty((0, frame_height, frame_width), dtype=np.float32)
#         return empty_frames, empty_frames, empty_frames, empty_frames

#     # --- 1. Set time reference point and calculate frame indices (vectorized method) ---
#     events = events[np.argsort(events[:, 3])]
#     min_timestamp = events[0, 3]
#     time_window_duration = 1.0 / fps

#     # Calculate frame indices for all events at once
#     frame_indices = np.floor((events[:, 3] - min_timestamp) / time_window_duration).astype(int)
    
#     # Determine total number of frames needed
#     num_frames = frame_indices.max() + 1
    
#     # --- 2. Initialize frame arrays ---
#     all_input_frames = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
#     all_real_event_gt = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
#     all_noise_event_gt = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
    
#     # --- 3. Fill frames with events (advanced indexing) ---
#     # Filter events with valid coordinates only
#     x_coords = events[:, 1].astype(int)
#     y_coords = events[:, 2].astype(int)
#     valid_mask = (x_coords >= 0) & (x_coords < frame_width) & \
#                  (y_coords >= 0) & (y_coords < frame_height)

#     valid_indices = frame_indices[valid_mask]
#     valid_y = y_coords[valid_mask]
#     valid_x = x_coords[valid_mask]
#     valid_labels = events[valid_mask, 0]

#     # Set all event positions to 1 at once
#     all_input_frames[valid_indices, valid_y, valid_x] = 1.0

#     # Set real event (label=0) positions to 1
#     real_mask = valid_labels == 0
#     all_real_event_gt[valid_indices[real_mask], valid_y[real_mask], valid_x[real_mask]] = 1.0
    
#     # Set noise event (label>0) positions to 1
#     noise_mask = valid_labels != 0
#     all_noise_event_gt[valid_indices[noise_mask], valid_y[noise_mask], valid_x[noise_mask]] = 1.0

#     # evaluation_mask is same as input_frames
#     all_evaluation_mask = all_input_frames

#     return all_input_frames, all_real_event_gt, all_noise_event_gt, all_evaluation_mask



def process_folder_to_frame_lists(folder_path: str,
                                  file_pattern: str,
                                  dataset_name: str,
                                  config_obj) -> list[dict]:
    print(f"\nProcessing {dataset_name} data from folder: {folder_path}")
    file_path_pattern = os.path.join(folder_path, file_pattern)
    file_list = sorted(glob.glob(file_path_pattern))
    processed_files_data_list = []

    if not file_list:
        print(f"  - Warning: No files found matching pattern '{file_pattern}' in '{folder_path}'")
        return processed_files_data_list

    print(f"  - Found {len(file_list)} files to process for {dataset_name} set.")
    
    for i, npy_file_path in enumerate(tqdm(file_list, desc=f"  Processing {dataset_name} files", unit="file",dynamic_ncols=True)):
        base_filename = os.path.basename(npy_file_path)
        try:
            # --- Lazy Loading Optimization: Check if Cache Exists FIRST ---
            save_dir = getattr(config_obj, 'PROCESSED_DATA_SAVE_DIR', './processed_cache')
            split_name = dataset_name # Train, Validation, Test
            final_save_dir = os.path.join(save_dir, split_name)
            
            base_name_no_ext = os.path.splitext(base_filename)[0]
            saved_path = os.path.join(final_save_dir, f"{base_name_no_ext}_stacked.npy")

            if os.path.exists(saved_path):
                # print(f"    - Cache found for {base_filename}. Skipping processing.")
                try:
                    # Load Metadata from Cache
                    mmap_data = np.load(saved_path, mmap_mode='r')
                    num_frames = mmap_data.shape[0]
                    
                    # Estimate min_timestamp quickly (without full sort if possible, or just load first event)
                    # To be 100% accurate identical to events_to_frames, we need min(t).
                    # We can load raw events just for this, or partial load.
                    # Since loading raw_events IS fast compared to frame gen, let's load it.
                    raw_events = np.load(npy_file_path)
                    if raw_events.shape[0] > 0:
                        min_ts = np.min(raw_events[:, 3])
                    else:
                        min_ts = 0.0
                    
                    processed_files_data_list.append({
                        'file_path': npy_file_path,
                        'processed_path': saved_path,
                        'num_frames': num_frames,
                        'min_timestamp': min_ts
                    })
                    continue # SKIP heavy processing
                except Exception as e:
                    print(f"    - Error reading cache {saved_path}: {e}. Re-processing.")
                    # Fallback to normal processing if cache read fails

            # --- Normal Processing (If Cache Missing) ---
            raw_events = np.load(npy_file_path) # Original event stream (including labels)

            if not isinstance(raw_events, np.ndarray) or raw_events.ndim != 2 or raw_events.shape[1] != 5:
                print(f"    - Warning for {base_filename}: Invalid data format. Skipping file.")
                continue
            if raw_events.shape[0] == 0:
                print(f"    - Info for {base_filename}: File is empty. Skipping file.")
                continue

            input_f, real_gt_f, noise_gt_f, eval_mask_f, min_ts = events_to_frames(
                raw_events, # raw_events is the original stream with signal/noise labels
                fps=config_obj.FPS,
                frame_width=config_obj.FRAME_WIDTH,
                frame_height=config_obj.FRAME_HEIGHT
            )

            if input_f.shape[0] >= config_obj.WINDOW_SIZE: # WINDOW_SIZE is the sequence length for train/inference
                # --- Lazy Loading: Save to Disk (Stacked for mmap) ---
                # save_dir and final_save_dir are already defined above.
                os.makedirs(final_save_dir, exist_ok=True)
                
                # saved_path is already defined above.
                
                # Stack: [T, 4, H, W] -> 0:Input, 1:RealGT, 2:NoiseGT, 3:EvalMask
                stacked_frames = np.stack([input_f, real_gt_f, noise_gt_f, eval_mask_f], axis=1)
                
                # Save Uncompressed .npy for mmap
                np.save(saved_path, stacked_frames)
                                    
                # Store Metadata ONLY
                processed_files_data_list.append({
                    'file_path': npy_file_path,               # Original Event File
                    'processed_path': saved_path,             # Cached Stacked Frame File
                    'num_frames': input_f.shape[0],           # Metadata
                    'min_timestamp' : min_ts,                 # Metadata
                    # 'original_labeled_event_stream': raw_events # Removed to save RAM. Load if needed.
                })
            else:
                # Minimize logging when using tqdm
                pass 

        except Exception as e:
            print(f"    - Error processing file {base_filename}: {e}. Skipping.")

    print(f"--- Finished processing {dataset_name} folder. {len(processed_files_data_list)} files successfully processed and saved to {final_save_dir}. ---")
    return processed_files_data_list


# (Optional) Sliding window generation function
# Note: This is not strictly required for EventFrameLazyDataset as it handles slicing internally,
# but can be kept for reference or used with EventFrameWindowDataset.
def create_all_sliding_windows(input_frames_seq: np.ndarray,
                               real_gt_frames_seq: np.ndarray,
                               noise_gt_frames_seq: np.ndarray,
                               eval_mask_frames_seq: np.ndarray,
                               window_size: int,
                               stride: int) -> tuple:
    """
    Create sliding window samples from time-series frame data for a SINGLE file's sequence.
    Note: This is memory-intensive if applied to all files and concatenated.
          EventFrameLazyDataset avoids this by creating windows on-the-fly.
    """
    num_total_frames = input_frames_seq.shape[0]
    frame_h = input_frames_seq.shape[1]
    frame_w = input_frames_seq.shape[2]

    if num_total_frames < window_size:
        # print(f"Warning: num_frames ({num_total_frames}) < window_size ({window_size}). Cannot create windows.")
        # Return empty arrays with correct dimensions (except N_win=0)
        return (np.empty((0, window_size, frame_h, frame_w), dtype=input_frames_seq.dtype),
                np.empty((0, window_size, frame_h, frame_w), dtype=real_gt_frames_seq.dtype),
                np.empty((0, window_size, frame_h, frame_w), dtype=noise_gt_frames_seq.dtype),
                np.empty((0, window_size, frame_h, frame_w), dtype=eval_mask_frames_seq.dtype))

    num_windows = (num_total_frames - window_size) // stride + 1

    # Efficient window generation using np.lib.stride_tricks.as_strided
    # Result is a view; use .copy() for an independent array
    win_inputs = np.lib.stride_tricks.as_strided(
        input_frames_seq,
        shape=(num_windows, window_size, frame_h, frame_w),
        strides=(input_frames_seq.strides[0] * stride,
                 input_frames_seq.strides[0],
                 input_frames_seq.strides[1],
                 input_frames_seq.strides[2]),
        writeable=False
    ).copy() # Safely copy

    win_real_gt = np.lib.stride_tricks.as_strided(
        real_gt_frames_seq,
        shape=(num_windows, window_size, frame_h, frame_w),
        strides=(real_gt_frames_seq.strides[0] * stride,
                 real_gt_frames_seq.strides[0],
                 real_gt_frames_seq.strides[1],
                 real_gt_frames_seq.strides[2]),
        writeable=False
    ).copy()

    win_noise_gt = np.lib.stride_tricks.as_strided(
        noise_gt_frames_seq,
        shape=(num_windows, window_size, frame_h, frame_w),
        strides=(noise_gt_frames_seq.strides[0] * stride,
                 noise_gt_frames_seq.strides[0],
                 noise_gt_frames_seq.strides[1],
                 noise_gt_frames_seq.strides[2]),
        writeable=False
    ).copy()

    win_eval_mask = np.lib.stride_tricks.as_strided(
        eval_mask_frames_seq,
        shape=(num_windows, window_size, frame_h, frame_w),
        strides=(eval_mask_frames_seq.strides[0] * stride,
                 eval_mask_frames_seq.strides[0],
                 eval_mask_frames_seq.strides[1],
                 eval_mask_frames_seq.strides[2]),
        writeable=False
    ).copy()

    # print(f"Created {num_windows} sliding window samples from a sequence of {num_total_frames} frames.")
    return win_inputs, win_real_gt, win_noise_gt, win_eval_mask