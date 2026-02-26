# data_processing.py
import numpy as np
import os
import glob
from tqdm import tqdm # 파일 처리 진행 상황 표시
import math
# --- ★ 1. events_to_frames 함수 수정 ★ ---
# 반환 값에 min_timestamp를 추가합니다.
def events_to_frames(events: np.ndarray,
                     fps: int,
                     frame_width: int,
                     frame_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: # 반환 타입에 float 추가
    """
    Raw event data를 다양한 프레임 표현으로 변환합니다. (수정된 버전)
    """
    if not isinstance(events, np.ndarray) or events.ndim != 2 or events.shape[1] != 5 or len(events) == 0:
        empty_frames = np.empty((0, frame_height, frame_width), dtype=np.float32)
        # min_timestamp로 0.0을 반환
        return empty_frames, empty_frames, empty_frames, empty_frames, 0.0

    # --- 1. 시간 기준점 설정 및 프레임 인덱스 계산 (벡터화 방식) ---
    events = events[np.argsort(events[:, 3])]
    min_timestamp = events[0, 3] # 이 값을 반환해야 합니다.
    time_window_duration = 1.0 / fps

    # ... (함수의 나머지 부분은 기존과 동일) ...
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

    # ★ 마지막 반환 값에 min_timestamp 추가 ★
    return all_input_frames, all_real_event_gt, all_noise_event_gt, all_evaluation_mask, min_timestamp
# def events_to_frames(events: np.ndarray,
#                      fps: int,
#                      frame_width: int,
#                      frame_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Raw event data를 다양한 프레임 표현으로 변환합니다. (수정된 버전)
#     """
#     if not isinstance(events, np.ndarray) or events.ndim != 2 or events.shape[1] != 5 or len(events) == 0:
#         empty_frames = np.empty((0, frame_height, frame_width), dtype=np.float32)
#         return empty_frames, empty_frames, empty_frames, empty_frames

#     # --- 1. 시간 기준점 설정 및 프레임 인덱스 계산 (벡터화 방식) ---
#     events = events[np.argsort(events[:, 3])]
#     min_timestamp = events[0, 3]
#     time_window_duration = 1.0 / fps

#     # 모든 이벤트에 대한 프레임 인덱스를 한 번에 계산
#     frame_indices = np.floor((events[:, 3] - min_timestamp) / time_window_duration).astype(int)
    
#     # 필요한 총 프레임 수 결정
#     num_frames = frame_indices.max() + 1
    
#     # --- 2. 프레임 배열 초기화 ---
#     all_input_frames = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
#     all_real_event_gt = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
#     all_noise_event_gt = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
    
#     # --- 3. 프레임에 이벤트 채우기 (고급 인덱싱) ---
#     # 유효한 좌표를 가진 이벤트만 필터링
#     x_coords = events[:, 1].astype(int)
#     y_coords = events[:, 2].astype(int)
#     valid_mask = (x_coords >= 0) & (x_coords < frame_width) & \
#                  (y_coords >= 0) & (y_coords < frame_height)

#     valid_indices = frame_indices[valid_mask]
#     valid_y = y_coords[valid_mask]
#     valid_x = x_coords[valid_mask]
#     valid_labels = events[valid_mask, 0]

#     # 모든 이벤트 위치를 한 번에 1로 설정
#     all_input_frames[valid_indices, valid_y, valid_x] = 1.0

#     # 실제 이벤트(label=0) 위치를 1로 설정
#     real_mask = valid_labels == 0
#     all_real_event_gt[valid_indices[real_mask], valid_y[real_mask], valid_x[real_mask]] = 1.0
    
#     # 노이즈 이벤트(label>0) 위치를 1로 설정
#     noise_mask = valid_labels != 0
#     all_noise_event_gt[valid_indices[noise_mask], valid_y[noise_mask], valid_x[noise_mask]] = 1.0

#     # evaluation_mask는 input_frames와 동일
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
            raw_events = np.load(npy_file_path) # 원본 이벤트 스트림 (레이블 포함)

            if not isinstance(raw_events, np.ndarray) or raw_events.ndim != 2 or raw_events.shape[1] != 5:
                print(f"    - Warning for {base_filename}: Invalid data format. Skipping file.")
                continue
            if raw_events.shape[0] == 0:
                print(f"    - Info for {base_filename}: File is empty. Skipping file.")
                continue

            # events_to_frames 함수 호출하여 BAF 프레임 데이터 생성
            input_f, real_gt_f, noise_gt_f, eval_mask_f, min_ts = events_to_frames(
                raw_events, # 여기서 raw_events는 신호/노이즈 레이블이 있는 원본 스트림
                fps=config_obj.FPS,
                frame_width=config_obj.FRAME_WIDTH,
                frame_height=config_obj.FRAME_HEIGHT
            )

            if input_f.shape[0] >= config_obj.WINDOW_SIZE: # WINDOW_SIZE는 학습/추론 시퀀스 길이
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
                # tqdm 사용 시 로그 최소화
                pass 

        except Exception as e:
            print(f"    - Error processing file {base_filename}: {e}. Skipping.")

    print(f"--- Finished processing {dataset_name} folder. {len(processed_files_data_list)} files successfully processed and saved to {final_save_dir}. ---")
    return processed_files_data_list


# (참고) 슬라이딩 윈도우 생성 함수 (원본 파일의 In[8] 부분)
# 이 함수는 EventFrameLazyDataset을 사용할 경우 dataset.py 내부에서 직접 슬라이싱하므로
# data_processing.py에 필수는 아니지만, 참고용으로 남겨둘 수 있습니다.
# 만약 미리 모든 윈도우를 생성하는 방식(EventFrameWindowDataset)을 사용한다면 이 함수가 필요합니다.
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

    # np.lib.stride_tricks.as_strided를 사용한 효율적인 윈도우 생성
    # 결과는 view이므로, 독립적인 배열을 원하면 .copy() 사용
    win_inputs = np.lib.stride_tricks.as_strided(
        input_frames_seq,
        shape=(num_windows, window_size, frame_h, frame_w),
        strides=(input_frames_seq.strides[0] * stride,
                 input_frames_seq.strides[0],
                 input_frames_seq.strides[1],
                 input_frames_seq.strides[2]),
        writeable=False
    ).copy() # 안전하게 copy

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