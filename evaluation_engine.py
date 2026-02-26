# evaluation_engine.py
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import pandas as pd # 전체 집계 시 DataFrame 사용
import math

# utils.py에서 compute_metrics 함수를 가져온다고 가정
from utils import compute_metrics
# 사용자께서 제공해주신 compute_metrics 함수 정의 (만약 utils.py에 없다면 여기에 직접 정의)
# 외부 파일에 있다면 위 import 문 사용
from sklearn.metrics import roc_auc_score # compute_metrics에서 사용


def compute_event_stream_metrics(S_denoised: np.ndarray, S_stream_GT: np.ndarray, config_obj) -> dict:
    epsilon_e = getattr(config_obj, 'EPSILON_EVENT_METRICS', 1e-10)

    if (S_denoised.ndim < 2 or S_denoised.shape[0] == 0) and \
       (S_stream_GT.ndim < 2 or S_stream_GT.shape[0] == 0):
        return {'tp_event': 0, 'fp_event': 0, 'fn_event': 0, 'tn_event': 0,
                'precision_event': 1.0, 'recall_event': 1.0, 'f1_event': 1.0,
                'noise_rejection_rate_event': 1.0, 'snr_tp_fp_event': 0.0,
                'total_gt_signal_events_stream': 0, 'total_gt_noise_events_stream': 0,
                'total_denoised_events_stream': 0}
    if (S_stream_GT.ndim < 2 or S_stream_GT.shape[0] == 0):
        denoised_count = S_denoised.shape[0] if S_denoised.ndim > 1 and S_denoised.shape[0] > 0 else 0
        return {'tp_event': 0, 'fp_event': denoised_count, 'fn_event': 0, 'tn_event': 0,
                'precision_event': 0.0, 'recall_event': 0.0, 'f1_event': 0.0,
                'noise_rejection_rate_event': 0.0, 'snr_tp_fp_event': float('-inf') if denoised_count > 0 else 0.0,
                'total_gt_signal_events_stream': 0, 'total_gt_noise_events_stream': 0,
                'total_denoised_events_stream': denoised_count}

    tp_e, fp_e = 0, 0
    if S_denoised.ndim > 1 and S_denoised.shape[0] > 0:
        for i in range(S_denoised.shape[0]):
            original_label_of_denoised_event = S_denoised[i, 0]
            if original_label_of_denoised_event == 0:
                tp_e += 1
            else:
                fp_e += 1
    
    total_signal_events_in_gt_stream = np.sum(S_stream_GT[:, 0] == 0) if S_stream_GT.ndim > 1 and S_stream_GT.shape[0] > 0 else 0
    total_noise_events_in_gt_stream = np.sum(S_stream_GT[:, 0] != 0) if S_stream_GT.ndim > 1 and S_stream_GT.shape[0] > 0 else 0

    fn_e = max(0, total_signal_events_in_gt_stream - tp_e)
    tn_e = max(0, total_noise_events_in_gt_stream - fp_e)

    precision_e = tp_e / (tp_e + fp_e + epsilon_e)
    recall_e = tp_e / (tp_e + fn_e + epsilon_e)
    f1_e = 2 * (precision_e * recall_e) / (precision_e + recall_e + epsilon_e)
    noise_rejection_rate_e = tn_e / (total_noise_events_in_gt_stream + epsilon_e) if total_noise_events_in_gt_stream > 0 else 1.0 if fp_e == 0 else 0.0
    
    if fp_e + epsilon_e == 0:
        snr_event = float('inf') if tp_e > 0 else 0.0
    elif tp_e + epsilon_e == 0:
        snr_event = float('-inf')
    else:
        snr_event = 10 * math.log10((tp_e + epsilon_e) / (fp_e + epsilon_e))
        
    return {
        'tp_event': tp_e, 'fp_event': fp_e, 'fn_event': fn_e, 'tn_event': tn_e,
        'precision_event': precision_e, 'recall_event': recall_e, 'f1_event': f1_e,
        'noise_rejection_rate_event': noise_rejection_rate_e,
        'snr_tp_fp_event': snr_event,
        'total_gt_signal_events_stream': total_signal_events_in_gt_stream,
        'total_gt_noise_events_stream': total_noise_events_in_gt_stream,
        'total_denoised_events_stream': S_denoised.shape[0] if S_denoised.ndim > 1 and S_denoised.shape[0] > 0 else 0
    }

def compute_masked_stream_metrics(
    S_stream_GT: np.ndarray,
    denoised_frames: np.ndarray,
    min_timestamp: float,
    fps: int,
    frame_width: int,
    frame_height: int,
    config_obj=None
) -> dict:
    """
    Denoised된 프레임을 '결정 마스크'로 사용하여 원본 GT 이벤트 스트림을 평가합니다.

    Args:
        S_stream_GT (np.ndarray): 원본 Ground Truth 이벤트 스트림 [label, x, y, t, p].
        denoised_frames (np.ndarray): 모델이 출력한 denoised 프레임 시퀀스 (T, H, W).
        fps (int): 프레임 변환에 사용된 FPS.
        frame_width (int): 프레임 너비.
        frame_height (int): 프레임 높이.
        config_obj: Epsilon 값을 가져오기 위한 설정 객체.

    Returns:
        dict: TP, FP, FN, TN 및 파생 메트릭을 포함하는 딕셔너리.
    """
    epsilon_e = getattr(config_obj, 'EPSILON_EVENT_METRICS', 1e-10)

    # --- 1. 예외 처리 ---
    if S_stream_GT.ndim < 2 or S_stream_GT.shape[0] == 0:
        # GT 이벤트가 없으면 평가 불가
        return {'tp_event': 0, 'fp_event': 0, 'fn_event': 0, 'tn_event': 0, 'f1_event': 1.0}

    # # --- 2. 평가 기준 시간 설정 ---
    # # GT의 실제 이벤트(label=0)를 기준으로 시작 시간 결정
    # gt_real_events = S_stream_GT[S_stream_GT[:, 0] == 0]
    # if len(gt_real_events) == 0:
    #     # 실제 이벤트가 없으면 가장 빠른 이벤트를 기준으로 함
    #     min_timestamp = S_stream_GT[0, 3]
    # else:
    #     min_timestamp = gt_real_events[0, 3]

    # --- 3. TP, FP, FN, TN 계산 ---
    tp_e, fp_e, fn_e, tn_e = 0, 0, 0, 0
    num_frames = denoised_frames.shape[0]

    # 원본 GT 스트림의 모든 이벤트를 순회
    for event in S_stream_GT:
        label, x, y, ts, _ = event
        x, y = int(x), int(y)

        # 이벤트의 시간(ts)을 프레임 인덱스(t_idx)로 변환
        t_idx = int((ts - min_timestamp) * fps)

        # 유효한 좌표인지 확인
        if not (0 <= t_idx < num_frames and 0 <= y < frame_height and 0 <= x < frame_width):
            continue

        # 모델의 결정 확인 (프레임 마스크 값)
        model_kept_event = (denoised_frames[t_idx, y, x] == 1)
        event_is_signal = (label == 0)

        if model_kept_event and event_is_signal:
            # 모델이 '유지'했고, 이벤트가 실제 '신호'인 경우 -> TP
            tp_e += 1
        elif model_kept_event and not event_is_signal:
            # 모델이 '유지'했는데, 이벤트가 실제 '노이즈'인 경우 -> FP
            fp_e += 1
        elif not model_kept_event and event_is_signal:
            # 모델이 '제거'했는데, 이벤트가 실제 '신호'인 경우 -> FN
            fn_e += 1
        elif not model_kept_event and not event_is_signal:
            # 모델이 '제거'했고, 이벤트가 실제 '노이즈'인 경우 -> TN
            tn_e += 1

    # --- 4. 최종 메트릭 계산 ---
    total_gt_signal = tp_e + fn_e
    total_gt_noise = fp_e + tn_e

    precision_e = tp_e / (tp_e + fp_e + epsilon_e)
    recall_e = tp_e / (tp_e + fn_e + epsilon_e)
    f1_e = 2 * (precision_e * recall_e) / (precision_e + recall_e + epsilon_e)
    noise_rejection_rate_e = tn_e / (total_gt_noise + epsilon_e) if total_gt_noise > 0 else 1.0

    if fp_e + epsilon_e == 0:
        snr_event = float('inf') if tp_e > 0 else 0.0
    elif tp_e + epsilon_e == 0:
        snr_event = float('-inf')
    else:
        snr_event = 10 * math.log10((tp_e + epsilon_e) / (fp_e + epsilon_e))

    return {
        'tp_event': tp_e, 'fp_event': fp_e, 'fn_event': fn_e, 'tn_event': tn_e,
        'precision_event': precision_e, 'recall_event': recall_e, 'f1_event': f1_e,
        'noise_rejection_rate_event': noise_rejection_rate_e,
        'event_snr_db': snr_event,
        'total_gt_signal_events_stream': total_gt_signal,
        'total_gt_noise_events_stream': total_gt_noise,
        'total_denoised_events_stream': tp_e + fp_e
    }

def evaluate_model_on_dataset(
    model_to_evaluate: nn.Module,
    test_data_list: list[dict],
    config_obj,
    device: torch.device
) -> tuple[dict | None, list[dict] | None,
           list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None,
           list[dict] | None, dict | None]:

    if not test_data_list:
        print("ERROR (evaluation_engine): test_data_list is empty. Cannot evaluate.")
        return None, None, None, None, None, None, None

    model_to_evaluate.eval()

    per_file_frame_metrics_list = []
    per_file_event_stream_metrics_list = []
    all_files_preds_list = []
    all_files_probs_list = []
    all_files_targets_list = []
    all_files_eval_masks_list = []
    all_event_stream_probs = []
    all_event_stream_labels = []

    eval_batch_size_per_inference = getattr(config_obj, 'EVAL_BATCH_SIZE', config_obj.BATCH_SIZE * 2)

    print(f"\n--- Starting Evaluation on {len(test_data_list)} Test Files ---")
    for file_idx, single_file_data_dict in enumerate(test_data_list):
        file_path = single_file_data_dict['file_path']
        base_filename = os.path.basename(file_path)
        print(f"\nProcessing Test File {file_idx + 1}/{len(test_data_list)}: {base_filename}")

        input_frames_np = single_file_data_dict.get('input_frames')
        real_event_gt_np = single_file_data_dict.get('real_event_gt')
        noise_event_gt_np = single_file_data_dict.get('noise_event_gt')
        eval_mask_np = single_file_data_dict.get('evaluation_mask')
        original_labeled_event_stream = single_file_data_dict.get('original_labeled_event_stream') # Restore this!
        min_ts_from_data = single_file_data_dict.get('min_timestamp')

        # --- Lazy Loading Support ---
        if input_frames_np is None and 'processed_path' in single_file_data_dict:
            try:
                processed_path = single_file_data_dict['processed_path']
                if os.path.exists(processed_path):
                     # Load packed data: [T, 4, H, W]
                    stacked_data = np.load(processed_path, mmap_mode='r')
                    input_frames_np = stacked_data[:, 0]
                    real_event_gt_np = stacked_data[:, 1]
                    noise_event_gt_np = stacked_data[:, 2]
                    eval_mask_np = stacked_data[:, 3]
                    
                    # Load Raw Events for Stream Metrics if not present
                    if original_labeled_event_stream is None and 'file_path' in single_file_data_dict:
                        original_labeled_event_stream = np.load(single_file_data_dict['file_path'])

                else:
                    print(f"  - Error: Processed path not found: {processed_path}")
            except Exception as e:
                print(f"  - Error loading from processed path {processed_path}: {e}")

        if input_frames_np is None or input_frames_np.ndim != 3 or input_frames_np.shape[0] == 0:
            print(f"  - Skipping file {base_filename}: input_frames_np is None, has invalid dimensions {input_frames_np.shape if input_frames_np is not None else 'None'}, or no frames.")
            per_file_frame_metrics_list.append({'file_path': file_path, 'error': 'Invalid or empty input_frames_np'})
            # 스트림 메트릭 리스트에도 해당 파일에 대한 플레이스홀더 또는 에러 정보 추가
            per_file_event_stream_metrics_list.append({'file_path': file_path, 'error': 'Skipped due to invalid input_frames_np'})
            continue
            
        num_total_frames_in_file, height, width = input_frames_np.shape

        inputs_tensor_full_seq = torch.FloatTensor(
            np.expand_dims(input_frames_np, axis=1)
        ).unsqueeze(0).to(device)
        
        file_all_logits_tensors_for_metrics = []
        file_all_probs_tensors_for_vis = []
        file_all_preds_tensors_for_vis = []

        # Initialize SNN state (mem) for this file
        current_mem = None

        print(f"  - Running inference on {num_total_frames_in_file} frames (batch size: {eval_batch_size_per_inference})...")
        with torch.no_grad():
            for i in tqdm(range(0, num_total_frames_in_file, eval_batch_size_per_inference), dynamic_ncols=True,
                          desc=f"  Inferring {base_filename[:20]}...", leave=False, unit="batch"):
                current_batch_input_tensor = inputs_tensor_full_seq[:, i:min(i + eval_batch_size_per_inference, num_total_frames_in_file), :, :, :]
                if current_batch_input_tensor.shape[1] == 0: continue

                # Stateful Inference: Pass and update current_mem
                batch_logits_output, current_mem = model_to_evaluate(current_batch_input_tensor, mem=current_mem, regulate=True)
                
                # Detach mem to prevent computation graph from growing indefinitely (though torch.no_grad makes this less critical)
                if current_mem is not None:
                     current_mem = current_mem.detach()

                file_all_logits_tensors_for_metrics.append(batch_logits_output.squeeze(0).cpu())
                
                batch_logits_pos_class = batch_logits_output.squeeze(0)[:, 1, :, :]
                batch_probs_pos_class = torch.sigmoid(batch_logits_pos_class)
                batch_preds_pos_class = (batch_probs_pos_class > config_obj.EVALUATION_THRESHOLD).float()
                
                file_all_probs_tensors_for_vis.append(batch_probs_pos_class.cpu())
                file_all_preds_tensors_for_vis.append(batch_preds_pos_class.cpu())

        if not file_all_logits_tensors_for_metrics:
            print(f"  - No inference results for {base_filename} (logits list empty). Skipping metrics for this file.")
            per_file_frame_metrics_list.append({'file_path': file_path, 'error': 'No inference results (logits list empty)'})
            per_file_event_stream_metrics_list.append({'file_path': file_path, 'error': 'No inference results (logits list empty) for event stream metrics'})
            continue
            
        full_file_logits_for_metrics_tensor = torch.cat(file_all_logits_tensors_for_metrics, dim=0).cpu()
        full_file_preds_np_for_vis = torch.cat(file_all_preds_tensors_for_vis, dim=0).numpy()
        full_file_probs_np_for_vis = torch.cat(file_all_probs_tensors_for_vis, dim=0).numpy()

        all_files_preds_list.append(full_file_preds_np_for_vis)
        all_files_probs_list.append(full_file_probs_np_for_vis)
        all_files_targets_list.append(real_event_gt_np)
        all_files_eval_masks_list.append(eval_mask_np)

        print("  - Calculating frame-level metrics for this file...")
        real_event_gt_tensor = torch.FloatTensor(real_event_gt_np).unsqueeze(0)
        noise_event_gt_tensor = torch.FloatTensor(noise_event_gt_np).unsqueeze(0)
        eval_mask_tensor = torch.FloatTensor(eval_mask_np).unsqueeze(0)

        
        try:
            metrics_for_this_file = compute_metrics(
                logits=full_file_logits_for_metrics_tensor.unsqueeze(0),
                real_event_gt=real_event_gt_tensor,
                noise_event_gt=noise_event_gt_tensor,
                evaluation_mask=eval_mask_tensor,
                config_obj=config_obj
            )
            metrics_for_this_file['file_path'] = file_path
            per_file_frame_metrics_list.append(metrics_for_this_file)
            print(f"  - File Frame Metrics: F1={metrics_for_this_file.get('f1', 0.0):.4f}, Recall={metrics_for_this_file.get('recall', 0.0):.4f}, Precision={metrics_for_this_file.get('precision', 0.0):.4f}")
            if config_obj.CALC_SNR_TP_FP and 'snr_tp_fp' in metrics_for_this_file:
                snr_val = metrics_for_this_file['snr_tp_fp']
                snr_display = f"{snr_val:.2f} dB" if isinstance(snr_val, (float, np.floating)) and np.isfinite(snr_val) else str(snr_val)
                print(f"                 SNR(TP/FP): {snr_display}")
        except Exception as e_metric:
            print(f"  - Error calculating frame-level metrics for file {base_filename}: {e_metric}")
            per_file_frame_metrics_list.append({'file_path': file_path, 'error': f'Frame metrics calculation error: {e_metric}'})

        if original_labeled_event_stream is not None and \
           original_labeled_event_stream.ndim == 2 and \
           original_labeled_event_stream.shape[0] > 0 and \
           original_labeled_event_stream.shape[1] == 5:
            print(f"  - Performing event-stream level comparison for {base_filename}...")
            S_denoised = []
            min_ts_original_stream = original_labeled_event_stream[0, 3] # 이미 정렬된 것으로 가정
            time_window_duration = 1.0 / config_obj.FPS

            for ev_idx in range(len(original_labeled_event_stream)):
                original_event = original_labeled_event_stream[ev_idx]
                ev_x_orig, ev_y_orig = int(original_event[1]), int(original_event[2])
                ev_ts_orig = original_event[3]

                if time_window_duration <= 1e-9: frame_idx_for_event = 0
                else: frame_idx_for_event = min(int(max(0, ev_ts_orig - min_ts_original_stream) / time_window_duration), num_total_frames_in_file - 1)
                frame_idx_for_event = max(0, frame_idx_for_event)

                if 0 <= ev_x_orig < width and 0 <= ev_y_orig < height:
                    if frame_idx_for_event < full_file_preds_np_for_vis.shape[0] and \
                       full_file_preds_np_for_vis[frame_idx_for_event, ev_y_orig, ev_x_orig] > 0.5:
                        S_denoised.append(original_event)
            
            S_denoised_np = np.array(S_denoised) if S_denoised else np.empty((0,5))
            
            # stream_metrics = compute_event_stream_metrics(S_denoised_np, original_labeled_event_stream, config_obj)
            stream_metrics = compute_masked_stream_metrics(
                S_stream_GT=original_labeled_event_stream,
                denoised_frames=full_file_preds_np_for_vis, # 모델이 예측한 프레임 마스크
                min_timestamp=min_ts_from_data,
                fps=config_obj.FPS,                         # 설정에서 FPS 값 가져오기
                frame_width=width,                          # 이전에 정의된 변수 사용
                frame_height=height,                        # 이전에 정의된 변수 사용
                config_obj=config_obj
            )

            tp_e = stream_metrics.get('tp_event', 0)
            fp_e = stream_metrics.get('fp_event', 0)
            fn_e = stream_metrics.get('fn_event', 0)
            tn_e = stream_metrics.get('tn_event', 0)
            epsilon = 1e-10

            # 1. Duan et al. (LED)의 DA, SR, NR 계산
            gp = tp_e + fn_e  # Ground-truth Positives
            gn = fp_e + tn_e  # Ground-truth Negatives
            sr = tp_e / (gp + epsilon)  # Signal Retain (Recall/TPR과 동일)
            nr = tn_e / (gn + epsilon)  # Noise Removal (Specificity/TNR과 동일)
            da = 0.5 * (sr + nr)      # Denoising Accuracy

            stream_metrics['signal_retain_sr'] = sr
            stream_metrics['noise_removal_nr'] = nr
            stream_metrics['denoising_accuracy_da'] = da

            # 2. Wu et al.의 EDP, ESNR 계산
            total_denoised = tp_e + fp_e
            edp = tp_e / (total_denoised + epsilon) # Event Denoising Precision (Precision과 동일)
            
            # ESNR (dB) 계산
            if fp_e == 0:
                esnr = float('inf') if tp_e > 0 else 0.0
            else:
                esnr = 20 * math.log10((tp_e + epsilon) / (fp_e + epsilon))

            stream_metrics['event_denoising_precision_edp'] = edp
            stream_metrics['event_esnr_db'] = esnr
            # <<< 계산 로직 추가 완료 >>>

            # <<< [핵심 수정] 개별 파일의 이벤트 스트림 AUC 계산 및 저장 >>>
            file_event_labels = []
            file_event_probs = []

            # <<< [ADDED] 이벤트 스트림 AUC를 위한 데이터 수집 >>>
            valid_indices = [i for i, event in enumerate(original_labeled_event_stream) if 0 <= int((event[3] - min_ts_from_data) * config_obj.FPS) < num_total_frames_in_file and 0 <= int(event[2]) < height and 0 <= int(event[1]) < width]
            valid_events = original_labeled_event_stream[valid_indices]
            if len(valid_events) > 0:
                labels = (valid_events[:, 0] == 0).astype(int)
                t_indices = np.floor((valid_events[:, 3] - min_ts_from_data) * config_obj.FPS).astype(int)
                y_indices = valid_events[:, 2].astype(int)
                x_indices = valid_events[:, 1].astype(int)
                probs = full_file_probs_np_for_vis[t_indices, y_indices, x_indices]
                all_event_stream_labels.append(labels)
                all_event_stream_probs.append(probs)

                # 개별 파일 AUC 계산
                if len(np.unique(labels)) > 1:
                    try:
                        file_auc = roc_auc_score(labels, probs)
                        stream_metrics['event_stream_auc'] = file_auc # 딕셔너리에 추가
                    except Exception as e:
                        stream_metrics['event_stream_auc'] = np.nan # 계산 불가 시 NaN으로 처리
                else:
                    stream_metrics['event_stream_auc'] = np.nan # 단일 클래스만 존재 시 NaN
            
            per_file_event_stream_metrics_list.append(stream_metrics)

            stream_metrics['file_path'] = file_path
            per_file_event_stream_metrics_list.append(stream_metrics)
            print(f"  - Event-Stream Metrics: F1={stream_metrics.get('f1_event',0):.4f}, DA={stream_metrics.get('denoising_accuracy_da',0):.4f}, ESNR={stream_metrics.get('event_snr_esnr_db',0):.2f}dB")
            # print(f"  - Event-Stream Metrics: F1={stream_metrics.get('f1_event',0):.4f}, DA={da:.4f}, EDP={edp:.4f}, ESNR={esnr:.2f}dB")
            # print(f"  - Event-Stream Metrics: F1_e={stream_metrics.get('f1_event',0):.4f}, Recall_e={stream_metrics.get('recall_event',0):.4f}, Precision_e={stream_metrics.get('precision_event',0):.4f}, NoiseRej_e={stream_metrics.get('noise_rejection_rate_event',0):.4f}, SNR_e={stream_metrics.get('snr_tp_fp_event',0):.2f}dB")
        else:
            error_msg = 'Original labeled event stream not available'
            if original_labeled_event_stream is not None:
                if original_labeled_event_stream.ndim != 2 or original_labeled_event_stream.shape[0] == 0 or original_labeled_event_stream.shape[1] != 5:
                    error_msg = f'Original labeled event stream has invalid shape: {original_labeled_event_stream.shape if original_labeled_event_stream is not None else "None"}'
            per_file_event_stream_metrics_list.append({'file_path': file_path, 'error': error_msg})


    final_aggregated_frame_metrics = None
    valid_frame_metrics_for_agg = [m for m in per_file_frame_metrics_list if m and 'error' not in m]
    if valid_frame_metrics_for_agg:
        metrics_df = pd.DataFrame(valid_frame_metrics_for_agg)
        if not metrics_df.empty:
            final_aggregated_frame_metrics = {}
            numeric_cols = metrics_df.select_dtypes(include=np.number).columns
            exclude_cols = {'tp_event', 'fp_event', 'tn_event', 'fn_event',
                            'total_gt_signal_events_stream', 'total_gt_noise_events_stream',
                            'total_denoised_events_stream'}
            cols_for_mean = [col for col in numeric_cols if col not in exclude_cols]
            # cols_for_mean = [col for col in numeric_cols if col not in ['tp', 'fp', 'tn', 'fn']]
            for col in cols_for_mean:
                if col in metrics_df: final_aggregated_frame_metrics[f'avg_per_file_{col}'] = metrics_df[col].mean(skipna=True)

            total_tp = metrics_df['tp'].sum(skipna=True)
            total_fp = metrics_df['fp'].sum(skipna=True)
            total_tn = metrics_df['tn'].sum(skipna=True)
            total_fn = metrics_df['fn'].sum(skipna=True)
            final_aggregated_frame_metrics['total_tp'] = total_tp
            final_aggregated_frame_metrics['total_fp'] = total_fp
            final_aggregated_frame_metrics['total_tn'] = total_tn # 이전 코드에 있던 오타 수정: final_aggregated_frame_etrics
            final_aggregated_frame_metrics['total_fn'] = total_fn
            epsilon = 1e-10
            final_aggregated_frame_metrics['overall_accuracy'] = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + epsilon) if (total_tp + total_fp + total_tn + total_fn) > 0 else 0.0
            final_aggregated_frame_metrics['overall_precision'] = total_tp / (total_tp + total_fp + epsilon)
            final_aggregated_frame_metrics['overall_recall'] = total_tp / (total_tp + total_fn + epsilon)
            final_aggregated_frame_metrics['overall_f1'] = 2 * (final_aggregated_frame_metrics['overall_precision'] * final_aggregated_frame_metrics['overall_recall']) / \
                                            (final_aggregated_frame_metrics['overall_precision'] + final_aggregated_frame_metrics['overall_recall'] + epsilon)
            if config_obj.CALC_SNR_TP_FP:
                if total_fp + epsilon == 0: final_snr_tp_fp = float('inf') if total_tp > 0 else 0.0
                elif total_tp + epsilon == 0: final_snr_tp_fp = float('-inf')
                else: final_snr_tp_fp = 10 * math.log10((total_tp + epsilon) / (total_fp + epsilon))
                final_aggregated_frame_metrics['overall_snr_tp_fp'] = final_snr_tp_fp

            try:
                masked_probs = [p[m > 0] for p, m in zip(all_files_probs_list, all_files_eval_masks_list)]
                masked_targets = [t[m > 0] for t, m in zip(all_files_targets_list, all_files_eval_masks_list)]
                final_probs = np.concatenate(masked_probs)
                final_targets = np.concatenate(masked_targets)
                if len(np.unique(final_targets)) > 1:
                    final_aggregated_frame_metrics['overall_frame_auc'] = roc_auc_score(final_targets, final_probs)
            except Exception as e:
                print(f"Warning: Could not compute frame-level AUC. Reason: {e}")

    else:
        print("Warning: No valid frame-level metrics to aggregate.")
    

    # final_aggregated_event_stream_metrics = None
    final_aggregated_event_stream_metrics = {} 
    valid_event_stream_metrics_for_agg = [m for m in per_file_event_stream_metrics_list if m and 'error' not in m]
    if valid_event_stream_metrics_for_agg:
        event_stream_metrics_df = pd.DataFrame(valid_event_stream_metrics_for_agg)
        if not event_stream_metrics_df.empty:
            # final_aggregated_event_stream_metrics = {}
            # <<< [핵심 수정] 숫자 타입 선택 전에 inf 값을 NaN으로 변환 >>>
            event_stream_metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            numeric_cols = event_stream_metrics_df.select_dtypes(include=np.number).columns
            exclude_cols = {'tp_event', 'fp_event', 'tn_event', 'fn_event',
                            'total_gt_signal_events_stream', 'total_gt_noise_events_stream',
                            'total_denoised_events_stream'}
            cols_for_mean = [col for col in numeric_cols if col not in exclude_cols]
            
            for col in cols_for_mean:
                # 이제 inf가 없으므로 finite_values 변환이 필요 없음
                if not event_stream_metrics_df[col].dropna().empty:
                    final_aggregated_event_stream_metrics[f'avg_per_file_{col}'] = event_stream_metrics_df[col].mean(skipna=True)
                else:
                    final_aggregated_event_stream_metrics[f'avg_per_file_{col}'] = np.nan
            
            # numeric_cols = event_stream_metrics_df.select_dtypes(include=np.number).columns
            # cols_for_mean = [col for col in numeric_cols if 'tp_' not in col and 'fp_' not in col and 'tn_' not in col and 'fn_' not in col]
            # for col in cols_for_mean:
            #     if col in event_stream_metrics_df:
            #         # 무한대(inf) 값을 NaN으로 바꿔 평균 계산에서 제외
            #         finite_values = event_stream_metrics_df[col].replace([np.inf, -np.inf], np.nan)
            #         if not finite_values.dropna().empty:
            #             final_aggregated_event_stream_metrics[f'avg_per_file_{col}'] = finite_values.mean(skipna=True)
            #         else:
            #             final_aggregated_event_stream_metrics[f'avg_per_file_{col}'] = np.nan
            # cols_to_avg_stream = ['precision_event', 'recall_event', 'f1_event', 'noise_rejection_rate_event', 'snr_tp_fp_event']
            # cols_to_avg_stream = ['precision_event', 'recall_event', 'f1_event', 
            #                       'noise_rejection_rate_event', 'snr_tp_fp_event',
            #                       'signal_retain_sr', 'noise_removal_nr', 
            #                       'denoising_accuracy_da', 'event_denoising_precision_edp',
            #                       'event_snr_esnr_db'] # ESNR 추가
            # for col in cols_to_avg_stream:
            #     if col in event_stream_metrics_df and pd.api.types.is_numeric_dtype(event_stream_metrics_df[col]):
            #         finite_values = event_stream_metrics_df[col][np.isfinite(event_stream_metrics_df[col])]
            #         if not finite_values.empty:
            #             final_aggregated_event_stream_metrics[f'avg_per_file_{col}'] = finite_values.mean()
            #         else:
            #              final_aggregated_event_stream_metrics[f'avg_per_file_{col}'] = np.nan
            #     else: # 컬럼이 없거나 숫자형이 아닌 경우
            #         final_aggregated_event_stream_metrics[f'avg_per_file_{col}'] = np.nan


            total_tp_e = event_stream_metrics_df['tp_event'].sum(skipna=True)
            total_fp_e = event_stream_metrics_df['fp_event'].sum(skipna=True)
            total_fn_e = event_stream_metrics_df['fn_event'].sum(skipna=True) # skipnaTrue 오타 수정
            total_tn_e = event_stream_metrics_df['tn_event'].sum(skipna=True)
            final_aggregated_event_stream_metrics['total_tp_event'] = total_tp_e
            final_aggregated_event_stream_metrics['total_fp_event'] = total_fp_e
            final_aggregated_event_stream_metrics['total_fn_event'] = total_fn_e
            final_aggregated_event_stream_metrics['total_tn_event'] = total_tn_e
            
            epsilon_e = getattr(config_obj, 'EPSILON_EVENT_METRICS', 1e-10)
            agg_precision_e = total_tp_e / (total_tp_e + total_fp_e + epsilon_e)
            agg_recall_e = total_tp_e / (total_tp_e + total_fn_e + epsilon_e)
            agg_f1_e = 2 * (agg_precision_e * agg_recall_e) / (agg_precision_e + agg_recall_e + epsilon_e)
            total_gt_noise_stream_all_files = event_stream_metrics_df['total_gt_noise_events_stream'].sum(skipna=True)
            agg_noise_rejection_e = total_tn_e / (total_gt_noise_stream_all_files + epsilon_e) if total_gt_noise_stream_all_files > 0 else 1.0 if total_fp_e == 0 else 0.0
            
            if total_fp_e + epsilon_e == 0: agg_snr_event = float('inf') if total_tp_e > 0 else 0.0
            elif total_tp_e + epsilon_e == 0: agg_snr_event = float('-inf')
            else: agg_snr_event = 10 * math.log10((total_tp_e + epsilon_e) / (total_fp_e + epsilon_e))

            final_aggregated_event_stream_metrics['overall_precision_event'] = agg_precision_e
            final_aggregated_event_stream_metrics['overall_recall_event'] = agg_recall_e
            final_aggregated_event_stream_metrics['overall_f1_event'] = agg_f1_e
            final_aggregated_event_stream_metrics['overall_noise_rejection_rate_event'] = agg_noise_rejection_e
            final_aggregated_event_stream_metrics['overall_snr_tp_fp_event'] = agg_snr_event

            # <<< [신규] 전체 데이터셋에 대한 DA, ESNR 집계 >>>
            total_gp_e = total_tp_e + total_fn_e
            total_gn_e = total_fp_e + total_tn_e
            overall_sr = total_tp_e / (total_gp_e + epsilon)
            overall_nr = total_tn_e / (total_gn_e + epsilon)
            overall_da = 0.5 * (overall_sr + overall_nr)
            
            overall_edp = total_tp_e / (total_tp_e + total_fp_e + epsilon)

            if total_fp_e == 0:
                overall_esnr = float('inf') if total_tp_e > 0 else 0.0
            else:
                overall_esnr = 20 * math.log10((total_tp_e + epsilon) / (total_fp_e + epsilon))
            
            final_aggregated_event_stream_metrics['overall_signal_retain_sr'] = overall_sr
            final_aggregated_event_stream_metrics['overall_noise_removal_nr'] = overall_nr
            final_aggregated_event_stream_metrics['overall_denoising_accuracy_da'] = overall_da
            final_aggregated_event_stream_metrics['overall_event_denoising_precision_edp'] = overall_edp
            final_aggregated_event_stream_metrics['overall_event_snr_esnr_db'] = overall_esnr
            # <<< 집계 로직 추가 완료 >>>

            try:
                if all_event_stream_labels and all_event_stream_probs:
                    final_event_labels = np.concatenate(all_event_stream_labels)
                    final_event_probs = np.concatenate(all_event_stream_probs)
                    if len(np.unique(final_event_labels)) > 1:
                        final_aggregated_event_stream_metrics['overall_event_stream_auc'] = roc_auc_score(final_event_labels, final_event_probs)
            except Exception as e:
                print(f"Warning: Could not compute event-stream-level AUC. Reason: {e}")
    else:
        print("Warning: No valid event-stream level metrics to aggregate.")

    print(f"--- Model Evaluation Finished ---")
    
    # return (final_aggregated_frame_metrics, per_file_frame_metrics_list,
    #         all_files_preds_list, all_files_probs_list, all_files_targets_list,
    #         per_file_event_stream_metrics_list, final_aggregated_event_stream_metrics)

    # return (final_aggregated_frame_metrics, per_file_frame_metrics_list,
    #         all_files_preds_list, all_files_probs_list, all_files_targets_list,
    #         all_files_eval_masks_list, # 반환 값 추가
    #         per_file_event_stream_metrics_list, final_aggregated_event_stream_metrics)
    return (final_aggregated_frame_metrics, per_file_frame_metrics_list,
            all_files_preds_list, all_files_probs_list, all_files_targets_list, all_files_eval_masks_list,
            per_file_event_stream_metrics_list, final_aggregated_event_stream_metrics)
