"""
Utility functions for classical filter evaluation.

Adapted from v8_bconvsnn/utils.py and evaluation_engine.py
"""

import numpy as np
import math
from sklearn.metrics import roc_auc_score, confusion_matrix
from typing import Dict, Tuple


def compute_metrics(predictions: np.ndarray, 
                   ground_truth: np.ndarray, 
                   mask: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics from predictions and ground truth.
    
    Args:
        predictions: Binary predictions (0=signal, 1=noise) or probabilities
        ground_truth: Binary ground truth labels (0=signal, 1=noise)
        mask: Optional mask to select which pixels/events to evaluate
    
    Returns:
        Dictionary containing various metrics
    """
    if mask is not None:
        predictions = predictions[mask > 0]
        ground_truth = ground_truth[mask > 0]
    
    # Ensure binary predictions
    if predictions.dtype == np.float32 or predictions.dtype == np.float64:
        pred_binary = (predictions > 0.5).astype(np.uint8)
    else:
        pred_binary = predictions.astype(np.uint8)
    
    gt_binary = ground_truth.astype(np.uint8)
    
    # Compute confusion matrix
    # For event denoising: 0=signal (positive), 1=noise (negative)
    tn, fp, fn, tp = confusion_matrix(gt_binary, pred_binary, labels=[1, 0]).ravel()
    
    # Basic metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Precision, Recall, F1 for signal detection
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Denoising Accuracy (DA) - percentage of correctly classified events
    da = accuracy
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC-ROC
    # For binary predictions, AUC = 0.5 * (TPR + (1 - FPR)) = 0.5 * (TPR + TNR)
    # TPR = Recall, TNR = Specificity
    try:
        if len(np.unique(predictions)) > 2:  # Probabilities
            auc = roc_auc_score(gt_binary, predictions)
        else:
            # Binary predictions: AUC from single point on ROC curve
            # AUC = 0.5 * (1 + TPR - FPR) = 0.5 * (Recall + Specificity)
            auc = 0.5 * (recall + specificity)
    except:
        auc = 0.5 * (recall + specificity)  # Fallback to balanced accuracy
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'denoising_accuracy_da': da,
        'specificity': specificity,
        'auc': auc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_events': int(total)
    }


def events_to_frame_predictions(events: np.ndarray,
                                predictions: np.ndarray,
                                fps: int,
                                frame_width: int,
                                frame_height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert event-level predictions to frame-based representation.
    
    Args:
        events: (N, 5) array [label, x, y, t, polarity]
        predictions: (N,) array of binary predictions (0=signal, 1=noise)
        fps: Frames per second
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    
    Returns:
        predicted_frames: (T, H, W) - frames with predicted signal events
        gt_signal_frames: (T, H, W) - frames with ground truth signal events
        gt_noise_frames: (T, H, W) - frames with ground truth noise events
    """
    if len(events) == 0:
        empty = np.zeros((0, frame_height, frame_width), dtype=np.float32)
        return empty, empty, empty
    
    # Sort by time
    events = events[np.argsort(events[:, 3])]
    predictions = predictions[np.argsort(events[:, 3])]
    
    # Calculate frame indices
    min_timestamp = events[0, 3]
    time_window_duration = 1.0 / fps
    frame_indices = np.floor((events[:, 3] - min_timestamp) / time_window_duration).astype(int)
    num_frames = frame_indices.max() + 1
    
    # Initialize frames
    predicted_frames = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
    gt_signal_frames = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
    gt_noise_frames = np.zeros((num_frames, frame_height, frame_width), dtype=np.float32)
    
    # Extract coordinates
    x_coords = events[:, 1].astype(int)
    y_coords = events[:, 2].astype(int)
    gt_labels = events[:, 0].astype(int)
    
    # Filter valid coordinates
    valid_mask = (x_coords >= 0) & (x_coords < frame_width) & \
                 (y_coords >= 0) & (y_coords < frame_height)
    
    valid_frames = frame_indices[valid_mask]
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_preds = predictions[valid_mask]
    valid_gt = gt_labels[valid_mask]
    
    # Fill predicted frames (only events predicted as signal)
    signal_mask = valid_preds == 0
    predicted_frames[valid_frames[signal_mask], valid_y[signal_mask], valid_x[signal_mask]] = 1.0
    
    # Fill ground truth frames
    gt_signal_mask = valid_gt == 0
    gt_signal_frames[valid_frames[gt_signal_mask], valid_y[gt_signal_mask], valid_x[gt_signal_mask]] = 1.0
    
    gt_noise_mask = valid_gt != 0
    gt_noise_frames[valid_frames[gt_noise_mask], valid_y[gt_noise_mask], valid_x[gt_noise_mask]] = 1.0
    
    return predicted_frames, gt_signal_frames, gt_noise_frames


def compute_frame_level_metrics(predicted_frames: np.ndarray,
                                gt_signal_frames: np.ndarray,
                                gt_noise_frames: np.ndarray) -> Dict[str, float]:
    """
    Compute frame-level metrics from frame representations.
    
    Args:
        predicted_frames: (T, H, W) - predicted signal events
        gt_signal_frames: (T, H, W) - ground truth signal events
        gt_noise_frames: (T, H, W) - ground truth noise events
    
    Returns:
        Dictionary of frame-level metrics
    """
    # Flatten frames
    pred_flat = predicted_frames.flatten()
    gt_signal_flat = gt_signal_frames.flatten()
    gt_noise_flat = gt_noise_frames.flatten()
    
    # Create evaluation mask (where events exist)
    eval_mask = (gt_signal_flat + gt_noise_flat) > 0
    
    # Compute metrics on masked regions
    if eval_mask.sum() == 0:
        return {
            'frame_accuracy': 0.0,
            'frame_precision': 0.0,
            'frame_recall': 0.0,
            'frame_f1': 0.0,
            'frame_da': 0.0
        }
    
    # Ground truth: 0=signal, 1=noise
    gt_flat = gt_noise_flat  # 1 where noise, 0 where signal
    pred_flat_binary = (pred_flat > 0.5).astype(np.uint8)
    
    # Invert: predicted_frames has 1 for signal, so we need to invert for noise
    pred_noise = 1 - pred_flat_binary
    
    metrics = compute_metrics(pred_noise, gt_flat, eval_mask)
    
    # Calculate SNR and ESNR for frames
    tp = metrics['tp']
    fp = metrics['fp']
    epsilon = 1e-10
    
    if fp + epsilon == 0:
        snr_db = float('inf') if tp > 0 else 0.0
    elif tp + epsilon == 0:
        snr_db = float('-inf')
    else:
        snr_db = 10 * math.log10((tp + epsilon) / (fp + epsilon))
        
    # ESNR
    if fp + epsilon == 0:
        esnr_db = float('inf') if tp > 0 else 0.0
    elif tp + epsilon == 0:
        esnr_db = float('-inf')
    else:
        esnr_db = 20 * math.log10((tp + epsilon) / (fp + epsilon))
    
    return {
        'frame_accuracy': metrics['accuracy'],
        'frame_precision': metrics['precision'],
        'frame_recall': metrics['recall'],
        'frame_f1': metrics['f1'],
        'frame_da': metrics['denoising_accuracy_da'],
        'frame_snr_db': snr_db,
        'frame_esnr_db': esnr_db,
        'frame_nrr': metrics['specificity'],  # Noise Rejection Rate = Specificity
        'frame_sr': metrics['recall'],        # Signal Retain = Recall
        'frame_nr': metrics['specificity'],   # Noise Removal = Specificity
        'frame_edp': metrics['precision'],    # Event Denoising Precision = Precision
        'frame_auc': metrics['auc'],
        'frame_tp': metrics['tp'],
        'frame_tn': metrics['tn'],
        'frame_fp': metrics['fp'],
        'frame_fn': metrics['fn']
    }


def compute_event_stream_metrics(events: np.ndarray,
                                 predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute event-stream level metrics directly from events.
    
    Args:
        events: (N, 5) array [label, x, y, t, polarity]
        predictions: (N,) array of binary predictions (0=signal, 1=noise)
    
    Returns:
        Dictionary of event-stream metrics including SNR and ESNR
    """
    if len(events) == 0:
        return {
            'stream_accuracy': 0.0,
            'stream_precision': 0.0,
            'stream_recall': 0.0,
            'stream_f1': 0.0,
            'stream_da': 0.0,
            'stream_snr_db': 0.0,
            'stream_esnr_db': 0.0
        }
    
    gt_labels = events[:, 0].astype(int)
    
    # Compute metrics
    metrics = compute_metrics(predictions, gt_labels)
    
    # Calculate SNR and ESNR
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    tn = metrics['tn']
    epsilon = 1e-10
    
    # SNR (TP/FP) in dB: 10 * log10(TP / FP)
    if fp + epsilon == 0:
        snr_db = float('inf') if tp > 0 else 0.0
    elif tp + epsilon == 0:
        snr_db = float('-inf')
    else:
        snr_db = 10 * math.log10((tp + epsilon) / (fp + epsilon))
    
    # ESNR (Event SNR) in dB: 20 * log10(TP / FP)
    if fp + epsilon == 0:
        esnr_db = float('inf') if tp > 0 else 0.0
    elif tp + epsilon == 0:
        esnr_db = float('-inf')
    else:
        esnr_db = 20 * math.log10((tp + epsilon) / (fp + epsilon))
    
    # Denoising Accuracy (DA) = 0.5 * (SR + NR)
    # SR (Signal Retain) = TP / (TP + FN) = Recall
    # NR (Noise Removal) = TN / (TN + FP) = Specificity
    total_signal = tp + fn
    total_noise = tn + fp
    sr = tp / (total_signal + epsilon) if total_signal > 0 else 0.0
    nr = tn / (total_noise + epsilon) if total_noise > 0 else 0.0
    da = 0.5 * (sr + nr)
    
    return {
        'stream_accuracy': metrics['accuracy'],
        'stream_precision': metrics['precision'],
        'stream_recall': metrics['recall'],
        'stream_f1': metrics['f1'],
        'stream_da': da,
        'stream_snr_db': snr_db,
        'stream_esnr_db': esnr_db,
        'stream_nrr': metrics['specificity'],  # Noise Rejection Rate
        'stream_sr': sr,                       # Signal Retain
        'stream_nr': nr,                       # Noise Removal
        'stream_edp': metrics['precision'],    # Event Denoising Precision
        'stream_tp': metrics['tp'],
        'stream_tn': metrics['tn'],
        'stream_fp': metrics['fp'],
        'stream_fn': metrics['fn']
    }


def format_hw_ops_summary(hw_ops: Dict[str, int], total_events: int) -> Dict[str, float]:
    """
    Format hardware operations summary with per-event statistics.
    
    Args:
        hw_ops: Dictionary of operation counts
        total_events: Total number of events processed
    
    Returns:
        Dictionary with formatted statistics
    """
    from config import cfg
    
    total_ops = sum(hw_ops[op] * cfg.HW_OP_COSTS[op] for op in hw_ops)
    
    return {
        'total_hw_ops': total_ops,
        'ops_per_event': total_ops / total_events if total_events > 0 else 0,
        'comparisons': hw_ops.get('comparison', 0),
        'additions': hw_ops.get('addition', 0),
        'multiplications': hw_ops.get('multiplication', 0),
        'divisions': hw_ops.get('division', 0),
        'sqrt_ops': hw_ops.get('sqrt', 0),
        'exp_ops': hw_ops.get('exp', 0),
        'memory_accesses': hw_ops.get('memory_access', 0),
    }
