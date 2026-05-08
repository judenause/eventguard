# train_engine.py
import torch
import torch.nn as nn # For checking nn.DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm # Progress bar
import numpy as np # For checking finite values in metric averaging, etc.
import math # For SNR calculations, etc.
from torch.utils.tensorboard import SummaryWriter

# Import necessary functions and objects from other modules
# from config import cfg # cfg object is passed directly as a function argument
from utils import compute_metrics, visualize_batch_results # focal_loss is passed directly from main_train

def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, # For type hinting
                    loss_function, # Receives the loss function directly (e.g., utils.focal_loss)
                    device: torch.device,
                    config_obj, # cfg instance from config.py
                    epoch_num: int,
                    visualization_save_dir: str,
                    writer: SummaryWriter = None,
                    scaler: torch.cuda.amp.GradScaler = None,
                    force_stateless: bool = False) -> dict:
    """
    Performs one epoch of training for the model.

    Args:
        model: PyTorch model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler.
        loss_function: Loss function.
        device: Device to use for training (CPU or CUDA).
        config_obj: Configuration object.
        epoch_num: Current epoch number.
        visualization_save_dir: Path to save batch visualization results.
        writer: TensorBoard SummaryWriter.
        scaler: GradScaler for AMP.
        force_stateless: Whether to force stateless mode (ignore membrane potential).

    Returns:
        dict: Dictionary containing average training loss and performance metrics for the epoch.
    """
    model.train()  # Set model to training mode
    epoch_total_loss = 0.0
    
    # Variables to accumulate TP, FP, TN, FN across the epoch (for more accurate epoch average metrics)
    epoch_total_tp = 0
    epoch_total_fp = 0
    epoch_total_tn = 0
    epoch_total_fn = 0
    
    # For accumulating other metrics (AUC, etc.)
    accumulated_metrics_sum = {} # e.g., {'auc': 0.0, 'snr_tp_fp': 0.0}
    num_valid_metric_batches = {} # Number of valid batches per metric (excluding NaN/inf)

    # Display batch progress using tqdm
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} [TRAIN]", leave=False, unit="batch",dynamic_ncols=True)

    # Stateful Training State
    current_mem = None
    
    # Initialize optimizer gradients before loop (for gradient accumulation)
    optimizer.zero_grad()

    for batch_idx, batch_data_tuple in enumerate(progress_bar):
        # batch_data_tuple: (inputs, real_gt, noise_gt, eval_mask, is_new_file)
        if len(batch_data_tuple) == 5:
             inputs, real_gt, _, eval_mask, is_new_file = batch_data_tuple
             is_new_file = is_new_file.to(device, non_blocking=True)
        else:
             # Fallback for old dataset version if needed (though we just updated it)
             inputs, real_gt, _, eval_mask = batch_data_tuple
             is_new_file = torch.zeros(inputs.size(0), device=device) # Assume no new file if missing
        
        inputs = inputs.to(device, non_blocking=True)
        real_gt = real_gt.to(device, non_blocking=True)
        # noise_gt_for_metrics = noise_gt.to(device, non_blocking=True) # Use if needed for metric calculation
        eval_mask = eval_mask.to(device, non_blocking=True)

        # Note: optimizer.zero_grad() is now called after optimizer.step() for gradient accumulation support

        # --- Stateful Logic ---
        # 1. If this is the first batch, initialize mem
        if current_mem is None:
             # model handles None -> zeros internally, but we need shape for masking
             # actually model.forward(..., mem=None) works. 
             pass

        # 2. Reset mem for samples that are starting a new file
        # current_mem: [Batch, Channels, Height, Width] (Approximation, it depends on neuron type)
        # QuantLeaky mem shape matches input shape usually: [B, C, H, W]
        # We need to apply mask `is_new_file` (shape [B]) to `current_mem`.
        # Also handle batch size change (e.g., last batch smaller)
        if current_mem is not None:
             if current_mem.size(0) != inputs.size(0):
                 # Batch size changed, reset membrane
                 current_mem = None
             else:
                 # is_new_file is 1.0 for new file, 0.0 otherwise.
                 # We want to keep mem where is_new_file is 0.0.
                 # Ensure masking is broadcastable.
                 # is_new_file shape: [B], mem shape: [B, C, H, W]
                 mask_shape = [-1] + [1] * (current_mem.dim() - 1)
                 reset_mask = (1.0 - is_new_file).view(mask_shape)
                 current_mem = current_mem * reset_mask

        # Forward pass
        with torch.amp.autocast('cuda', enabled=config_obj.USE_AMP):
            # If Stateless Mode, force memory reset (pass None)
            if force_stateless:
                current_mem = None
            
            # Pass current_mem and get updated mem
            logits, current_mem = model(inputs, mem=current_mem, regulate=True)  

            # Detach mem for TBPTT (Truncated Backpropagation Through Time)
            # We only backpropagate through the current window.
            if current_mem is not None:
                current_mem = current_mem.detach()

            # Calculate loss (e.g., utils.focal_loss)
            # Focal loss requires alpha/gamma, Tversky/FocalTversky only need 3 args
            if config_obj.LOSS_TYPE == 'Focal':
                loss = loss_function(logits, real_gt, eval_mask, config_obj.FOCAL_ALPHA, config_obj.FOCAL_GAMMA)
            else:
                # TverskyLoss, FocalTverskyLoss only need (logits, targets, mask)
                loss = loss_function(logits, real_gt, eval_mask)
            
            # Normalize loss for gradient accumulation
            accumulation_steps = getattr(config_obj, 'GRADIENT_ACCUMULATION_STEPS', 1)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

        # Backward pass (accumulate gradients)
        if scaler is not None and config_obj.USE_AMP:
            scaler.scale(loss).backward()
            
            # Only step optimizer every N batches (gradient accumulation)
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Gradient Clipping
                if config_obj.GRADIENT_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config_obj.GRADIENT_CLIP_NORM)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reset gradients after optimizer step
        else:
            loss.backward()

            # Only step optimizer every N batches (gradient accumulation)
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Gradient Clipping
                if config_obj.GRADIENT_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config_obj.GRADIENT_CLIP_NORM)

                optimizer.step()
                optimizer.zero_grad()  # Reset gradients after optimizer step

                # Step the learning rate scheduler (OneCycleLR steps per batch)
                # ReduceLROnPlateau is skipped here and stepped in main_train.py after validation
                if lr_scheduler and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step()

        epoch_total_loss += loss.item()

        # Calculate performance metrics (can be simplified or skipped during training for speed)
        # Calculate every 10 batches (speed optimization)
        if batch_idx % 10 == 0:
            with torch.no_grad(): # Gradient tracking not needed for metrics
                # compute_metrics also takes noise_gt as an argument
                noise_gt_for_metrics = batch_data_tuple[2].to(device, non_blocking=True)
                batch_metrics_dict = compute_metrics(logits.detach(), real_gt, noise_gt_for_metrics, eval_mask, config_obj)
            
            epoch_total_tp += batch_metrics_dict.get('tp', 0)
            epoch_total_fp += batch_metrics_dict.get('fp', 0)
            epoch_total_tn += batch_metrics_dict.get('tn', 0)
            epoch_total_fn += batch_metrics_dict.get('fn', 0)

            for key, value in batch_metrics_dict.items():
                if key not in ['tp', 'fp', 'tn', 'fn'] and isinstance(value, (int, float)) and np.isfinite(value):
                    accumulated_metrics_sum[key] = accumulated_metrics_sum.get(key, 0.0) + value
                    num_valid_metric_batches[key] = num_valid_metric_batches.get(key, 0) + 1
            
            # Update current batch loss and key metrics in the progress bar
            current_lr = optimizer.param_groups[0]['lr'] # Current LR changed by scheduler
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'F1': f"{batch_metrics_dict.get('f1', 0.0):.3f}",
                'SNR(dB)': f"{batch_metrics_dict.get('snr_tp_fp', float('nan')):.2f}",
                'LR': f"{current_lr:.2E}",
                'AUC' : f"{batch_metrics_dict.get('auc',float('nan')):.3f}",
                'DA' : f"{batch_metrics_dict.get('denoising_accuracy_da',float('nan')):.3f}"
            })
            
            # TensorBoard Logging (Batch Level) - Record every 10 batches
            if writer is not None and config_obj.USE_TENSORBOARD:
                global_step = (epoch_num - 1) * len(dataloader) + batch_idx
                writer.add_scalar('Train/Loss_Batch', loss.item(), global_step)
                writer.add_scalar('Train/F1_Batch', batch_metrics_dict.get('f1', 0.0), global_step)
                # Log Learnable Parameters (Beta, Threshold)
                target_model = model.module if isinstance(model, nn.DataParallel) else model
                if hasattr(target_model, 'snn_act'):
                    if hasattr(target_model.snn_act, 'beta') and isinstance(target_model.snn_act.beta, nn.Parameter):
                        writer.add_scalar('Param/Beta', target_model.snn_act.beta.item(), global_step)
                    if hasattr(target_model.snn_act, 'threshold') and isinstance(target_model.snn_act.threshold, nn.Parameter):
                        writer.add_scalar('Param/Threshold', target_model.snn_act.threshold.item(), global_step)
                
                # Log Learnable Loss Parameters (Alpha, Gamma)
                if hasattr(loss_function, 'alpha') and hasattr(loss_function, 'gamma'):
                    # Check if they are properties or attributes
                    alpha_val = loss_function.alpha if not callable(loss_function.alpha) else loss_function.alpha
                    gamma_val = loss_function.gamma if not callable(loss_function.gamma) else loss_function.gamma
                    
                    # If they are tensors (e.g. from property), get item()
                    if isinstance(alpha_val, torch.Tensor): alpha_val = alpha_val.item()
                    if isinstance(gamma_val, torch.Tensor): gamma_val = gamma_val.item()
                    
                    writer.add_scalar('Loss/Alpha', alpha_val, global_step)
                    writer.add_scalar('Loss/Gamma', gamma_val, global_step)
        else:
             # For batches where metrics are not calculated, only update loss
             current_lr = optimizer.param_groups[0]['lr']
             progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{current_lr:.2E}"
            })

        # (Optional) Visualize at specific batch intervals (e.g., first or middle batch)
        # if batch_idx == 0 or (batch_idx + 1) == len(dataloader) // 2 :
        # if batch_idx == 0 and epoch_num % 5 == 0 : # First batch, every 5 epochs
        #      if visualization_save_dir: # Only if a save directory is specified
        #         visualize_batch_results(batch_data_tuple, batch_metrics_dict, epoch_num, batch_idx, 'train', visualization_save_dir)

    # --- After epoch ends ---
    avg_epoch_loss = epoch_total_loss / len(dataloader)
    
    # Calculate epoch average metrics using accumulated TP, FP, TN, FN
    # (can be more accurate than averaging individual batch metrics)
    epsilon = 1e-10
    epoch_accuracy = (epoch_total_tp + epoch_total_tn) / (epoch_total_tp + epoch_total_fp + epoch_total_tn + epoch_total_fn + epsilon)
    epoch_precision = epoch_total_tp / (epoch_total_tp + epoch_total_fp + epsilon)
    epoch_recall = epoch_total_tp / (epoch_total_tp + epoch_total_fn + epsilon)
    epoch_f1 = 2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall + epsilon)
    
    final_epoch_metrics = {
        'loss': avg_epoch_loss,
        'accuracy': epoch_accuracy,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1,
        'tp': epoch_total_tp, # Also return totals
        'fp': epoch_total_fp,
        'tn': epoch_total_tn,
        'fn': epoch_total_fn
    }
    
    # Add other averaged metrics
    for key, total_sum in accumulated_metrics_sum.items():
        if key not in final_epoch_metrics: # tp, fp, etc. already handled above
            valid_batches = num_valid_metric_batches.get(key, 0)
            final_epoch_metrics[key] = total_sum / valid_batches if valid_batches > 0 else 0.0 # or float('nan')

    # Recalculate SNR (TP/FP) with accumulated TP, FP
    if config_obj.CALC_SNR_TP_FP:
        if epoch_total_fp + epsilon == 0:
            epoch_snr_tp_fp = float('inf') if epoch_total_tp > 0 else 0.0
        elif epoch_total_tp + epsilon == 0:
            epoch_snr_tp_fp = float('-inf')
        else:
            epoch_snr_tp_fp = 10 * math.log10((epoch_total_tp + epsilon) / (epoch_total_fp + epsilon))
        final_epoch_metrics['snr_tp_fp'] = epoch_snr_tp_fp
        
    return final_epoch_metrics


def validate_one_epoch(model: nn.Module,
                       dataloader: DataLoader,
                       loss_function, # Receives the loss function directly
                       device: torch.device,
                       config_obj, # cfg instance from config.py
                       epoch_num: int, # For log and visualization filenames (optional)
                       visualization_save_dir: str) -> dict:
    """
    Performs one epoch of validation for the model.

    Args:
        model: PyTorch model to evaluate.
        dataloader: Validation data loader.
        loss_function: Loss function.
        device: Device to use for evaluation.
        config_obj: Configuration object.
        epoch_num: Current epoch number.
        visualization_save_dir: Path to save batch visualization results.

    Returns:
        dict: Average validation loss and performance metrics for the epoch.
    """
    model.eval()  # Set model to evaluation mode
    
    # Pre-compute quantized threshold for inference (DAC2026-style)
    # Handle DDP/DataParallel wrapper - need to access model.module
    target_model = model.module if hasattr(model, 'module') else model
    if hasattr(target_model, 'prepare_for_inference'):
        target_model.prepare_for_inference(thr_bit=4)
    epoch_total_loss = 0.0
    
    epoch_total_tp = 0
    epoch_total_fp = 0
    epoch_total_tn = 0
    epoch_total_fn = 0
    accumulated_metrics_sum = {}
    num_valid_metric_batches = {}

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} [VALIDATE]", leave=False, unit="batch",dynamic_ncols=True)

    # Stateful Validation State (mirroring train_one_epoch)
    current_mem = None

    with torch.no_grad(): # Gradients not needed during validation
        for batch_idx, batch_data_tuple in enumerate(progress_bar):
            # Extract is_new_file for stateful membrane reset
            if len(batch_data_tuple) == 5:
                 inputs, real_gt, _, eval_mask, is_new_file = batch_data_tuple
                 is_new_file = is_new_file.to(device, non_blocking=True)
            else:
                 inputs, real_gt, _, eval_mask = batch_data_tuple
                 is_new_file = torch.zeros(inputs.size(0), device=device)
                 
            inputs = inputs.to(device, non_blocking=True)
            real_gt = real_gt.to(device, non_blocking=True)
            eval_mask = eval_mask.to(device, non_blocking=True)

            # Reset membrane for samples starting a new file
            # Also handle batch size change (e.g., last batch smaller)
            if current_mem is not None:
                if current_mem.size(0) != inputs.size(0):
                    # Batch size changed, reset membrane
                    current_mem = None
                else:
                    mask_shape = [-1] + [1] * (current_mem.dim() - 1)
                    reset_mask = (1.0 - is_new_file).view(mask_shape)
                    current_mem = current_mem * reset_mask
            
            with torch.amp.autocast('cuda', enabled=config_obj.USE_AMP):
                logits, current_mem = model(inputs, mem=current_mem, regulate=True)
            # Focal loss requires alpha/gamma, Tversky/FocalTversky only need 3 args
            if config_obj.LOSS_TYPE == 'Focal':
                loss = loss_function(logits, real_gt, eval_mask, config_obj.FOCAL_ALPHA, config_obj.FOCAL_GAMMA)
            else:
                loss = loss_function(logits, real_gt, eval_mask)
            epoch_total_loss += loss.item()

            noise_gt_for_metrics = batch_data_tuple[2].to(device, non_blocking=True)
            batch_metrics_dict = compute_metrics(logits.detach(), real_gt, noise_gt_for_metrics, eval_mask, config_obj)
            
            epoch_total_tp += batch_metrics_dict.get('tp', 0)
            epoch_total_fp += batch_metrics_dict.get('fp', 0)
            epoch_total_tn += batch_metrics_dict.get('tn', 0)
            epoch_total_fn += batch_metrics_dict.get('fn', 0)

            for key, value in batch_metrics_dict.items():
                if key not in ['tp', 'fp', 'tn', 'fn'] and isinstance(value, (int, float)) and np.isfinite(value):
                    accumulated_metrics_sum[key] = accumulated_metrics_sum.get(key, 0.0) + value
                    num_valid_metric_batches[key] = num_valid_metric_batches.get(key, 0) + 1

            # progress_bar.set_postfix({
            #     'Loss': f"{loss.item():.4f}",
            #     'F1': f"{batch_metrics_dict.get('f1', 0.0):.3f}",
            #     'SNR(dB)': f"{batch_metrics_dict.get('snr_tp_fp', float('nan')):.2f}"
            # })

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'F1': f"{batch_metrics_dict.get('f1', 0.0):.3f}",
                'SNR(dB)': f"{batch_metrics_dict.get('snr_tp_fp', float('nan')):.2f}",
                'AUC' : f"{batch_metrics_dict.get('auc',float('nan')):.3f}",
                'DA' : f"{batch_metrics_dict.get('denoising_accuracy_da',float('nan')):.3f}"
            })
            
            # (Optional) Visualize first batch of validation set
            # if batch_idx == 0 and epoch_num % 5 == 0 : # First batch, every 5 epochs
            #     if visualization_save_dir:
            #         visualize_batch_results(batch_data_tuple, batch_metrics_dict, epoch_num, batch_idx, 'val', visualization_save_dir)

    avg_epoch_loss = epoch_total_loss / len(dataloader)
    epsilon = 1e-10
    epoch_accuracy = (epoch_total_tp + epoch_total_tn) / (epoch_total_tp + epoch_total_fp + epoch_total_tn + epoch_total_fn + epsilon)
    epoch_precision = epoch_total_tp / (epoch_total_tp + epoch_total_fp + epsilon)
    epoch_recall = epoch_total_tp / (epoch_total_tp + epoch_total_fn + epsilon)
    epoch_f1 = 2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall + epsilon)
    
    final_epoch_metrics = {
        'loss': avg_epoch_loss,
        'accuracy': epoch_accuracy,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1,
        'tp': epoch_total_tp,
        'fp': epoch_total_fp,
        'tn': epoch_total_tn,
        'fn': epoch_total_fn
    }

    for key, total_sum in accumulated_metrics_sum.items():
        if key not in final_epoch_metrics:
            valid_batches = num_valid_metric_batches.get(key, 0)
            final_epoch_metrics[key] = total_sum / valid_batches if valid_batches > 0 else 0.0

    if config_obj.CALC_SNR_TP_FP:
        if epoch_total_fp + epsilon == 0:
            epoch_snr_tp_fp = float('inf') if epoch_total_tp > 0 else 0.0
        elif epoch_total_tp + epsilon == 0:
            epoch_snr_tp_fp = float('-inf')
        else:
            epoch_snr_tp_fp = 10 * math.log10((epoch_total_tp + epsilon) / (epoch_total_fp + epsilon))
        final_epoch_metrics['snr_tp_fp'] = epoch_snr_tp_fp
            
    return final_epoch_metrics