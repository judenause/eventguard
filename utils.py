import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import imageio
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm

matplotlib.use('Agg')

# --- 손실 함수 ---
def focal_loss(logits: torch.Tensor,
               targets: torch.Tensor,
               evaluation_mask: torch.Tensor,
               alpha: float,
               gamma: float,
               reduction: str = 'mean') -> torch.Tensor:
    """
    Focal Loss for class imbalance, based on BCEWithLogitsLoss.
    Assumes logits are for 2 classes [B, T, C=2, H, W] and targets are for positive class [B, T, H, W].

    Args:
        logits: Model output logits. Shape: [B, T, C, H, W].
        targets: Ground truth labels for the positive class (real_event_gt). Shape: [B, T, H, W].
        evaluation_mask: Mask indicating pixels to evaluate. Shape: [B, T, H, W].
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter.
        reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum'.

    Returns:
        Calculated focal loss.
    """
    if logits.shape[2] != 2:
        raise ValueError(f"Expected logits with 2 channels (C=2), but got {logits.shape[2]}")

    # Logits for the positive class (real event, assumed to be at index 1)
    logits_pos_class = logits[:, :, 1, :, :]  # Shape: [B, T, H, W]

    targets = targets.float()
    evaluation_mask = evaluation_mask.float()

    bce_loss = F.binary_cross_entropy_with_logits(logits_pos_class, targets, reduction='none')

    p = torch.sigmoid(logits_pos_class)
    # pt = p if targets == 1 else 1-p
    pt = torch.where(targets == 1, p, 1 - p)

    # alpha_factor = alpha if targets == 1 else 1-alpha
    alpha_factor = torch.where(targets == 1, alpha, 1 - alpha)

    # focal_weight = (1-pt)^gamma
    focal_weight = (1.0 - pt).pow(gamma)

    # Final focal loss: alpha * (1-pt)^gamma * bce_loss
    loss = alpha_factor * focal_weight * bce_loss

    # Apply evaluation mask
    masked_loss = loss * evaluation_mask

    if reduction == 'mean':
        # Mean loss over *only* the masked pixels
        if evaluation_mask.sum() > 0:
            return masked_loss.sum() / evaluation_mask.sum()
        else:
            # Avoid division by zero if mask is all zeros
            return torch.tensor(0.0, device=logits.device, requires_grad=True if logits.requires_grad else False)
    elif reduction == 'sum':
        return masked_loss.sum()
    elif reduction == 'none':
        return masked_loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    
class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation, with a focus on controlling FP/FN trade-off.
    Loss = 1 - TverskyIndex
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        """
        Args:
            alpha (float): Weight for False Positives (FP).
            beta (float): Weight for False Negatives (FN).
                         To penalize FP more (for higher SNR), set beta > alpha.
            smooth (float): A small value to prevent division by zero.
        """
        super().__init__()
        # SNR 향상을 위해서는 FP에 대한 페널티를 높여야 하므로, beta > alpha 로 설정합니다.
        # 예: alpha=0.3, beta=0.7
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor, 
                eval_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output logits. Shape: [B, T, C=2, H, W].
            targets: Ground truth for the positive class. Shape: [B, T, H, W].
            eval_mask: Mask for evaluation. Shape: [B, T, H, W].
        """
        # Logits for the positive class (assumed to be at index 1)
        logits_pos = logits[:, :, 1, :, :] # Shape: [B, T, H, W]
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits_pos)
        
        # Ensure targets and mask are float
        targets = targets.float()
        eval_mask = eval_mask.float()
        
        # Flatten all tensors and apply mask
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        mask_flat = eval_mask.view(-1)
        
        # Calculate TP, FP, FN only on masked pixels
        TP = (probs_flat * targets_flat * mask_flat).sum()
        FP = (probs_flat * (1 - targets_flat) * mask_flat).sum()
        FN = ((1 - probs_flat) * targets_flat * mask_flat).sum()
        
        # Calculate Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Return Tversky Loss
        return 1 - tversky_index

class FocalTverskyLoss(nn.Module):
    """
    A loss function that combines Tversky Index with a focusing parameter (gamma)
    to concentrate on hard-to-classify examples.
    Loss = (1 - TverskyIndex)^gamma
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0, smooth: float = 1e-6):
        """
        Args:
            alpha (float): Weight for False Positives (FP).
            beta (float): Weight for False Negatives (FN).
            gamma (float): Focusing parameter. Higher gamma focuses more on hard examples.
            smooth (float): A small value to prevent division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor, 
                eval_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output logits. Shape: [B, T, C=2, H, W].
            targets: Ground truth for the positive class. Shape: [B, T, H, W].
            eval_mask: Mask for evaluation. Shape: [B, T, H, W].
        """
        # Logits for the positive class (assumed to be at index 1)
        logits_pos = logits[:, :, 1, :, :] # Shape: [B, T, H, W]
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits_pos)
        
        # Ensure targets and mask are float
        targets = targets.float()
        eval_mask = eval_mask.float()
        
        # Flatten all tensors and apply mask
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        mask_flat = eval_mask.view(-1)
        
        # Calculate TP, FP, FN only on masked pixels
        TP = (probs_flat * targets_flat * mask_flat).sum()
        FP = (probs_flat * (1 - targets_flat) * mask_flat).sum()
        FN = ((1 - probs_flat) * targets_flat * mask_flat).sum()
        
        # Calculate Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Calculate Focal Tversky Loss
        focal_tversky_loss = torch.pow(1 - tversky_index, self.gamma)
        
        return focal_tversky_loss

        return focal_tversky_loss

class LearnableFocalLoss(nn.Module):
    """
    Focal Loss with learnable alpha and gamma parameters.
    """
    def __init__(self, init_alpha: float = 0.5, init_gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        # Initialize raw parameters (inverse sigmoid/softplus will be applied implicitly during training)
        # We start with values that map close to init_alpha and init_gamma
        
        # alpha = sigmoid(raw_alpha) -> raw_alpha = log(alpha / (1 - alpha))
        # Avoid 0 or 1 for stability
        init_alpha = max(0.01, min(0.99, init_alpha))
        raw_alpha_val = math.log(init_alpha / (1 - init_alpha))
        self.raw_alpha = nn.Parameter(torch.tensor(float(raw_alpha_val)))
        
        # gamma = softplus(raw_gamma) -> raw_gamma = inverse_softplus(gamma)
        # softplus(x) = log(1 + exp(x)) -> x = log(exp(gamma) - 1)
        # For stability, ensure init_gamma > 0
        init_gamma = max(0.1, init_gamma)
        raw_gamma_val = math.log(math.exp(init_gamma) - 1)
        self.raw_gamma = nn.Parameter(torch.tensor(float(raw_gamma_val)))
        
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, evaluation_mask: torch.Tensor, *args) -> torch.Tensor:
        """
        Args:
            logits: [B, T, C=2, H, W]
            targets: [B, T, H, W]
            evaluation_mask: [B, T, H, W]
            *args: Ignored (to be compatible with other loss signatures that take fixed alpha/gamma)
        """
        # Constrain parameters
        alpha = torch.sigmoid(self.raw_alpha)
        gamma = F.softplus(self.raw_gamma)
        
        # Use the functional focal_loss with the learnable parameters
        return focal_loss(logits, targets, evaluation_mask, alpha, gamma, self.reduction)

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha).item()

    @property
    def gamma(self):
        return F.softplus(self.raw_gamma).item()


# --- 성능 지표 계산 함수 ---
def compute_metrics(logits: torch.Tensor,
                    real_event_gt: torch.Tensor,
                    noise_event_gt: torch.Tensor, # 이 인자는 현재 SNR 계산 외에는 직접 사용되지 않음
                    evaluation_mask: torch.Tensor,
                    config_obj, # cfg 객체를 직접 전달받음
                    ) -> dict:
    """
    Compute various performance metrics for event denoising.
    Uses config_obj.EVALUATION_THRESHOLD and config_obj.CALC_SNR_TP_FP.

    Args:
        logits: Model output logits. Shape: [B, T, C=2, H, W].
        real_event_gt: Ground truth for real events. Shape: [B, T, H, W].
        noise_event_gt: Ground truth for noise events. Shape: [B, T, H, W].
        evaluation_mask: Mask indicating pixels to evaluate. Shape: [B, T, H, W].
        config_obj: The configuration object (cfg).

    Returns:
        A dictionary containing calculated metrics.
    """
    threshold = config_obj.EVALUATION_THRESHOLD
    epsilon = 1e-10  # For numerical stability

    device = logits.device
    real_event_gt = real_event_gt.to(device).float()
    noise_event_gt = noise_event_gt.to(device).float() # 사용되지 않더라도 타입 통일
    evaluation_mask = evaluation_mask.to(device).float()

    # Probabilities and predictions for the positive class (real event)
    probs = torch.sigmoid(logits[:, :, 1, :, :])  # Shape: [B, T, H, W]
    preds = (probs > threshold).float()           # Shape: [B, T, H, W]

    final_preds = preds * evaluation_mask.float()

    # Flatten tensors and apply mask
    evaluation_mask_flat = evaluation_mask.reshape(-1)
    active_indices = evaluation_mask_flat > 0

    

    if active_indices.sum() == 0:
        # print("Warning: No active pixels in evaluation mask for metric calculation.")
        metrics_dict = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0,
            'pred_classes_last_step': preds[:, -1, :, :].cpu().numpy(), # 마지막 스텝 예측 (시각화용)
            'pred_probs_last_step': probs[:, -1, :, :].cpu().numpy()    # 마지막 스텝 확률 (시각화용)
        }
        if config_obj.CALC_SNR_TP_FP:
            metrics_dict['snr_tp_fp'] = float('-inf')
        # CALC_SNR_EVAL_PCPNET 관련 로직은 원본 코드에 해당 계산 부분이 명확하지 않아 일단 제외
        return metrics_dict

    targets_flat_masked = real_event_gt.reshape(-1)[active_indices]
    preds_flat_masked = final_preds.reshape(-1)[active_indices]
    # preds_flat_masked = preds.reshape(-1)[active_indices]
    probs_flat_masked = probs.reshape(-1)[active_indices]
    noise_gt_flat_masked = noise_event_gt.reshape(-1)[active_indices] # 필요시 사용

    # tp = ((preds_flat_masked == 1) & (targets_flat_masked == 1)).sum().item() #real-real
    # #fp = ((preds_flat_masked == 1) & (targets_flat_masked == 0)).sum().item()
    # fp = ((preds_flat_masked == 1) & (noise_gt_flat_masked == 1)).sum().item() #real-noise
    # #tn = ((preds_flat_masked == 0) & (targets_flat_masked == 0)).sum().item()
    # tn = ((preds_flat_masked == 0) & (noise_gt_flat_masked == 1)).sum().item() #noise-noise
    # fn = ((preds_flat_masked == 0) & (targets_flat_masked == 1)).sum().item() #noise-real

        # --- ★★★ 우선순위를 적용한 새로운 TP, FP, FN, TN 계산 ★★★ ---

    # 2. TP (True Positive)를 가장 먼저 확정합니다.
    # 모델이 예측(1)했고, GT도 실제 이벤트(1)인 위치
    is_tp = (preds_flat_masked == 1) & (targets_flat_masked == 1)
    tp = is_tp.sum().item()

    # 3. FN (False Negative)을 확정합니다.
    # 모델이 예측 안했고(0), GT는 실제 이벤트(1)인 위치
    is_fn = (preds_flat_masked == 0) & (targets_flat_masked == 1)
    fn = is_fn.sum().item()

    # 4. FP (False Positive)를 계산합니다.
    # 모델이 예측(1)했는데, TP가 아니었던 위치 중에서 GT가 노이즈(1)인 경우
    is_fp = (preds_flat_masked == 1) & (~is_tp) & (noise_gt_flat_masked == 1)
    fp = is_fp.sum().item()

    # 5. TN (True Negative)을 계산합니다.
    # 모델이 예측 안했고(0), FN이 아니었던 위치 중에서 GT가 노이즈(1)인 경우
    is_tn = (preds_flat_masked == 0) & (~is_fn) & (noise_gt_flat_masked == 1)
    tn = is_tn.sum().item()
    
    # --- ★★★ 수정 완료 ★★★ ---

    total_active_pixels = tp + fp + tn + fn
    accuracy = (tp + tn) / total_active_pixels if total_active_pixels > 0 else 0.0
    precision = tp / (tp + fp + epsilon) # 분모 0 방지
    recall = tp / (tp + fn + epsilon)    # 분모 0 방지 (TPR)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon) # 분모 0 방지

    auc = 0.0
    if len(torch.unique(targets_flat_masked)) > 1: # AUC는 두 클래스 모두 존재해야 계산 가능
        try:
            auc = roc_auc_score(targets_flat_masked.cpu().numpy(), probs_flat_masked.cpu().numpy())
        except ValueError as e:
            # print(f"Warning: AUC calculation failed. {e}")
            auc = 0.0 # 또는 float('nan')
    # <<< [핵심 추가] DA (Denoising Accuracy) 계산 >>>
    # SR (Signal Retain)은 recall과 동일
    # NR (Noise Removal)은 tn / (tn + fp)
    signal_retain_sr = recall
    noise_removal_nr = tn / (tn + fp + epsilon)
    denoising_accuracy_da = 0.5 * (signal_retain_sr + noise_removal_nr)

    metrics_dict = {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc,
        # <<< [추가] DA와 그 구성요소를 딕셔너리에 포함 >>>
        'signal_retain_sr': signal_retain_sr,
        'noise_removal_nr': noise_removal_nr,
        'denoising_accuracy_da': denoising_accuracy_da,
        'pred_classes_last_step': final_preds[:, -1, :, :].cpu().numpy(),
        'pred_probs_last_step': probs[:, -1, :, :].cpu().numpy()
    }

    hallucinated_pixels = ((preds == 1) & (evaluation_mask == 0)).sum().item()

    # 기존 metrics_dict에 새로운 지표 추가
    metrics_dict['hallucinated_pixels'] = hallucinated_pixels

    if config_obj.CALC_SNR_TP_FP:
        if fp + epsilon == 0: # FP가 0일 때
            snr_tp_fp = float('inf') if tp > 0 else 0.0 # TP가 있으면 무한대, 없으면 0
        elif tp + epsilon == 0: # TP가 0일 때 (FP는 0이 아님)
            snr_tp_fp = float('-inf')
        else:
            snr_tp_fp = 10 * math.log10((tp + epsilon) / (fp + epsilon))
        metrics_dict['snr_tp_fp'] = snr_tp_fp

    # CALC_SNR_EVAL_PCPNET 관련 SNR 계산 로직은 원본 파일에서 해당 부분이 명확하지 않아 추가하지 않음.
    # 만약 snr_eval_pcpnet 계산 로직이 있다면 여기에 추가.

    return metrics_dict


# --- 시각화 함수들 ---

def visualize_batch_results(batch_data_tuple: tuple,
                            batch_metrics: dict,
                            epoch_num: int,
                            batch_idx: int,
                            phase: str, # 'train' or 'val'
                            save_dir_path: str,
                            sample_to_show_idx: int = 0,
                            time_step_to_show_idx: int = -1): # 마지막 시간 스텝
    """
    Visualize results for a single sample within a batch during training/validation.
    Saves the plot to a file.
    """
    # 시각화 결과 저장 폴더가 없으면 생성 (config.create_save_directories 에서 이미 생성했을 것)
    os.makedirs(save_dir_path, exist_ok=True)

    try:
        inputs_tensor, real_gt_tensor, noise_gt_tensor, eval_mask_tensor = batch_data_tuple

        # 선택된 샘플 및 시간 스텝 데이터 추출 (CPU로 옮기고 NumPy 배열로 변환)
        # inputs_tensor: [B, T, C, H, W] -> C=0 (첫번째 채널)
        input_frame = inputs_tensor[sample_to_show_idx, time_step_to_show_idx, 0].cpu().numpy()
        real_gt_frame = real_gt_tensor[sample_to_show_idx, time_step_to_show_idx].cpu().numpy()
        noise_gt_frame = noise_gt_tensor[sample_to_show_idx, time_step_to_show_idx].cpu().numpy()
        eval_mask_frame = eval_mask_tensor[sample_to_show_idx, time_step_to_show_idx].cpu().numpy()

        # batch_metrics에서 마지막 시간 스텝의 예측값 가져오기
        # pred_classes_last_step: [B, H, W]
        if batch_metrics['pred_classes_last_step'].shape[0] > sample_to_show_idx:
            pred_frame = batch_metrics['pred_classes_last_step'][sample_to_show_idx] # 이미 numpy 배열
        else: # 혹시 모를 인덱스 오류 방지
            pred_frame = np.zeros_like(real_gt_frame)


        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{phase.upper()} - Epoch {epoch_num} - Batch {batch_idx} (Sample {sample_to_show_idx}, Step {time_step_to_show_idx})', fontsize=14)

        # Plotting
        im = axes[0, 0].imshow(input_frame, cmap='binary'); axes[0, 0].set_title('Input (Any Event)'); fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
        im = axes[0, 1].imshow(real_gt_frame, cmap='binary'); axes[0, 1].set_title('Real Event GT'); fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        im = axes[0, 2].imshow(pred_frame, cmap='binary'); axes[0, 2].set_title('Prediction (Real Event)'); fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        im = axes[1, 0].imshow(noise_gt_frame, cmap='binary'); axes[1, 0].set_title('Noise Event GT'); fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        im = axes[1, 1].imshow(eval_mask_frame, cmap='binary'); axes[1, 1].set_title('Evaluation Mask'); fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Error map (TP: Blue, FP: Red, FN: Yellow, Masked Out: Gray)
        error_map_viz = np.zeros_like(pred_frame, dtype=int) # 0: Masked Out or TN
        active_pixels_mask = eval_mask_frame > 0
        tp_map = active_pixels_mask & (pred_frame == 1) & (real_gt_frame == 1)
        fp_map = active_pixels_mask & (pred_frame == 1) & (real_gt_frame == 0)
        fn_map = active_pixels_mask & (pred_frame == 0) & (real_gt_frame == 1)

        error_map_viz[tp_map] = 1  # TP
        error_map_viz[fp_map] = 2  # FP
        error_map_viz[fn_map] = 3  # FN

        # TN은 active_pixels_mask & (pred_frame == 0) & (real_gt_frame == 0) 이지만, 배경색(회색)으로 처리
        cmap_errors = matplotlib.colors.ListedColormap(['#AAAAAA', 'blue', 'red', 'yellow']) # 0:Gray, 1:TP(Blue), 2:FP(Red), 3:FN(Yellow)
        bounds_errors = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm_errors = matplotlib.colors.BoundaryNorm(bounds_errors, cmap_errors.N)

        im_err = axes[1, 2].imshow(error_map_viz, cmap=cmap_errors, norm=norm_errors)
        axes[1, 2].set_title('Errors (TP:Blu, FP:Red, FN:Yel)')
        cbar = fig.colorbar(im_err, ax=axes[1, 2], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['Other/TN', 'TP', 'FP', 'FN'])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle과의 간격 조절
        save_filename = os.path.join(save_dir_path, f'{phase}_epoch_{epoch_num}_batch_{batch_idx}_sample_{sample_to_show_idx}.png')
        plt.savefig(save_filename)
        plt.close(fig) # 메모리 해제

    except Exception as e:
        print(f"Error during batch visualization (epoch {epoch_num}, batch {batch_idx}): {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number): # 오류 발생 시에도 figure 닫기
            plt.close(fig)


def visualize_training_history(train_history_list: list,
                               val_history_list: list,
                               config_obj): # cfg 객체 전달 (SAVE_DIR 접근 위해)
    """
    Visualize training and validation history for various metrics.
    Saves the plot to a file.
    """
    save_path = os.path.join(config_obj.SAVE_DIR, 'visualizations', 'training_history.png')
    print(f"Visualizing training history and saving to {save_path}...")

    if not train_history_list:
        print("No training history data to visualize.")
        return

    epochs_range = range(1, len(train_history_list) + 1)

    # Plot할 메트릭 키 정의
    metric_keys_to_plot_base = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_keys_snr = []
    if config_obj.CALC_SNR_TP_FP:
        metric_keys_snr.append('snr_tp_fp')
    # if config_obj.CALC_SNR_EVAL_PCPNET: # 이 플래그가 True이면 snr_eval_pcpnet도 추가
    #     metric_keys_snr.append('snr_eval_pcpnet')

    all_metrics_to_plot = metric_keys_to_plot_base + metric_keys_snr
    num_metrics = len(all_metrics_to_plot)
    num_cols = 3
    num_rows = math.ceil(num_metrics / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)
    axes = axes.flatten() # 다루기 쉽도록 1D 배열로 변경

    fig.suptitle('Training and Validation History', fontsize=16)

    for i, key in enumerate(all_metrics_to_plot):
        if i >= len(axes): break # 서브플롯 개수 초과 방지

        # Train history 값 추출 (NaN/inf 아닌 유효한 값만)
        train_values = [epoch_metrics.get(key, float('nan')) for epoch_metrics in train_history_list]
        valid_train_epochs = [e for e, v in zip(epochs_range, train_values) if isinstance(v, (int, float)) and np.isfinite(v)]
        valid_train_values = [v for v in train_values if isinstance(v, (int, float)) and np.isfinite(v)]
        if valid_train_epochs:
            axes[i].plot(valid_train_epochs, valid_train_values, 'bo-', label=f'Train {key}')

        # Validation history 값 추출 (NaN/inf 아닌 유효한 값만)
        if val_history_list:
            val_values = [epoch_metrics.get(key, float('nan')) for epoch_metrics in val_history_list]
            valid_val_epochs = [e for e, v in zip(epochs_range, val_values) if isinstance(v, (int, float)) and np.isfinite(v)]
            valid_val_values = [v for v in val_values if isinstance(v, (int, float)) and np.isfinite(v)]
            if valid_val_epochs:
                axes[i].plot(valid_val_epochs, valid_val_values, 'ro-', label=f'Validation {key}')

        axes[i].set_title(key.replace('_', ' ').title())
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(key.replace('_', ' ').title())
        axes[i].legend()
        axes[i].grid(True)

    # 남는 빈 서브플롯 숨기기
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Training history plot saved to {save_path}")


def visualize_test_set_results(aggregated_final_metrics: dict,
                               config_obj, # cfg 객체 (SAVE_DIR 접근)
                               all_pred_probs_numpy: np.ndarray = None, # (선택) 전체 테스트셋 예측 확률
                               all_targets_numpy: np.ndarray = None):   # (선택) 전체 테스트셋 실제 값
    """
    Visualize final test results: Confusion Matrix, and optionally prediction probability distribution.
    Saves plots and a summary text file.
    """
    save_dir = os.path.join(config_obj.SAVE_DIR, 'visualizations')
    print(f"Visualizing test results and saving to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True) # 이미 생성되었겠지만, 확인차

    try:
        # 1. Confusion Matrix (from aggregated TP, FP, TN, FN)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        tp = aggregated_final_metrics.get('tp', 0)
        fp = aggregated_final_metrics.get('fp', 0)
        tn = aggregated_final_metrics.get('tn', 0)
        fn = aggregated_final_metrics.get('fn', 0)

        cm_array = np.array([[tn, fp], [fn, tp]])
        total_pixels_in_cm = cm_array.sum()

        if total_pixels_in_cm > 0:
            cm_percent = cm_array / total_pixels_in_cm * 100
            sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', ax=ax_cm,
                        xticklabels=['Predicted BG/Noise', 'Predicted Real Event'],
                        yticklabels=['Actual BG/Noise', 'Actual Real Event'])
            ax_cm.set_title(f'Test Confusion Matrix (%)\nTotal Masked Pixels: {total_pixels_in_cm}')
        else:
            ax_cm.set_title('Test Confusion Matrix (No Data)')
            ax_cm.text(0.5, 0.5, 'No data for CM', horizontalalignment='center', verticalalignment='center')

        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        plt.tight_layout()
        cm_save_path = os.path.join(save_dir, 'test_confusion_matrix.png')
        plt.savefig(cm_save_path)
        plt.close(fig_cm)
        print(f"Test confusion matrix saved to {cm_save_path}")

            # <<< [신규] ROC 커브 및 AUC 계산 및 시각화 >>>
        if all_pred_probs_numpy is not None and all_targets_numpy is not None and len(all_targets_numpy) > 0:
            # ROC 커브 계산
            fpr, tpr, thresholds = roc_curve(all_targets_numpy, all_pred_probs_numpy)
            roc_auc = auc(fpr, tpr)
            
            # 집계된 메트릭에 AUC 추가
            if aggregated_final_metrics: # 프레임 레벨 메트릭이 있는 경우
                aggregated_final_metrics['overall_auc'] = roc_auc

            # ROC 커브 시각화
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True)
            roc_save_path = os.path.join(save_dir, 'test_roc_curve.png')
            plt.savefig(roc_save_path)
            plt.close(fig_roc)
            print(f"Test ROC curve saved to {roc_save_path}")

        # 2. (Optional) Prediction Probability Distribution (if data provided)
        if all_pred_probs_numpy is not None and all_targets_numpy is not None: # 타겟도 함께 있어야 의미 해석 가능
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            # 실제 이벤트(positive class)로 예측된 확률만 필터링
            probs_for_positive_class_actual_positive = all_pred_probs_numpy[all_targets_numpy == 1]
            probs_for_positive_class_actual_negative = all_pred_probs_numpy[all_targets_numpy == 0]

            if len(probs_for_positive_class_actual_positive) > 0 :
                sns.histplot(probs_for_positive_class_actual_positive, bins=50, ax=ax_hist,
                             label='Prob (Actual Positive)', color='green', kde=True, stat="density")
            if len(probs_for_positive_class_actual_negative) > 0:
                sns.histplot(probs_for_positive_class_actual_negative, bins=50, ax=ax_hist,
                             label='Prob (Actual Negative)', color='red', kde=True, stat="density")

            ax_hist.set_title('Test Prediction Probabilities for Real Event Class')
            ax_hist.set_xlabel('Predicted Probability of Real Event')
            ax_hist.set_ylabel('Density')
            ax_hist.legend()
            ax_hist.grid(True)
            hist_save_path = os.path.join(save_dir, 'test_probability_distribution.png')
            plt.savefig(hist_save_path)
            plt.close(fig_hist)
            print(f"Test probability distribution plot saved to {hist_save_path}")

        # 3. Save summary text file with all metrics
        summary_text_path = os.path.join(config_obj.SAVE_DIR, 'test_results_summary.txt') # SAVE_DIR 최상위에 저장
        with open(summary_text_path, 'w') as f:
            f.write(f"Test Results Summary (Aggregated/Averaged over masked pixels)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Masked Pixels Evaluated (Sum for CM): {total_pixels_in_cm}\n\n")
            metrics_to_report = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'snr_tp_fp', 'tp', 'fp', 'tn', 'fn']
            # CALC_SNR_EVAL_PCPNET 관련 SNR도 있다면 추가
            # if config_obj.CALC_SNR_EVAL_PCPNET: metrics_to_report.append('snr_eval_pcpnet')

            for key in metrics_to_report:
                if key in aggregated_final_metrics:
                    value = aggregated_final_metrics[key]
                    is_count_metric = key in ['tp', 'fp', 'tn', 'fn']
                    metric_name_display = key.replace('_', ' ').title()
                    if is_count_metric:
                        f.write(f"{metric_name_display:<20}: {int(value)}\n")
                    else:
                        f.write(f"{metric_name_display:<20}: {value:.4f}\n")
        print(f"Test results summary saved to {summary_text_path}")

    except Exception as e:
        print(f"Error visualizing test results: {e}")
        if 'fig_cm' in locals() and plt.fignum_exists(fig_cm.number): plt.close(fig_cm)
        if 'fig_hist' in locals() and plt.fignum_exists(fig_hist.number): plt.close(fig_hist)


def create_evaluation_gif(input_frames_seq: np.ndarray,      # [T, H, W]
                            real_gt_frames_seq: np.ndarray,    # [T, H, W]
                            noise_gt_frames_seq: np.ndarray,   # [T, H, W]
                            predicted_frames_seq: np.ndarray,  # [T, H, W] (binary predictions)
                            eval_mask_frames_seq: np.ndarray,  # [T, H, W]
                            config_obj, # cfg 객체 (SAVE_DIR, FPS, FRAME_WIDTH/HEIGHT)
                            output_gif_filename_base: str): # 예: "test_file_X_result"
    """
    Create a GIF visualizing the input, GTs, and predictions over time for a single sequence.
    Saves the GIF to the visualizations folder.
    """
    vis_save_dir = os.path.join(config_obj.SAVE_DIR, 'visualizations')
    os.makedirs(vis_save_dir, exist_ok=True)
    output_gif_path = os.path.join(vis_save_dir, f'{output_gif_filename_base}.gif')

    num_total_frames = input_frames_seq.shape[0]
    if not (num_total_frames == real_gt_frames_seq.shape[0] == \
            noise_gt_frames_seq.shape[0] == predicted_frames_seq.shape[0] == \
            eval_mask_frames_seq.shape[0]):
        print(f"Error creating GIF {output_gif_filename_base}: Frame counts mismatch.")
        return
    if num_total_frames == 0:
        print(f"Warning for GIF {output_gif_filename_base}: No frames to process.")
        return

    print(f"Creating GIF for {output_gif_filename_base} ({num_total_frames} frames) -> {output_gif_path}...")

    # GIF 프레임 속도 및 크기 설정
    gif_display_fps = max(1, int(config_obj.FPS / 4)) # 원본 FPS의 1/4 또는 최소 1 FPS
    target_display_width = 320 # GIF 내 각 프레임의 목표 너비
    scale_factor = min(1.0, target_display_width / config_obj.FRAME_WIDTH) if config_obj.FRAME_WIDTH > 0 else 1.0
    display_w = int(config_obj.FRAME_WIDTH * scale_factor)
    display_h = int(config_obj.FRAME_HEIGHT * scale_factor)

    gif_frames_for_output = []

    for frame_idx in tqdm(range(num_total_frames), desc=f"  GIF:{output_gif_filename_base}", leave=False, unit="frame"):
        current_input = input_frames_seq[frame_idx]
        current_real_gt = real_gt_frames_seq[frame_idx]
        current_noise_gt = noise_gt_frames_seq[frame_idx]
        current_pred = predicted_frames_seq[frame_idx]
        current_eval_mask = eval_mask_frames_seq[frame_idx]

        # 시각화를 위해 각 프레임을 [0, 255] 범위의 uint8 BGR 이미지로 변환
        def format_frame_for_gif(frame_data, mask_data, colormap=cv2.COLORMAP_VIRIDIS, is_input_type=False):
            masked_frame = frame_data * mask_data # 평가 마스크 적용
            # Normalize to 0-1 if not already binary
            norm_frame = masked_frame.astype(np.float32)
            min_val, max_val = norm_frame.min(), norm_frame.max()
            if max_val > min_val:
                norm_frame = (norm_frame - min_val) / (max_val - min_val)
            else: # 모든 값이 같거나 마스크로 인해 0인 경우
                norm_frame = np.zeros_like(norm_frame)

            uint8_frame = (norm_frame * 255).astype(np.uint8)

            if is_input_type: # 입력은 그레이스케일 -> BGR
                colored_frame = cv2.cvtColor(uint8_frame, cv2.COLOR_GRAY2BGR)
            else: # 나머지는 컬러맵 적용
                colored_frame = cv2.applyColorMap(uint8_frame, colormap)

            return cv2.resize(colored_frame, (display_w, display_h), interpolation=cv2.INTER_NEAREST)

        vis_input_frame = format_frame_for_gif(current_input, current_eval_mask, is_input_type=True)
        vis_real_gt_frame = format_frame_for_gif(current_real_gt, current_eval_mask, colormap=cv2.COLORMAP_COOL) # 파란 계열
        vis_noise_gt_frame = format_frame_for_gif(current_noise_gt, current_eval_mask, colormap=cv2.COLORMAP_AUTUMN) # 주황/빨강 계열
        vis_pred_frame = format_frame_for_gif(current_pred, current_eval_mask, colormap=cv2.COLORMAP_HOT) # 밝은 노랑/빨강 계열

        # 4개 프레임을 가로로 연결
        combined_image = np.hstack((vis_input_frame, vis_real_gt_frame, vis_noise_gt_frame, vis_pred_frame))

        # 프레임 번호 텍스트 추가
        cv2.putText(combined_image, f"Frame: {frame_idx + 1}/{num_total_frames}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        gif_frames_for_output.append(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)) # imageio는 RGB 순서

    if gif_frames_for_output:
        imageio.mimsave(output_gif_path, gif_frames_for_output, fps=gif_display_fps)
        print(f"GIF saved: {output_gif_path}")
    else:
        print(f"No frames were generated for GIF: {output_gif_filename_base}")


def visualize_first_data_samples(set_name_str: str,
                                 processed_data_list: list,
                                 config_obj): # cfg 객체 (SAVE_DIR 접근)
    """
    Visualize sample frames from the first file in a processed data list (train, val, or test).
    Saves the plot to a file. Used for initial data check.
    """
    vis_save_dir = os.path.join(config_obj.SAVE_DIR, 'visualizations')
    print(f"\nVisualizing example frames for the first file in {set_name_str} set...")
    if not processed_data_list:
        print(f"  - No data to visualize for {set_name_str} set.")
        return

    # 첫 번째 파일 데이터 가져오기
    first_file_dict = processed_data_list[0]
    input_f_seq = first_file_dict['input_frames']       # [T, H, W]
    real_gt_f_seq = first_file_dict['real_event_gt']    # [T, H, W]
    noise_gt_f_seq = first_file_dict['noise_event_gt']  # [T, H, W]
    eval_mask_f_seq = first_file_dict['evaluation_mask']# [T, H, W]
    source_file_path = first_file_dict['file_path']
    num_frames_in_file = input_f_seq.shape[0]

    print(f"  - Visualizing samples from: {os.path.basename(source_file_path)}")

    if num_frames_in_file == 0:
        print(f"  - No frames to visualize in this file.")
        return

    # 보여줄 프레임 인덱스: 첫 프레임, 중간 프레임, 마지막 프레임
    indices_to_display = sorted(list(set([0, num_frames_in_file // 2, num_frames_in_file - 1])))
    print(f"  - Showing frames at indices: {indices_to_display}")

    for frame_idx_to_show in indices_to_display:
        if frame_idx_to_show >= num_frames_in_file: continue # 유효 인덱스 확인

        fig, axes = plt.subplots(1, 4, figsize=(20, 5)) # 1행 4열
        fig.suptitle(f'{set_name_str} Set (File: {os.path.basename(source_file_path)}) - Frame Index: {frame_idx_to_show}', fontsize=16)

        im0 = axes[0].imshow(input_f_seq[frame_idx_to_show], cmap='binary'); axes[0].set_title('Input Frame (All Events)'); fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(real_gt_f_seq[frame_idx_to_show], cmap='binary'); axes[1].set_title('Real Event GT'); fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(noise_gt_f_seq[frame_idx_to_show], cmap='binary'); axes[2].set_title('Noise Event GT'); fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        im3 = axes[3].imshow(eval_mask_f_seq[frame_idx_to_show], cmap='binary'); axes[3].set_title('Evaluation Mask'); fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        for ax_item in axes: ax_item.axis('off') # 축 정보 숨기기

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_filename = os.path.join(vis_save_dir, f'{set_name_str}_first_file_sample_frame_{frame_idx_to_show}.png')
        plt.savefig(save_filename)
        plt.close(fig)
        print(f"    - Saved sample frame plot to {save_filename}")


# ===================================================================
# 프로젝트 실행 환경 설정을 위한 헬퍼 함수들 (config.py에서 이동)
# ===================================================================

def setup_device_and_batch_size(config_instance):
    """Config 객체를 받아 DEVICE 및 BATCH_SIZE를 설정합니다."""
    physical_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if physical_gpu_count > 0:
        if config_instance.USE_MULTI_GPU and physical_gpu_count > 1:
            # --- DDP Device Setup ---
            if "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])
                config_instance.DEVICE = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(local_rank) # IMPORTANT
                config_instance.N_GPU_EFFECTIVE = physical_gpu_count
            else:
                # Fallback for non-distributed multi-gpu (DataParallel?) - Not recommended
                print("⚠️ Warning: USE_MULTI_GPU is True but LOCAL_RANK not found. Defaulting to cuda:0")
                config_instance.DEVICE = torch.device("cuda:0")
                config_instance.N_GPU_EFFECTIVE = physical_gpu_count
            
            config_instance.BATCH_SIZE = config_instance.BASE_BATCH_SIZE * config_instance.N_GPU_EFFECTIVE
        else:
            target_gpu_id = config_instance.SPECIFIC_GPU_ID
            if not isinstance(target_gpu_id, int) or target_gpu_id >= physical_gpu_count or target_gpu_id < 0:
                target_gpu_id = 0
            config_instance.DEVICE = torch.device(f"cuda:{target_gpu_id}")
            if torch.cuda.is_available():
                torch.cuda.set_device(target_gpu_id) # Set device for single GPU too

            config_instance.N_GPU_EFFECTIVE = 1
            config_instance.BATCH_SIZE = config_instance.BASE_BATCH_SIZE
    else:
        config_instance.DEVICE = torch.device("cpu")
        config_instance.N_GPU_EFFECTIVE = 0
        config_instance.BATCH_SIZE = config_instance.BASE_BATCH_SIZE
    print(f"✅ Effective Device set to: {config_instance.DEVICE} | Effective Batch size: {config_instance.BATCH_SIZE}")

def set_seed_all(seed_value: int):
    """모든 랜덤 시드를 고정하여 재현성을 확보합니다."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to: {seed_value}")

def create_save_directories(config_instance):
    """결과 저장을 위한 디렉토리를 생성합니다."""
    base_save_dir = config_instance.SAVE_DIR
    vis_save_dir = os.path.join(base_save_dir, 'visualizations')
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)
    print(f"✅ Results will be saved in: {base_save_dir}")



# utils.py 파일에 추가 (기존에 있었다면 아래 코드로 대체)
import pandas as pd
import numpy as np

def save_metrics_to_csv(filename: str, 
                        results_dir: str, 
                        aggregated_metrics: dict, 
                        per_file_metrics_list: list[dict], 
                        summary_title: str):
    """
    요약(Aggregated)과 상세(Per-File) 내역을 하나의 CSV 파일로 지능적으로 저장합니다.
    전달된 딕셔너리의 모든 키를 동적으로 처리합니다.
    """
    # 유효한 상세 메트릭 데이터만 필터링
    valid_per_file_metrics = [m for m in per_file_metrics_list if m and 'error' not in m]
    if not valid_per_file_metrics and not aggregated_metrics:
        print(f"No valid metrics to save for {filename}.")
        return

    csv_path = os.path.join(results_dir, filename)

    try:
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            # --- 1. 요약(Aggregated) 정보 동적 저장 ---
            f.write(f'--- {summary_title} ---\n')
            f.write('Metric,Value\n') # CSV 헤더
            if aggregated_metrics:
                for key, value in aggregated_metrics.items():
                    # 보기 좋게 key 이름 변환 (e.g., 'overall_f1' -> 'Overall F1')
                    metric_name = key.replace('_', ' ').title() 
                    val_str = f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
                    f.write(f'"{metric_name}","{val_str}"\n')
            
            # --- 2. 상세(Per-File) 정보 동적 저장 ---
            if valid_per_file_metrics:
                f.write('\n--- Detailed Per-File Metrics ---\n')
                # 리스트의 딕셔너리로부터 자동으로 DataFrame 생성
                df = pd.DataFrame(valid_per_file_metrics)
                
                # 파일 경로에서 기본 이름만 남겨서 가독성 향상
                if 'file_path' in df.columns:
                    df['file_name'] = df['file_path'].apply(os.path.basename)
                    # file_name을 첫 번째 열로 재배치
                    cols = ['file_name'] + [col for col in df.columns if col != 'file_name' and col != 'file_path']
                    df = df[cols]

                df.to_csv(f, index=False)

        print(f"✅ Unified metrics report saved to: {csv_path}")

    except Exception as e:
        print(f"❌ Error saving unified metrics to CSV: {e}")