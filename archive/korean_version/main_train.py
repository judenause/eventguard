import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import pandas as pd
from collections import OrderedDict

# --- 프로젝트 모듈 임포트 ---
from config import cfg
from utils import save_metrics_to_csv,setup_device_and_batch_size, set_seed_all, create_save_directories, visualize_training_history, TverskyLoss, focal_loss, FocalTverskyLoss
from data_processing import process_folder_to_frame_lists
from dataset import EventFrameLazyDataset
from model import Hybrid_SNN_Pure_BNN
from train_engine import train_one_epoch, validate_one_epoch
from evaluation_engine import evaluate_model_on_dataset

# def save_metrics_to_csv(filename, results_dir, aggregated_metrics, per_file_metrics_list, summary_title):
#     """요약과 상세 내역을 포함하여 CSV 파일로 저장하는 유틸리티 함수"""
#     valid_metrics_list = [m for m in per_file_metrics_list if m and 'error' not in m]
#     if not valid_metrics_list:
#         print(f"No valid per-file metrics to save for {filename}.")
#         return

#     csv_path = os.path.join(results_dir, filename)
    
#     # --- 1. 요약 정보 문자열 생성 ---
#     summary_lines = [f"--- {summary_title} ---"]
#     for key, value in aggregated_metrics.items():
#         val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
#         summary_lines.append(f'"{key.replace("_", " ").title()}","{val_str}"')
    
#     summary_lines.append("\n--- Detailed Per-File Metrics ---")
#     summary_header = "\n".join(summary_lines) + "\n"

#     # --- 2. CSV 파일 저장 ---
#     try:
#         df = pd.DataFrame(valid_metrics_list)
#         with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
#             f.write(summary_header) # 요약 헤더 먼저 쓰기
#             df.to_csv(f, index=False) # 그 아래에 상세 데이터프레임 추가
#         print(f"✅ Metrics report saved to: {csv_path}")
#     except Exception as e:
#         print(f"❌ Error saving metrics to CSV: {e}")

def load_model_for_evaluation(model_path: str, config: cfg, device: torch.device) -> nn.Module:
    # ... (내용 변경 없음)
    if not os.path.exists(model_path): 
        print(f"❌ Error: Model file not found at {model_path}")
        return None
        
    snn_params = {'beta': config.SNN_BETA, 'threshold': config.SNN_THRESHOLD}
    model = Hybrid_SNN_Pure_BNN(snn_params=snn_params, output_classes=config.OUTPUT_CLASSES, input_channels=config.INPUT_CHANNELS).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle DataParallel 'module.' prefix
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
            
        # Handle torch.compile '_orig_mod.' prefix
        # If the model was compiled, keys might start with '_orig_mod.'
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model state_dict: {e}")
        return None

def run_single_evaluation(model, test_data_folder, csv_name_prefix, config, device):
    """지정된 단일 데이터셋에 대해 평가를 수행하고 결과를 저장하는 함수"""
    print(f"\n\n--- Evaluating on dataset: {test_data_folder} ---")
    
    if not os.path.exists(test_data_folder):
        print(f"⚠️  Warning: Test data folder not found, skipping evaluation: {test_data_folder}")
        return

    test_data_list = process_folder_to_frame_lists(test_data_folder, config.DATA_FILE_PATTERN, f"Eval on {os.path.basename(test_data_folder)}", config)
    if not test_data_list:
        print(f"️️️⚠️  Warning: No data found in {test_data_folder}, skipping.")
        return

    model.eval()
    
    # Pre-compute quantized threshold for inference (DAC2026-style)
    # Handle DDP/DataParallel wrapper - need to access model.module
    target_model = model.module if hasattr(model, 'module') else model
    if hasattr(target_model, 'prepare_for_inference'):
        target_model.prepare_for_inference(thr_bit=4)

    # 모델 평가 실행
    (aggregated_frame_metrics, per_file_frame_metrics, 
     _, _, _,_, 
     per_file_stream_metrics, aggregated_stream_metrics) = evaluate_model_on_dataset(model, test_data_list, config, device)

    # Frame-Level Metrics 처리 및 저장
    if aggregated_frame_metrics:
        print("\n### Aggregated Frame-Level Metrics ###")
        for key, value in aggregated_frame_metrics.items():
            print(f"  {key:<30}: {value:.4f}" if isinstance(value, float) else f"  {key:<30}: {value}")
        save_metrics_to_csv(
            filename=f"{csv_name_prefix}_frame_metrics.csv",
            results_dir=config.SAVE_DIR,
            aggregated_metrics=aggregated_frame_metrics,
            per_file_metrics_list=per_file_frame_metrics,
            summary_title=f"Frame-Level Metrics for {os.path.basename(test_data_folder)}"
        )

    # Event-Stream Level Metrics 처리 및 저장
    if aggregated_stream_metrics:
        print("\n### Aggregated Event-Stream Level Metrics ###")
        for key, value in aggregated_stream_metrics.items():
            print(f"  {key:<30}: {value:.4f}" if isinstance(value, float) else f"  {key:<30}: {value}")
        save_metrics_to_csv(
            filename=f"{csv_name_prefix}_stream_metrics.csv",
            results_dir=config.SAVE_DIR,
            aggregated_metrics=aggregated_stream_metrics,
            per_file_metrics_list=per_file_stream_metrics,
            summary_title=f"Event-Stream Metrics for {os.path.basename(test_data_folder)}"
        )

def run_multiple_evaluations(config: cfg, model_filename: str, model=None):
    """학습 완료 후, 지정된 여러 테스트셋에 대해 최종 평가를 수행하는 함수."""
    print("\n\n" + "="*50 + "\n=== STARTING FINAL EVALUATION PHASE ===\n" + "="*50)
    
    if model is None:
        model_path = os.path.join(config.SAVE_DIR, model_filename)
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at {model_path}. Aborting evaluation.")
            return

        # 모델을 한 번만 로드합니다.
        model = load_model_for_evaluation(model_path, config, config.DEVICE)
        
    if model is None:
        print(f"❌ Error: Model is None. Aborting evaluation.")
        return

    # --- ⚙️ 여기에 평가할 작업들을 정의합니다 ---
    evaluation_tasks = [
        {
            "test_data_folder": config.TEST_DATA_FOLDER,
            "csv_name_prefix": config.CSV_NAME  # 예: 'exp1_results'
        },
        # 필요하다면 두 번째, 세 번째 작업을 계속 추가할 수 있습니다.
        # {
        #     "test_data_folder": config.TEST_DATA_FOLDER_2, # 두 번째 테스트셋 경로
        #     "csv_name_prefix": config.CSV_NAME_2 # 두 번째 결과 파일명
        # },
    ]

    # 정의된 작업들을 순회하며 평가 실행
    for task in evaluation_tasks:
        run_single_evaluation(
            model=model,
            test_data_folder=task["test_data_folder"],
            csv_name_prefix=task["csv_name_prefix"],
            config=config,
            device=config.DEVICE
        )

    print("\n\n--- All Final Evaluations Finished ---")



def run_evaluation_after_training(config: cfg, model_filename: str):
    """학습 완료 후, 저장된 모델로 최종 평가를 수행하는 함수."""
    print("\n\n" + "="*50 + "\n=== STARTING FINAL EVALUATION PHASE ===\n" + "="*50)
    
    model_path = os.path.join(config.SAVE_DIR, model_filename)
    if not os.path.exists(model_path): return

    test_data_list = process_folder_to_frame_lists(config.TEST_DATA_FOLDER, config.DATA_FILE_PATTERN, "Test (Final Eval)", config)
    if not test_data_list: return

    model = load_model_for_evaluation(model_path, config, config.DEVICE)
    if model is None: return
    model.eval()
    
    # Pre-compute quantized threshold for inference (DAC2026-style)
    # Handle DDP/DataParallel wrapper - need to access model.module
    target_model = model.module if hasattr(model, 'module') else model
    if hasattr(target_model, 'prepare_for_inference'):
        target_model.prepare_for_inference(thr_bit=4)

    print("\nRunning evaluation on the test dataset...")
    
    (aggregated_frame_metrics, per_file_frame_metrics, 
     _, _, _, 
     per_file_stream_metrics, aggregated_stream_metrics) = evaluate_model_on_dataset(model, test_data_list, config, config.DEVICE)

    # --- Frame-Level Metrics 처리 ---
    if aggregated_frame_metrics:
        print("\n### Aggregated Frame-Level Metrics ###")
        for key, value in aggregated_frame_metrics.items():
            print(f"  {key:<30}: {value:.4f}" if isinstance(value, float) else f"  {key:<30}: {value}")
        save_metrics_to_csv(
            filename=f"{cfg.CSV_NAME}_frame_metrics.csv",
            results_dir=config.SAVE_DIR,
            aggregated_metrics=aggregated_frame_metrics,
            per_file_metrics_list=per_file_frame_metrics,
            summary_title="Overall Aggregated Frame-Level Metrics"
        )

    # --- Event-Stream Level Metrics 처리 ---
    if aggregated_stream_metrics:
        print("\n### Aggregated Event-Stream Level Metrics ###")
        for key, value in aggregated_stream_metrics.items():
            print(f"  {key:<30}: {value:.4f}" if isinstance(value, float) else f"  {key:<30}: {value}")
        save_metrics_to_csv(
            filename=f"{cfg.CSV_NAME}_stream_metrics.csv",
            results_dir=config.SAVE_DIR,
            aggregated_metrics=aggregated_stream_metrics,
            per_file_metrics_list=per_file_stream_metrics,
            summary_title="Overall Aggregated Event-Stream Level Metrics"
        )

    print("\n--- Final Evaluation Finished ---")

def main_training_pipeline(args):
    """전체 학습 및 평가 파이프라인."""
    start_time = time.time()
    
    # --- DDP Initialization ---
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"✅ DDP Process Group Initialized (Rank {local_rank})")

    
    for key, value in vars(args).items():
        config_key = key.upper()
        if value is not None and hasattr(cfg, config_key):
            setattr(cfg, config_key, value)
            print(f"Config override from command line: {config_key} = {value}")

    print("\n--- 1. Initializing ---")
    # --- [Optimization] 1. Enable CuDNN Benchmark ---
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✅ CuDNN Benchmark Enabled")
        
    setup_device_and_batch_size(cfg)
    set_seed_all(cfg.SEED)
    create_save_directories(cfg)

    print("\n--- 2. Loading Data & Creating Dataloaders ---")
    train_loader = None
    val_loader = None
    
    if not cfg.EVAL_ONLY:
        print("\n--- 2. Loading Data & Creating Dataloaders ---")
        train_data_list = process_folder_to_frame_lists(cfg.TRAIN_DATA_FOLDER, cfg.DATA_FILE_PATTERN, "Train", cfg)
        val_data_list = process_folder_to_frame_lists(cfg.VAL_DATA_FOLDER, cfg.DATA_FILE_PATTERN, "Validation", cfg)
        train_dataset = EventFrameLazyDataset(train_data_list, cfg)
        val_dataset = EventFrameLazyDataset(val_data_list, cfg) if val_data_list else None
        
        # --- [Optimization] 2. Persistent Workers & Prefetch Factor ---
        # DataLoader arguments
        loader_kwargs = {
            'batch_size': cfg.BATCH_SIZE,
            'pin_memory': True,
            'num_workers': cfg.NUM_WORKERS
        }
        
        if cfg.NUM_WORKERS > 0:
            loader_kwargs['prefetch_factor'] = 2
            loader_kwargs['persistent_workers'] = True

        # If Stateless Training, we CAN shuffle! (Legacy Strategy)
        # If Stateful Training, we CANNOT shuffle (Must stay sequential).
        use_shuffle = args.stateless
        if use_shuffle:
            print("🔀 Stateless Mode: Enabling DataLoader Shuffle")
        else:
            print("🔗 Stateful Mode: Disabling DataLoader Shuffle (Sequential)")

        train_loader = DataLoader(train_dataset, shuffle=use_shuffle, drop_last=True, **loader_kwargs)
        
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset else None
        
        print(f"✅ Train Loader: {len(train_loader)} batches (Shuffle={'True' if use_shuffle else 'False'})")
        if val_loader:
            print(f"✅ Val Loader: {len(val_loader)} batches")
        else:
            print("⚠️  Val Loader: None (No validation data found)")


    # (Moved loading logic to after model init)


    print("\n--- 4. Initializing Model ---")
    snn_params = {'beta': cfg.SNN_BETA, 'threshold': cfg.SNN_THRESHOLD}
    
    # Parse conv_channels argument if provided (e.g., "32,64" -> [32, 64])
    conv_channels = [16, 32]  # Default
    if args.conv_channels:
        conv_channels = [int(c.strip()) for c in args.conv_channels.split(',')]
        print(f"🔧 Using custom conv_channels: {conv_channels}")
    
    model = Hybrid_SNN_Pure_BNN(snn_params=snn_params, output_classes=cfg.OUTPUT_CLASSES, 
                                 input_channels=cfg.INPUT_CHANNELS, conv_channels=conv_channels).to(cfg.DEVICE)
    
    # --- [New Feature] Load Pretrained Weights for Training ---
    # If not resuming from a full checkpoint, but load_model_path is provided, use it as init weights
    if not args.resume_checkpoint and args.load_model_path and not cfg.EVAL_ONLY:
         if os.path.exists(args.load_model_path):
             print(f"📥 Loading pretrained weights from {args.load_model_path} for training initialization...")
             loaded_data = torch.load(args.load_model_path, map_location=cfg.DEVICE)
             
             # Handle full checkpoint format (contains 'model_state_dict' key)
             if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                 print("  ℹ️  Detected full checkpoint format, extracting model_state_dict...")
                 state_dict = loaded_data['model_state_dict']
             else:
                 # Assume it's already a pure state_dict
                 state_dict = loaded_data
             
             # Handle DataParallel 'module.' prefix
             if all(key.startswith('module.') for key in state_dict.keys()) and not isinstance(model, nn.DataParallel):
                 from collections import OrderedDict
                 state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
             
             try:
                 model.load_state_dict(state_dict, strict=False)
                 print("✅ Pretrained weights loaded successfully.")
             except Exception as e:
                 print(f"⚠️  Warning: Failed to load some weights: {e}")
         else:
             print(f"❌ Error: Pretrained model path not found: {args.load_model_path}")

    # --- [Optimization] 3. PyTorch 2.0 Compile ---
    # Optional: Compile model for speed (PyTorch 2.0+)
    # print("🚀 Compiling model with torch.compile()...")
    # model = torch.compile(model)
    # print("✅ Model compiled successfully.")
    # except Exception as e: # This line is commented out because 'except' without 'try' is a syntax error.
    #     print(f"⚠️  Model compilation failed: {e}. Proceeding without compilation.")

    # --- Model Wrapping (DDP / DataParallel) ---
    if "LOCAL_RANK" in os.environ:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.DEVICE], output_device=cfg.DEVICE, find_unused_parameters=True)
        print(f"✅ Model wrapped in DistributedDataParallel (device: {cfg.DEVICE})")
    elif cfg.USE_MULTI_GPU and cfg.N_GPU_EFFECTIVE > 1: 
        model = nn.DataParallel(model)
        print(f"✅ Model wrapped in DataParallel")
    print(f"Initialized {cfg.MODEL_TYPE} model on {cfg.DEVICE}.")

    # --- 6. Loss Function Initialization ---
    loss_function = None
    learnable_loss_params = []

    if cfg.LOSS_TYPE == 'Focal':
        # train_engine passes (logits, targets, mask, alpha, gamma)
        # So we define it to accept them, but we can ignore the passed alpha/gamma if we want to use cfg's, 
        # OR we use the passed ones. 
        # train_engine.py passes config_obj.FOCAL_ALPHA.
        # So the lambda should simply call focal_loss.
        loss_function = focal_loss 
        print(f"Using FocalLoss (alpha={cfg.FOCAL_ALPHA}, gamma={cfg.FOCAL_GAMMA})")
    elif cfg.LOSS_TYPE == 'Tversky':
        loss_function = TverskyLoss(alpha=cfg.TVERSKY_ALPHA, beta=cfg.TVERSKY_BETA)
        print(f"Using TverskyLoss (alpha={cfg.TVERSKY_ALPHA}, beta={cfg.TVERSKY_BETA})")
    elif cfg.LOSS_TYPE == 'FocalTversky':
        loss_function = FocalTverskyLoss(alpha=cfg.FOCAL_TVERSKY_ALPHA, beta=cfg.FOCAL_TVERSKY_BETA, gamma=cfg.FOCAL_TVERSKY_GAMMA)
        print(f"Using FocalTverskyLoss (alpha={cfg.FOCAL_TVERSKY_ALPHA}, beta={cfg.FOCAL_TVERSKY_BETA}, gamma={cfg.FOCAL_TVERSKY_GAMMA})")
    elif cfg.LOSS_TYPE == 'LearnableFocal':
        from utils import LearnableFocalLoss
        loss_module = LearnableFocalLoss(init_alpha=cfg.FOCAL_ALPHA, init_gamma=cfg.FOCAL_GAMMA).to(cfg.DEVICE)
        loss_function = loss_module # forward method will be called
        learnable_loss_params = list(loss_module.parameters())
        print(f"Using LearnableFocalLoss (init_alpha={cfg.FOCAL_ALPHA}, init_gamma={cfg.FOCAL_GAMMA})")
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {cfg.LOSS_TYPE}")

    # --- 5. Optimizer & Scheduler ---
    optimizer = None
    scheduler = None
    
    if not cfg.EVAL_ONLY:
        print("\n--- 5. Optimizer & Scheduler ---")
        params_to_optimize = list(model.parameters()) + learnable_loss_params
        optimizer = optim.AdamW(params_to_optimize, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY_ADAMW)
        
        if hasattr(args, 'lr_scheduler_type'):
            scheduler_type = args.lr_scheduler_type
        else:
            scheduler_type = 'OneCycle' # Default

        if scheduler_type == 'OneCycle':
            if train_loader is not None and len(train_loader) > 0:
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.LEARNING_RATE, total_steps=cfg.NUM_EPOCHS * len(train_loader))
                print(f"✅ Scheduler: OneCycleLR (Max LR: {cfg.LEARNING_RATE})")
            else:
                print("⚠️ Train loader is empty or None. Skipping OneCycleLR.")
        elif scheduler_type == 'Cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
            print(f"✅ Scheduler: CosineAnnealingLR (T_max: {cfg.NUM_EPOCHS})")
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
            print(f"✅ Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
        elif scheduler_type == 'None' or scheduler_type is None:
            scheduler = None
            # print(f"✅ Scheduler: None (Constant LR: {cfg.LEARNING_RATE})")
            print(f"✅ Scheduler: None (Constant LR)")
        else:
            print(f"⚠️ Unknown Scheduler Type: {scheduler_type}, defaulting to None")
    else:
        print("🚀 EVAL_ONLY mode: Skipping Optimizer & Scheduler initialization.")


    # --- 6. Starting Training ---
    print(f"\n--- 6. Starting Training ---")
    
    start_epoch = 0
    best_metric_val = float('-inf')
    
    # --- [Resume Logic] Load Checkpoint if provided ---
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            print(f"🔄 Resuming training from checkpoint: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=cfg.DEVICE)
            
            # 1. Load Model Weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel/DDP prefix if needed
            from collections import OrderedDict
            model_is_ddp = isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
            ckpt_has_prefix = all(key.startswith('module.') for key in state_dict.keys())
            
            if ckpt_has_prefix and not model_is_ddp:
                # Checkpoint has 'module.' but model doesn't -> remove prefix
                state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                print("  ℹ️  Removed 'module.' prefix from checkpoint keys")
            elif not ckpt_has_prefix and model_is_ddp:
                # Checkpoint lacks 'module.' but model needs it -> add prefix
                state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
                print("  ℹ️  Added 'module.' prefix to checkpoint keys")
            
            model.load_state_dict(state_dict, strict=False)
            
            # 2. Load Optimizer & Scheduler
            if optimizer and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and isinstance(checkpoint, dict) and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            # 3. Load Training State
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if isinstance(checkpoint, dict) and 'best_metric_val' in checkpoint:
                best_metric_val = checkpoint.get('best_metric_val', float('-inf'))
            
            print(f"✅ Loaded checkpoint (Resuming from Epoch {start_epoch})")
        else:
            print(f"❌ Error: Checkpoint file not found: {args.resume_checkpoint}")
            return


    
    if cfg.EVAL_ONLY:
        print("🚀 Skipping Training (EVAL_ONLY=True)")
        if not cfg.LOAD_MODEL_PATH or not os.path.exists(cfg.LOAD_MODEL_PATH):
            print(f"❌ Error: LOAD_MODEL_PATH not provided or does not exist: {cfg.LOAD_MODEL_PATH}")
            return
        
        # Load model state dict
        print(f"📥 Loading model from {cfg.LOAD_MODEL_PATH}...")
        state_dict = torch.load(cfg.LOAD_MODEL_PATH, map_location=cfg.DEVICE)
        
        # Handle DataParallel 'module.' prefix
        if all(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
            
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully.")

        # --- Override Config Parameters if provided in args (for eval mode) ---
        if args.save_dir:
            cfg.SAVE_DIR = args.save_dir
            print(f"⚠️  Overriding SAVE_DIR to: {args.save_dir}")
        if args.num_workers is not None:
            cfg.NUM_WORKERS = args.num_workers
            print(f"⚠️  Overriding NUM_WORKERS to: {args.num_workers}")

        # --- Override SNN Parameters if provided in args ---
        if args.snn_threshold is not None:
            print(f"⚠️  Overriding SNN Threshold to: {args.snn_threshold}")
            if hasattr(model, 'snn_act') and hasattr(model.snn_act, 'threshold'):
                model.snn_act.threshold.data.fill_(args.snn_threshold)
            elif hasattr(model, 'module') and hasattr(model.module, 'snn_act'): # Handle DataParallel if applicable
                model.module.snn_act.threshold.data.fill_(args.snn_threshold)
            else:
                 print("❌ Warning: Could not find snn_act.threshold to override.")

        if args.snn_beta is not None:
            print(f"⚠️  Overriding SNN Beta to: {args.snn_beta}")
            if hasattr(model, 'snn_act') and hasattr(model.snn_act, 'beta'):
                model.snn_act.beta.data.fill_(args.snn_beta)
            elif hasattr(model, 'module') and hasattr(model.module, 'snn_act'):
                model.module.snn_act.beta.data.fill_(args.snn_beta)
            else:
                 print("❌ Warning: Could not find snn_act.beta to override.")
        
        # Run Evaluation
        run_multiple_evaluations(config=cfg, model_filename=os.path.basename(cfg.LOAD_MODEL_PATH), model=model)
        return

    # TensorBoard Writer Initialization
    writer = None
    if cfg.USE_TENSORBOARD:
        log_dir = os.path.join(cfg.SAVE_DIR, 'runs')
        writer = SummaryWriter(log_dir=log_dir)
        print(f"✅ TensorBoard logging enabled at: {log_dir}")

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)
    if cfg.USE_AMP:
        print("✅ Mixed Precision Training (AMP) Enabled")

    # The following lines are moved or modified based on the instruction's implied structure.
    # best_metric_val = -float('inf') # Already initialized above, potentially loaded from checkpoint
    # train_history, val_history = [], [] # Initialized above, will be appended to
    
    # --- Early Stopping Variables ---
    patience_counter = 0
    patience_counter = 0
    if args.patience is not None:
        early_stopping_patience = args.patience
    else:
        early_stopping_patience = getattr(cfg, 'EARLY_STOPPING_PATIENCE', 15)
    print(f"🛑 Early Stopping Enabled with Patience: {early_stopping_patience}")
    print(f"💾 Saving Best Model based on: {cfg.SAVE_METRIC.upper()}")
    
    train_history = []
    val_history = []
    
    print(f"🚀 Training Started! (Epochs: {cfg.NUM_EPOCHS}) - Best Metric: {cfg.SAVE_METRIC}")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, loss_function, cfg.DEVICE, cfg, epoch, cfg.SAVE_DIR, writer=writer, scaler=scaler, force_stateless=args.stateless)
        train_history.append(train_metrics)
        
        if val_loader:
            val_metrics = validate_one_epoch(model, val_loader, loss_function, cfg.DEVICE, cfg, epoch, cfg.SAVE_DIR)
            val_history.append(val_metrics)
            
            # --- Get Metrics ---
            train_f1 = train_metrics.get('f1', 0.0)
            train_snr = train_metrics.get('snr_tp_fp', float('-inf'))
            train_precision = train_metrics.get('precision', 0.0)
            train_recall = train_metrics.get('recall', 0.0)
            
            val_f1 = val_metrics.get('f1', 0.0)
            val_snr = val_metrics.get('snr_tp_fp', float('-inf'))
            val_precision = val_metrics.get('precision', 0.0)
            val_recall = val_metrics.get('recall', 0.0)
            val_da = val_metrics.get('denoising_accuracy_da', -1.0)
            val_nrr = val_metrics.get('noise_removal_nr', 0.0)
            val_loss = val_metrics.get('loss', 0.0)
            
            # --- Log Validation Metrics to TensorBoard ---
            if writer:
                writer.add_scalar('Train/F1', train_f1, epoch)
                writer.add_scalar('Train/SNR', train_snr, epoch) if train_snr != float('-inf') else None
                writer.add_scalar('Train/Precision', train_precision, epoch)
                writer.add_scalar('Train/Recall', train_recall, epoch)
                writer.add_scalar('Val/Loss', val_loss, epoch)
                writer.add_scalar('Val/F1', val_f1, epoch)
                writer.add_scalar('Val/SNR', val_snr, epoch) if val_snr != float('-inf') else None
                writer.add_scalar('Val/Precision', val_precision, epoch)
                writer.add_scalar('Val/Recall', val_recall, epoch)
                writer.add_scalar('Val/DA', val_da, epoch)
                writer.add_scalar('Val/NRR', val_nrr, epoch)
            
            # Compact epoch summary
            print(f"Epoch {epoch}/{cfg.NUM_EPOCHS} | Train F1: {train_f1:.4f} | Train SNR: {train_snr:.2f} | Train P: {train_precision:.4f} | Train R: {train_recall:.4f}")
            print(f"{'':27s} | Val F1: {val_f1:.4f} | Val SNR: {val_snr:.2f} | Val P: {val_precision:.4f} | Val R: {val_recall:.4f}")
            
            # --- Determine Metric to Monitor ---
            metric_key = cfg.SAVE_METRIC.lower()
            if metric_key in ['snr', 'snr_tp_fp']:
                current_val = val_snr
            elif metric_key == 'f1':
                current_val = val_f1
            elif metric_key == 'da':
                current_val = val_da
            elif metric_key == 'recall':
                current_val = val_metrics.get('recall', 0.0)
            elif metric_key == 'precision':
                current_val = val_metrics.get('precision', 0.0)
            elif metric_key == 'auc':
                current_val = val_metrics.get('auc', 0.0)
            else:
                current_val = val_f1 # Default
                
            # --- Save Best Model ---
            if current_val > best_metric_val:
                best_metric_val = current_val
                torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'best_model.pth'))
                print(f"🚀 New best model saved with {cfg.SAVE_METRIC.upper()}: {best_metric_val:.4f}")
                patience_counter = 0 # Reset patience
            else:
                patience_counter += 1
                print(f"⏳ No improvement in {cfg.SAVE_METRIC.upper()}. Patience: {patience_counter}/{early_stopping_patience}")
            
            # --- [Resume Logic] Save Latest Checkpoint ---
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_metric_val': best_metric_val,
                'config': vars(cfg) # Save config just in case
            }
            torch.save(checkpoint, os.path.join(cfg.SAVE_DIR, 'latest_checkpoint.pth'))

                
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early Stopping Triggered! No improvement for {early_stopping_patience} epochs.")
                break
            
            # --- Update Learning Rate Scheduler ---
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_val)  # ReduceLROnPlateau needs metric
                # OneCycleLR steps per batch in train_engine, so skip here
                elif not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    scheduler.step()  # Other schedulers step per epoch

        else:
            # (검증 세트가 없는 경우의 print문)
            train_da = train_metrics.get('da', 0.0)
            print(f"Epoch {epoch}/{cfg.NUM_EPOCHS} | Train DA: {train_da:.4f}")

    torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'last_model.pth'))
    visualize_training_history(train_history, val_history, cfg)
    if writer is not None:
        writer.close()
        
    print(f"--- Training Finished ---")

    # --- Save Final Learnable Parameters ---
    params_save_path = os.path.join(cfg.SAVE_DIR, 'final_learned_params.txt')
    with open(params_save_path, 'w') as f:
        f.write("Final Learned Parameters\n")
        f.write("========================\n")
        
        # SNN Parameters
        target_model = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(target_model, 'snn_act'):
            # Beta
            if hasattr(target_model.snn_act, 'beta') and isinstance(target_model.snn_act.beta, nn.Parameter):
                raw_beta = target_model.snn_act.beta.item()
                # Calculate Quantized Beta (Nearest 0.25 step for Shift-Add operations)
                # Supports 0, 0.25, 0.5, 0.75, 1.0
                clamped_beta = max(0.0, min(1.0, raw_beta))
                quantized_beta = round(clamped_beta * 4) / 4.0
                
                f.write(f"SNN Beta (Raw): {raw_beta:.6f}\n")
                f.write(f"SNN Beta (Hardware/Quantized): {quantized_beta:.6f}\n")

            # Threshold
            if hasattr(target_model.snn_act, 'threshold') and isinstance(target_model.snn_act.threshold, nn.Parameter):
                raw_thr = target_model.snn_act.threshold.item()
                # Calculate Quantized Threshold (4-bit Int)
                # Logic: Round to nearest integer (more robustness than floor) clamped to [-8, 7]
                q_bit = 4
                thr_max = (2**(q_bit-1)-1)
                thr_min = -(2**(q_bit-1))
                quantized_thr = max(thr_min, min(thr_max, round(raw_thr)))
                
                f.write(f"SNN Threshold (Raw): {raw_thr:.6f}\n")
                f.write(f"SNN Threshold (Hardware/Int4): {int(quantized_thr)}\n")
        
        # Loss Parameters
        if hasattr(loss_function, 'alpha') and hasattr(loss_function, 'gamma'):
             # Check if they are properties or attributes
             alpha_val = loss_function.alpha if not callable(loss_function.alpha) else loss_function.alpha
             gamma_val = loss_function.gamma if not callable(loss_function.gamma) else loss_function.gamma
             
             # If they are tensors (e.g. from property), get item()
             if isinstance(alpha_val, torch.Tensor): alpha_val = alpha_val.item()
             if isinstance(gamma_val, torch.Tensor): gamma_val = gamma_val.item()
             
             f.write(f"Loss Alpha: {alpha_val:.6f}\n")
             f.write(f"Loss Gamma: {gamma_val:.6f}\n")
    
    print(f"✅ Final learned parameters saved to: {params_save_path}")

    #run_evaluation_after_training(config=cfg, model_filename='best_model.pth')
    run_multiple_evaluations(config=cfg, model_filename='best_model.pth', model=model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the Hybrid_ConvSNN_BNN model.")
    
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--base_batch_size', type=int, default=None)
    parser.add_argument('--weight_decay_adamw', type=float, default=None)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--train_data_folder', type=str, default=None)
    parser.add_argument('--val_data_folder', type=str, default=None)
    parser.add_argument('--test_data_folder', type=str, default=None)
    parser.add_argument('--snn_beta', type=float, default=None, help='Override SNN Beta')
    parser.add_argument('--lr_scheduler_type', type=str, default='OneCycle', help='Scheduler type: OneCycle, Cosine, None')
    parser.add_argument('--snn_threshold', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None, help='Early Stopping Patience (default: 15)')
    parser.add_argument('--focal_alpha', type=float, default=None)
    parser.add_argument('--focal_gamma', type=float, default=None)
    parser.add_argument('--tversky_alpha', type=float, default=None, help="Alpha parameter for Tversky Loss.")
    parser.add_argument('--tversky_beta', type=float, default=None, help="Beta parameter for Tversky Loss.")
    parser.add_argument('--focal_tversky_alpha',type=float,default=None)
    parser.add_argument('--focal_tversky_beta',type=float,default=None)
    parser.add_argument('--focal_tversky_gamma',type=float,default=None)
    parser.add_argument('--specific_gpu_id', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--stateless', action='store_true', help="Enable Stateless Training (Reset memory every batch)")
    parser.add_argument('--num_workers', type=int, default=None, help="Number of workers for DataLoader")
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--csv_name', type=str, default=None)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help="Use multiple GPUs if available")
    parser.add_argument('--loss_type', type=str, default=None, help="Loss function type: 'Focal', 'Tversky', 'FocalTversky', 'LearnableFocal'")
    parser.add_argument('--save_metric', type=str, default='f1', help="Metric to use for saving best model (f1, snr, recall, precision, da, auc)")
    parser.add_argument('--eval_only', action='store_true', default=False, help="Skip training and run evaluation only.")
    parser.add_argument('--load_model_path', type=str, default=None, help="Path to model to load for evaluation (required if eval_only is True).")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint(.pth) to resume training from.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Accumulate gradients over N steps before optimizer update (for single GPU training)")
    parser.add_argument('--processed_data_save_dir', type=str, default=None, help="Directory to save/load processed frame cache (use different dir for each FPS)")
    parser.add_argument('--conv_channels', type=str, default=None, help="Comma-separated list of conv channel sizes, e.g. '32,64' for doubled channels")

    args = parser.parse_args()
    main_training_pipeline(args)
