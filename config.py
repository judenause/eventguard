import torch
from datetime import datetime
import os

class Config:
    # --- Model & Data Parameters ---
    MODEL_TYPE = 'Fully_Binary_ConvSNN'
    INPUT_CHANNELS = 1
    OUTPUT_CLASSES = 2
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30 

    # --- Data Paths (Update these as needed) ---
    DATA_ROOT = './data/esd/total/'
    TRAIN_DATA_FOLDER = os.path.join(DATA_ROOT, 'train/')
    VAL_DATA_FOLDER = os.path.join(DATA_ROOT, 'val/')
    TEST_DATA_FOLDER = './sample_data/'
    DATA_FILE_PATTERN = '*.npy'
    
    # --- Data Caching (Lazy Loading) ---
    PROCESSED_DATA_SAVE_DIR = './processed_cache' 
    
    # --- SNN Parameters ---
    SNN_BETA = 0.5
    SNN_THRESHOLD = 1.0
    
    # --- Model Architecture ---
    DILATION_RATES = [1, 1, 1] 
    
    # --- Training Parameters ---
    NUM_EPOCHS = 100
    BASE_BATCH_SIZE = 4 
    USE_AMP = True 
    LEARNING_RATE = 2e-3 
    WEIGHT_DECAY_ADAMW = 1e-5
    GRADIENT_CLIP_NORM = 1.0
    GRADIENT_ACCUMULATION_STEPS = 1 
    EARLY_STOPPING_PATIENCE = 15 
    
    # --- Loss Configuration (Tversky Loss) ---
    LOSS_TYPE = 'Tversky' 
    TVERSKY_ALPHA = 0.3 # Penalty for FN (Missed Signals)
    TVERSKY_BETA = 0.7  # Penalty for FP (Noise)
    
    # --- Sliding Window for DataLoader ---
    WINDOW_SIZE = 5
    WINDOW_STRIDE = 1
    
    # --- GPU Usage Control ---
    USE_MULTI_GPU = True
    SPECIFIC_GPU_ID = 0
    
    # --- Evaluation Parameters ---
    EVALUATION_THRESHOLD = 0.5
    EVALUATION_METHOD = 'full_sequence'
    CALC_SNR_TP_FP = True
    CREATE_EVAL_GIF = False
    
    # --- Performance & Logging ---
    USE_RESIDUAL = True
    USE_TENSORBOARD = True
    USE_DATA_AUGMENTATION = True
    
    SAVE_METRIC = 'f1' 
    EVAL_ONLY = False
    LOAD_MODEL_PATH = None

    # --- Others ---
    SEED = 3407
    SAVE_DIR = f'./results/run_{datetime.now().strftime("%Y%m%d_%H%M")}'
    CSV_NAME = 'final_results'
    
    # --- Dynamic values (do not change) ---
    DEVICE = None
    N_GPU_EFFECTIVE = 0
    BATCH_SIZE = BASE_BATCH_SIZE
    NUM_WORKERS = 4 

# Instance to be used across the project
cfg = Config()
