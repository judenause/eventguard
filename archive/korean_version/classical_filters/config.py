import os

class Config:
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 120
    
    TEST_DATA_FOLDER_50 = '/local_data/EventGuard/EventSNN/data/esd/total/test_50/'
    TEST_DATA_FOLDER_100 = '/local_data/EventGuard/EventSNN/data/esd/total/test_100/'
    DATA_FILE_PATTERN = '*.npy'
    
    FILTER_CONFIGS = {
        'BAF': {
            'time_window': 0.024,  # 24ms
        },
        'BAF_SinglePixel': {
            'time_window': 0.024,  # 24ms
        },
        'STCF': {
            'spatial_radius': 1,       # 3x3 영역 (Java 원본)
            'temporal_window': 0.024,  # 24ms (FPS에 따라 Override)
            'min_neighbors': 1,        # k=1
        },
        'Refractory': {
            'refractory_period': 0.001,
        },
        'NN': {
            'spatial_radius': 1,
            'temporal_window': 0.024,
            'min_neighbors': 2,
        },
        'Bilateral': {
            'spatial_sigma': 1.5,
            'temporal_sigma': 0.024,
            'threshold': 2.0,
        },
        'STCF_Sub': {
            'block_size': 2,
            'time_window': 0.024,       # 24ms
        },
        'ONF': {
            'time_window': 0.024,       # 24ms
            'width': 1280,
            'height': 720,
        }
    }
    
    RESULTS_DIR = './results/'
    
    HW_OP_COSTS = {
        'comparison': 1,
        'addition': 1,
        'multiplication': 2,
        'division': 4,
        'sqrt': 8,
        'exp': 16,
        'memory_access': 1,
    }

cfg = Config()
