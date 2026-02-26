#!/usr/bin/env python3
"""
Run filter evaluation for multiple FPS settings.
Each FPS gets a matching temporal window (τ = 1/FPS).
Results are saved separately for each FPS.
"""

import subprocess
import os
import shutil

# FPS settings and corresponding temporal windows
FPS_CONFIGS = {
    30: 0.033,   # 33.3ms
    60: 0.017,   # 16.7ms
    90: 0.011,   # 11.1ms
    120: 0.008,  # 8.3ms
}

FILTERS = ['BAF', 'STCF', 'ONF', 'STCF_Sub']
CONFIG_TEMPLATE = '''import os

class Config:
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = {fps}
    
    TEST_DATA_FOLDER_50 = '/local_data/EventGuard/EventSNN/data/esd/total/test_50/'
    TEST_DATA_FOLDER_100 = '/local_data/EventGuard/EventSNN/data/esd/total/test_100/'
    DATA_FILE_PATTERN = '*.npy'
    
    FILTER_CONFIGS = {{
        'BAF': {{
            'time_window': {tau},
        }},
        'STCF': {{
            'spatial_radius': 1,
            'temporal_window': {tau},
            'min_neighbors': 2,
        }},
        'Refractory': {{
            'refractory_period': 0.001,
        }},
        'NN': {{
            'spatial_radius': 1,
            'temporal_window': {tau},
            'min_neighbors': 2,
        }},
        'Bilateral': {{
            'spatial_sigma': 1.5,
            'temporal_sigma': {tau},
            'threshold': 2.0,
        }},
        'STCF_Sub': {{
            'block_size': 2,
            'time_window': {tau},
        }},
        'ONF': {{
            'time_window': {tau},
            'width': 1280,
            'height': 720,
        }}
    }}
    
    RESULTS_DIR = './results/'
    
    HW_OP_COSTS = {{
        'comparison': 1,
        'addition': 1,
        'multiplication': 2,
        'division': 4,
        'sqrt': 8,
        'exp': 16,
        'memory_access': 1,
    }}

cfg = Config()
'''

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_base = os.path.join(base_dir, 'results')
    
    for fps, tau in FPS_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"🔄 Testing FPS={fps}, τ={tau*1000:.1f}ms")
        print(f"{'='*70}")
        
        # Update config.py
        config_content = CONFIG_TEMPLATE.format(fps=fps, tau=tau)
        config_path = os.path.join(base_dir, 'config.py')
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Create FPS-specific results directory
        fps_results_dir = os.path.join(results_base, f'fps{fps}')
        os.makedirs(fps_results_dir, exist_ok=True)
        
        # Run evaluation
        cmd = [
            'python', 'evaluate_filters.py',
            '--test_folder', 'both',
            '--filters'] + FILTERS + [
            '--output_dir', fps_results_dir
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=base_dir)
        
        if result.returncode != 0:
            print(f"❌ Error evaluating FPS={fps}")
        else:
            print(f"✅ FPS={fps} completed. Results in {fps_results_dir}")
    
    print(f"\n{'='*70}")
    print("✅ All FPS evaluations completed!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
