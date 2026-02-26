#!/bin/bash
# ONF Corrected Evaluation Script

# Hotelbar ONF
python evaluate_roc_auc.py \
    --data_path ../../data/hotelbar/hotelbar_poisson_5hz.npy \
    --dataset_name hotelbar_poisson_onf_corrected \
    --filters ONF

# Driving ONF
python evaluate_roc_auc.py \
    --data_path ../../data/driving/driving_poisson_5hz.npy \
    --dataset_name driving_poisson_onf_corrected \
    --filters ONF
