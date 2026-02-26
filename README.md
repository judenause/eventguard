# EventSNN: Robust Event-based Signal-Noise Normalization

This repository contains the official implementation of the paper: **"Robust Event-based Signal-Noise Normalization using Spiking Neural Networks"**.

## Quick Start (Demo)

We provide a sample event stream and pre-trained weights for a quick demonstration.

```bash
# Run evaluation on the sample data using provided weights
python main_evaluate.py --model_file best_model_v8.pth --save_dir ./weights --test_data_folder ./sample_data --create_eval_gif
```

You can find the demo result GIF at `demo_results/eval_best_model_v8.gif`.

## Project Structure

- `model.py`: Core SNN architecture (Fully Binary ConvSNN).
- `custom_layers.py`: Custom SNN and BNN layers used in the model.
- `train_engine.py` / `evaluation_engine.py`: Training and evaluation logic.
- `dataset.py` / `data_processing.py`: Data loading and event-to-frame conversion.
- `config.py`: Configuration and hyperparameters.
- `classical_filters/`: Implementation of baseline classical filters (BAF, STCF, ONF).

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Training
To train the model from scratch:
```bash
python main_train.py
```

### 2. Evaluation
To evaluate a trained model:
```bash
python main_evaluate.py
```

### 3. Quantitative Analysis
- Hardware Metrics: `python calc_hardware_metrics.py`
- Sparsity Analysis: `python measure_all_sparsity.py`

## Data Preparation
Please place your event dataset in the `data/` directory. The expected format is `.npy` event streams.
Modify `config.py` to point to your data locations if they differ from the default.

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{eventsnn2026,
  title={Robust Event-based Signal-Noise Normalization},
  author={...},
  journal={...},
  year={2026}
}
```
