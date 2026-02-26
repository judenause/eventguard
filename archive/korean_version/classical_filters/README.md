# Classical Event Noise Filters

This directory contains implementations of classical (non-learning-based) event noise filtering algorithms for comparison with learning-based approaches.

## 📁 Directory Structure

```
classical_filters/
├── config.py              # Configuration and filter parameters
├── filters.py             # All filter implementations
├── utils.py              # Utility functions for metrics computation
├── evaluate_filters.py   # Main evaluation script
├── visualize_results.py  # Visualization tools
├── results/              # Output directory for results
└── README.md            # This file
```

## 🔧 Implemented Filters

### 1. **BAF (Background Activity Filter)**
- **Principle**: Removes events from pixels with no recent activity
- **Parameters**: `time_window` (default: 10ms)
- **Use case**: Simple temporal filtering for background noise

### 2. **STCF (Spatio-Temporal Correlation Filter)**
- **Principle**: Keeps events with spatial neighbors within temporal window
- **Parameters**: 
  - `spatial_radius` (default: 1 pixel)
  - `temporal_window` (default: 5ms)
  - `min_neighbors` (default: 1)
- **Use case**: Removes isolated noise events

### 3. **Refractory Period Filter**
- **Principle**: Removes events from same pixel occurring too quickly
- **Parameters**: `refractory_period` (default: 1ms)
- **Use case**: Mimics biological neuron behavior

### 4. **NN Filter (Nearest Neighbor)**
- **Principle**: Requires minimum number of spatial neighbors
- **Parameters**:
  - `spatial_radius` (default: 1 pixel)
  - `temporal_window` (default: 10ms)
  - `min_neighbors` (default: 2)
- **Use case**: Stricter spatial correlation filtering

### 5. **Bilateral Filter**
- **Principle**: Weighted spatial-temporal filtering with Gaussian kernels
- **Parameters**:
  - `spatial_sigma` (default: 1.5)
  - `temporal_sigma` (default: 5ms)
  - `threshold` (default: 0.5)
- **Use case**: Smooth filtering with edge preservation

## 🚀 Usage

### Basic Evaluation

Evaluate all filters on test_50 dataset:
```bash
cd /local_data/EventGuard/EventSNN/code/classical_filters
python evaluate_filters.py --test_folder test_50
```

Evaluate on test_100 dataset:
```bash
python evaluate_filters.py --test_folder test_100
```

Evaluate on both datasets:
```bash
python evaluate_filters.py --test_folder both
```

### Evaluate Specific Filters

```bash
python evaluate_filters.py --test_folder test_50 --filters BAF STCF
```

### Custom Output Directory

```bash
python evaluate_filters.py --test_folder test_50 --output_dir ./my_results/
```

### Generate Visualizations

After running evaluation, generate plots:
```bash
python visualize_results.py \
    --results_csv ./results/test_50_aggregated_results.csv \
    --output_dir ./results/ \
    --dataset_name test_50
```

## 📊 Output Files

After running evaluation, the following files are generated in `results/`:

### CSV Files
- `{dataset}_detailed_results.csv` - Per-file, per-filter metrics
- `{dataset}_aggregated_results.csv` - Aggregated metrics per filter
- `{dataset}_summary.txt` - Human-readable summary report

### Visualizations (after running visualize_results.py)
- `{dataset}_comparison.png` - Comprehensive comparison charts
- `{dataset}_operation_breakdown.png` - Hardware operation breakdown

## 📈 Metrics Computed

### Event-Stream Level Metrics
- **DA (Denoising Accuracy)**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives

### Frame-Level Metrics
- Same metrics computed on frame-based representations
- Useful for comparison with frame-based models

### Hardware Complexity Metrics
- **Total Operations**: Weighted sum of all operations
- **Operations per Event**: Average computational cost
- **Operation Breakdown**: Detailed count by operation type
  - Comparisons
  - Additions/Subtractions
  - Multiplications
  - Divisions
  - Exponentials
  - Memory Accesses

## ⚙️ Customizing Filter Parameters

Edit `config.py` to modify filter parameters:

```python
FILTER_CONFIGS = {
    'BAF': {'time_window': 0.01},  # Adjust time window
    'STCF': {
        'spatial_radius': 2,  # Increase neighborhood size
        'temporal_window': 0.01,
        'min_neighbors': 2
    },
    # ... other filters
}
```

## 🔬 Hardware Operation Costs

Operation costs are defined in `config.py`:

```python
HW_OP_COSTS = {
    'comparison': 1,
    'addition': 1,
    'multiplication': 2,
    'division': 4,
    'sqrt': 8,
    'exp': 16,
    'memory_access': 1,
}
```

These weights reflect relative computational complexity for hardware implementation.

## 📝 Example Results

```
AGGREGATED RESULTS - FILTER COMPARISON
======================================================================

Filter       Stream F1  Stream DA  Frame F1   Ops/Event   
----------------------------------------------------------------------
STCF         0.8234     0.8567     0.8123     45.23       
BAF          0.7891     0.8234     0.7856     12.45       
NN           0.8012     0.8345     0.7967     67.89       
Refractory   0.7456     0.7890     0.7234     8.90        
Bilateral    0.8456     0.8678     0.8345     123.45      
----------------------------------------------------------------------
```

## 🤝 Integration with v8_bconvsnn

To compare with your learning-based model:

1. Run classical filter evaluation:
   ```bash
   python evaluate_filters.py --test_folder both
   ```

2. Compare results with your model's output:
   - `v8_bconvsnn/results/*/final_results_stream_metrics.csv`
   - `classical_filters/results/test_50_aggregated_results.csv`

3. Key comparison points:
   - **Performance**: F1 score, DA
   - **Complexity**: Ops/event vs model parameters/FLOPs
   - **Speed**: Processing time per file

## 📚 References

- BAF: Delbruck, T. (2008). "Frame-free dynamic digital vision"
- STCF: Liu, H., et al. (2016). "Combined frame and event-based detection and tracking"
- Bilateral: Tomasi, C., & Manduchi, R. (1998). "Bilateral filtering for gray and color images"

## 🐛 Troubleshooting

**Issue**: "No files found in test folder"
- **Solution**: Check that `TEST_DATA_FOLDER_50` and `TEST_DATA_FOLDER_100` paths in `config.py` are correct

**Issue**: Memory error on large datasets
- **Solution**: Process files in batches or reduce dataset size

**Issue**: Slow evaluation
- **Solution**: Filters are implemented in pure Python/NumPy. For production, consider Cython or C++ implementation

## 📧 Notes

- All filters operate on raw event streams (not frames)
- Hardware operation counting is approximate and based on algorithmic analysis
- Results may vary based on filter parameter tuning
- For fair comparison, use same test datasets for all methods
