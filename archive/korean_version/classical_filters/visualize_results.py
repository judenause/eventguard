"""
Visualization utilities for classical filter comparison.

Generates comparison charts and plots for filter performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import os


def plot_filter_comparison(df_aggregated: pd.DataFrame, 
                          output_dir: str,
                          dataset_name: str):
    """
    Create comprehensive comparison plots for all filters.
    
    Args:
        df_aggregated: DataFrame with aggregated metrics per filter
        output_dir: Directory to save plots
        dataset_name: Name of the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Classical Filter Performance Comparison - {dataset_name}', 
                 fontsize=16, fontweight='bold')
    
    filters = df_aggregated['filter_name'].tolist()
    
    # 1. Stream-level metrics comparison
    ax1 = axes[0, 0]
    metrics = ['avg_stream_da', 'avg_stream_f1', 'avg_stream_precision', 'avg_stream_recall']
    metric_labels = ['DA', 'F1', 'Precision', 'Recall']
    
    x = np.arange(len(filters))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = df_aggregated[metric].values
        ax1.bar(x + i*width, values, width, label=label)
    
    ax1.set_xlabel('Filter', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Event-Stream Level Metrics', fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(filters, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Frame-level metrics comparison
    ax2 = axes[0, 1]
    metrics = ['avg_frame_da', 'avg_frame_f1', 'avg_frame_precision', 'avg_frame_recall']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = df_aggregated[metric].values
        ax2.bar(x + i*width, values, width, label=label)
    
    ax2.set_xlabel('Filter', fontweight='bold')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Frame-Level Metrics', fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(filters, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Hardware complexity comparison
    ax3 = axes[1, 0]
    ops_per_event = df_aggregated['avg_ops_per_event'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(filters)))
    
    bars = ax3.bar(filters, ops_per_event, color=colors)
    ax3.set_xlabel('Filter', fontweight='bold')
    ax3.set_ylabel('Operations per Event', fontweight='bold')
    ax3.set_title('Hardware Complexity (Lower is Better)', fontweight='bold')
    ax3.set_xticklabels(filters, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # 4. Performance vs Complexity scatter plot
    ax4 = axes[1, 1]
    stream_f1 = df_aggregated['avg_stream_f1'].values
    ops = df_aggregated['avg_ops_per_event'].values
    
    ax4.scatter(ops, stream_f1, s=200, alpha=0.6, c=colors)
    
    for i, filter_name in enumerate(filters):
        ax4.annotate(filter_name, (ops[i], stream_f1[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Operations per Event', fontweight='bold')
    ax4.set_ylabel('Stream F1 Score', fontweight='bold')
    ax4.set_title('Performance vs Complexity Trade-off', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add ideal region annotation (high F1, low ops)
    ax4.axhline(y=0.8, color='g', linestyle='--', alpha=0.3, label='High Performance')
    ax4.axvline(x=ops.mean(), color='r', linestyle='--', alpha=0.3, label='Avg Complexity')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{dataset_name}_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {plot_path}")
    
    plt.close()


def plot_operation_breakdown(df_aggregated: pd.DataFrame,
                             output_dir: str,
                             dataset_name: str):
    """
    Create stacked bar chart showing breakdown of hardware operations.
    
    Args:
        df_aggregated: DataFrame with aggregated metrics
        output_dir: Output directory
        dataset_name: Dataset name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    filters = df_aggregated['filter_name'].tolist()
    
    # Operation types to plot
    op_types = ['avg_comparisons', 'avg_additions', 'avg_multiplications', 
                'avg_divisions', 'avg_exp_ops', 'avg_memory_accesses']
    op_labels = ['Comparisons', 'Additions', 'Multiplications', 
                 'Divisions', 'Exp', 'Memory Access']
    
    # Prepare data
    op_data = []
    for op_type in op_types:
        op_data.append(df_aggregated[op_type].values)
    
    # Create stacked bar chart
    x = np.arange(len(filters))
    bottom = np.zeros(len(filters))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(op_types)))
    
    for i, (data, label, color) in enumerate(zip(op_data, op_labels, colors)):
        ax.bar(x, data, bottom=bottom, label=label, color=color)
        bottom += data
    
    ax.set_xlabel('Filter', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Operations', fontweight='bold', fontsize=12)
    ax.set_title(f'Hardware Operation Breakdown - {dataset_name}', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(filters, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{dataset_name}_operation_breakdown.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Operation breakdown plot saved to: {plot_path}")
    
    plt.close()


def create_all_visualizations(aggregated_csv_path: str, 
                              output_dir: str,
                              dataset_name: str):
    """
    Create all visualizations from aggregated results CSV.
    
    Args:
        aggregated_csv_path: Path to aggregated results CSV
        output_dir: Output directory for plots
        dataset_name: Name of the dataset
    """
    # Load data
    df_aggregated = pd.read_csv(aggregated_csv_path)
    
    print(f"\n📊 Generating visualizations for {dataset_name}...")
    
    # Generate plots
    plot_filter_comparison(df_aggregated, output_dir, dataset_name)
    plot_operation_breakdown(df_aggregated, output_dir, dataset_name)
    
    print(f"✅ All visualizations completed!\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualization plots")
    parser.add_argument('--results_csv', type=str, required=True,
                       help='Path to aggregated results CSV file')
    parser.add_argument('--output_dir', type=str, default='./results/',
                       help='Output directory for plots')
    parser.add_argument('--dataset_name', type=str, default='test_50',
                       help='Dataset name for plot titles')
    
    args = parser.parse_args()
    
    create_all_visualizations(args.results_csv, args.output_dir, args.dataset_name)
