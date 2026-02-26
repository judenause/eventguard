
import os
import pandas as pd
import numpy as np

def load_filter_results_from_txt(fps, dataset):
    """
    Parses the summary.txt file to extract metrics for each filter.
    Returns a dictionary: { 'FilterName': { 'SNR': ..., 'F1': ..., ... }, ... }
    """
    base_dir = "./results/dvsclean_frame/filters"
    path = os.path.join(base_dir, f"fps{fps}", dataset, f"{dataset}_summary.txt")
    
    if not os.path.exists(path):
        print(f"⚠️ Warning: Summary file not found: {path}")
        return {}
        
    results = {}
    current_filter = None
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Detect Filter Name (e.g., "1. BAF" or "5. Bilateral")
            # We assume the line starts with a number and dot, or just the name if formatted differently
            # Based on previous view: "5. Bilateral"
            if line and line[0].isdigit() and ". " in line:
                parts = line.split(". ")
                if len(parts) >= 2:
                    current_filter = parts[1].strip()
                    results[current_filter] = {}
                    continue
            
            if current_filter:
                # Parse Stream Metrics
                if "SNR (dB):" in line:
                    val = float(line.split(":")[1].strip())
                    results[current_filter]['SNR'] = val
                elif "Precision:" in line:
                    val = float(line.split(":")[1].strip())
                    results[current_filter]['Precision'] = val
                elif "Recall:" in line:
                    val = float(line.split(":")[1].strip())
                    results[current_filter]['Recall'] = val
                # Removed heuristic F1 parsing. Rely on second pass.
                elif "NRR (Noise Rejection):" in line:
                    val = float(line.split(":")[1].strip())
                    results[current_filter]['NRR'] = val
                    
        # Re-parse with section awareness to obtain F1 correctly
        current_filter = None
        in_stream_section = False
        
        for line in lines:
            line = line.strip()
            
            if line and line[0].isdigit() and ". " in line:
                parts = line.split(". ")
                if len(parts) >= 2:
                    current_filter = parts[1].strip()
                    in_stream_section = False
                    continue
            
            if "Stream Metrics:" in line:
                in_stream_section = True
            elif "Frame Metrics:" in line:
                in_stream_section = False
                
            if current_filter and in_stream_section:
                if "SNR (dB):" in line:
                    results[current_filter]['SNR'] = float(line.split(":")[1].strip())
                elif "Precision:" in line:
                    results[current_filter]['Precision'] = float(line.split(":")[1].strip())
                elif "Recall:" in line:
                    results[current_filter]['Recall'] = float(line.split(":")[1].strip())
                elif "F1 Score:" in line:
                    results[current_filter]['F1'] = float(line.split(":")[1].strip())
                elif "NRR (Noise Rejection):" in line:
                    results[current_filter]['NRR'] = float(line.split(":")[1].strip())

        return results
        
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return {}

def load_snn_results(fps, dataset):
    # Path: snn/fpsXX/test_XX/snn_fpsXX_test_XX_results_stream_metrics.csv
    base_dir = "./results/dvsclean_frame/snn"
    filename = f"snn_fps{fps}_{dataset}_results_stream_metrics.csv"
    path = os.path.join(base_dir, f"fps{fps}", dataset, filename)
    
    if not os.path.exists(path):
        return None
        
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        # print(f"Error reading {path}: {e}")
        return None

def main():
    fps_list = [30, 60, 90, 120]
    methods = ['BAF', 'STCF', 'Bilateral', 'Ours']
    datasets = ['test_50', 'test_100']
    
    final_table = []
    
    for method in methods:
        for fps in fps_list:
            row_data = {'Method': method, 'FPS': fps}
            
            # --- Get Metrics for Test 50 ---
            m50 = None
            if method == 'Ours':
                # SNN doesn't have summary.txt (yet), use CSV logic or skip
                # For consistency with user request, we should probably stick to CSV for SNN 
                # but ensure we align metrics format.
                df = load_snn_results(fps, 'test_50')
                if df is not None and not df.empty:
                    m50 = get_metrics_from_row(df.iloc[0], type='snn')
            else:
                # Load from TXT
                all_filters_res = load_filter_results_from_txt(fps, 'test_50')
                m50 = all_filters_res.get(method)
            
            if m50:
                for k, v in m50.items():
                    row_data[f'50_{k}'] = v
            else:
                 for k in ['SNR', 'Precision', 'Recall', 'F1', 'NRR']:
                    row_data[f'50_{k}'] = np.nan
            
            # --- Get Metrics for Test 100 ---
            m100 = None
            if method == 'Ours':
                df = load_snn_results(fps, 'test_100')
                if df is not None and not df.empty:
                    m100 = get_metrics_from_row(df.iloc[0], type='snn')
            else:
                 # Load from TXT
                all_filters_res = load_filter_results_from_txt(fps, 'test_100')
                m100 = all_filters_res.get(method)

            if m100:
                for k, v in m100.items():
                    row_data[f'100_{k}'] = v
            else:
                 for k in ['SNR', 'Precision', 'Recall', 'F1', 'NRR']:
                    row_data[f'100_{k}'] = np.nan

            # --- Calculate Average ---
            for k in ['SNR', 'Precision', 'Recall', 'F1', 'NRR']:
                v50 = row_data.get(f'50_{k}', np.nan)
                v100 = row_data.get(f'100_{k}', np.nan)
                
                if pd.notna(v50) and pd.notna(v100):
                    row_data[f'Avg_{k}'] = (v50 + v100) / 2
                elif pd.notna(v50):
                    row_data[f'Avg_{k}'] = v50
                elif pd.notna(v100):
                    row_data[f'Avg_{k}'] = v100
                else:
                    row_data[f'Avg_{k}'] = np.nan
            
            final_table.append(row_data)
            
    # Create DataFrame
    cols_order = ['Method', 'FPS']
    for prefix in ['50', '100', 'Avg']:
        for metric in ['SNR', 'Precision', 'Recall', 'F1', 'NRR']:
            cols_order.append(f'{prefix}_{metric}')
            
    df_final = pd.DataFrame(final_table)
    df_final = df_final[cols_order]
    
    # Save to CSV
    output_path = "./results/dvsclean_frame/final_consolidated_summary.csv"
    df_final.to_csv(output_path, index=False)
    print(f"✅ Consolidated summary saved to {output_path}")
    
    # Print formatted table for User
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n")
    print(df_final.to_string(index=False))

if __name__ == "__main__":
    main()
