#!/usr/bin/env python3
"""
로그 파일에서 필터 결과를 복구하는 스크립트.
final_evaluation.log에서 원래 5개 필터 결과를 추출하여 CSV/TXT 파일로 저장.
"""
import re
import os

def parse_aggregated_results(log_content, dataset_name):
    """로그에서 aggregated 결과 파싱"""
    # dataset_name에 해당하는 섹션 찾기
    pattern = rf"# Processing Dataset: {dataset_name}.*?AGGREGATED RESULTS.*?Filter\s+Stream F1\s+Stream DA\s+Frame F1\s+Ops/Event.*?-+\n(.*?)-+"
    match = re.search(pattern, log_content, re.DOTALL)
    
    if not match:
        print(f"❌ {dataset_name} 결과를 찾을 수 없습니다.")
        return None
    
    results_text = match.group(1).strip()
    results = []
    
    for line in results_text.split('\n'):
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 4:
                results.append({
                    'filter_name': parts[0],
                    'stream_f1': float(parts[1]),
                    'stream_da': float(parts[2]),
                    'frame_f1': float(parts[3]),
                    'ops_per_event': float(parts[4]) if len(parts) > 4 else 0.0
                })
    
    return results

def create_summary_txt(results, dataset_name, results_dir):
    """summary.txt 파일 생성"""
    # 결과를 Stream F1 기준으로 정렬
    sorted_results = sorted(results, key=lambda x: x['stream_f1'], reverse=True)
    
    output = f"""Classical Event Noise Filters - Evaluation Summary
======================================================================
Dataset: {dataset_name}
Number of files: 49

Filter Performance Ranking (by Stream F1 Score)
----------------------------------------------------------------------

"""
    
    for i, r in enumerate(sorted_results, 1):
        output += f"""{i}. {r['filter_name']}
   Stream Metrics:
     - F1 Score:                {r['stream_f1']:.4f}
     - DA (Denoising Accuracy): {r['stream_da']:.4f}
   Frame Metrics:
     - F1 Score:                {r['frame_f1']:.4f}
   Hardware Complexity:
     - Ops per event:           {r['ops_per_event']:.2f}

"""
    
    output_path = os.path.join(results_dir, f'{dataset_name}_summary.txt')
    with open(output_path, 'w') as f:
        f.write(output)
    print(f"✅ Summary saved to: {output_path}")

def create_aggregated_csv(results, dataset_name, results_dir):
    """aggregated_results.csv 파일 생성"""
    header = "filter_name,avg_stream_f1,avg_stream_da,avg_frame_f1,avg_ops_per_event"
    
    output_path = os.path.join(results_dir, f'{dataset_name}_aggregated_results.csv')
    with open(output_path, 'w') as f:
        f.write(header + '\n')
        for r in results:
            f.write(f"{r['filter_name']},{r['stream_f1']},{r['stream_da']},{r['frame_f1']},{r['ops_per_event']}\n")
    
    print(f"✅ Aggregated results saved to: {output_path}")

def main():
    log_path = './final_evaluation.log'
    results_dir = './results'
    
    if not os.path.exists(log_path):
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_path}")
        return
    
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    print("=" * 60)
    print("로그 파일에서 결과 복구 중...")
    print("=" * 60)
    
    for dataset_name in ['test_50', 'test_100']:
        print(f"\n📊 Processing {dataset_name}...")
        results = parse_aggregated_results(log_content, dataset_name)
        
        if results:
            print(f"   Found {len(results)} filters: {[r['filter_name'] for r in results]}")
            create_summary_txt(results, dataset_name, results_dir)
            create_aggregated_csv(results, dataset_name, results_dir)
    
    print("\n" + "=" * 60)
    print("✅ 결과 복구 완료!")
    print("=" * 60)
    print("\n⚠️  주의: detailed_results.csv는 로그에서 복구할 수 없습니다.")
    print("    상세 결과가 필요하면 evaluate_filters.py를 다시 실행하세요.")

if __name__ == "__main__":
    main()
