"""
python validate_correlation.py --input_file analyze.jsonl
"""
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter

# -----------------------------------------------------------------------------
# 1. 辅助函数
# -----------------------------------------------------------------------------
def strip_string(string):
    if string is None: return ""
    string = str(string)
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5": string = "\\frac{1}{2}"
    return string

# -----------------------------------------------------------------------------
# 2. 数据加载 (含 Normalized SC 计算)
# -----------------------------------------------------------------------------
def load_data(input_file):
    print(f"Loading data from {input_file}...")
    records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            responses = item.get('responses', [])
            
            # --- 1. 预处理：计算当前问题下所有答案的频次和总数 ---
            all_answers = [strip_string(r.get('extracted_answer')) for r in responses]
            total_responses_count = len(all_answers) # 分母：总采样数
            answer_counts = Counter(all_answers)
            
            for resp in responses:
                if 'avg_entropy' not in resp: continue
                
                ans = strip_string(resp.get('extracted_answer'))
                count = answer_counts[ans]
                
                # --- 2. 计算归一化 SC Score (0 ~ 1) ---
                # 避免除以0 (虽然一般不会发生)
                sc_norm = count / total_responses_count if total_responses_count > 0 else 0.0
                
                rec = {
                    'is_correct': float(resp.get('score', 0.0)),
                    'avg_entropy': float(resp.get('avg_entropy')),
                    'std_entropy': float(resp.get('std_entropy')),
                    'perplexity': float(resp.get('perplexity', 0.0)),
                    'avg_certainty': float(resp.get('avg_certainty', 0.0)),
                    'sc_score': float(sc_norm)  # 现在是 0.0 - 1.0 的浮点数
                }
                records.append(rec)
                
    return pd.DataFrame(records).dropna()

# -----------------------------------------------------------------------------
# 3. 分析逻辑
# -----------------------------------------------------------------------------
def get_calibration_coordinates(df, metric_name, n_bins=20):
    try:
        df['bin'] = pd.qcut(df[metric_name], q=n_bins, duplicates='drop')
    except ValueError:
        df['bin'] = pd.cut(df[metric_name], bins=n_bins)

    grouped = df.groupby('bin', observed=True).agg({
        metric_name: 'mean',
        'is_correct': 'mean'
    })
    return grouped[metric_name].tolist(), grouped['is_correct'].tolist()

def get_boxplot_stats(df, metric_name):
    stats = {}
    for label in [0.0, 1.0]:
        group_name = "Correct (1.0)" if label == 1.0 else "Wrong (0.0)"
        data = df[df['is_correct'] == label][metric_name].values
        if len(data) == 0:
            stats[group_name] = "No Data"
            continue
        stats[group_name] = {
            "min": float(np.min(data)),
            "q1": float(np.percentile(data, 25)),
            "median": float(np.median(data)),
            "q3": float(np.percentile(data, 75)),
            "max": float(np.max(data)),
            "count": len(data)
        }
    return stats

# --- SC 详细报表 (适配浮点数) ---
def analyze_sc_detail(df):
    if 'sc_score' not in df.columns: return

    print(f"\n{'='*60}")
    print("SC SCORE (Normalized 0-1) vs ACCURACY TABLE")
    print(f"{'='*60}")
    
    # 为了避免浮点数精度问题导致的聚合分散，我们可以先保留4位小数进行聚合
    df['sc_rounded'] = df['sc_score'].round(4)
    
    sc_stats = df.groupby('sc_rounded')['is_correct'].agg(['mean', 'count']).reset_index()
    sc_stats.columns = ['SC_Score', 'Accuracy', 'Sample_Count']
    sc_stats = sc_stats.sort_values('SC_Score')

    # 打印表格
    print(f"{'SC Score':<12} | {'Accuracy':<12} | {'Sample Count':<12}")
    print("-" * 42)
    
    for _, row in sc_stats.iterrows():
        print(f"{row['SC_Score']:.4f}       | {row['Accuracy']:.6f}     | {int(row['Sample_Count']):<12}")
    
    # CSV 格式
    print("\n[CSV Format for Curve Fitting]")
    print("sc_score,accuracy,count")
    for _, row in sc_stats.iterrows():
        print(f"{row['SC_Score']:.4f},{row['Accuracy']:.6f},{int(row['Sample_Count'])}")

def analyze_and_print(df, metric_name, reverse_direction=True):
    # sc_score 单独处理了，这里只打印基础统计做对比
    if metric_name == 'sc_score':
        print(f"\n(See specific SC table below for detailed {metric_name} data)")
    
    print(f"\n{'='*60}")
    print(f"METRIC ANALYSIS: {metric_name}")
    print(f"{'='*60}")
    
    corr = df['is_correct'].corr(df[metric_name])
    score_for_auc = -df[metric_name] if reverse_direction else df[metric_name]
    
    try:
        auc = roc_auc_score(df['is_correct'], score_for_auc) if len(df['is_correct'].unique()) > 1 else 0.5
    except:
        auc = 0.5

    print(f"Correlation: {corr:.4f} | AUROC: {auc:.4f}")
    
    x, y = get_calibration_coordinates(df, metric_name)
    print(f"Calibration X: {np.round(x, 4).tolist()}")
    print(f"Calibration Y: {np.round(y, 4).tolist()}")
    
    print("Boxplot Stats:", json.dumps(get_boxplot_stats(df, metric_name), indent=2))

def main(args):
    df = load_data(args.input_file)
    print(f"Total records: {len(df)}")
    if len(df) == 0: return

    # 1. 优先输出 SC 详细表格
    analyze_sc_detail(df)

    # 2. 常规指标分析
    metrics = [
        ('avg_entropy', True),
        ('std_entropy', True),
        ('perplexity', True),
        ('avg_certainty', False),
        ('sc_score', False) # SC Score 越大越好
    ]
    
    for metric, reverse in metrics:
        if metric in df.columns:
            analyze_and_print(df, metric, reverse_direction=reverse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    main(args)