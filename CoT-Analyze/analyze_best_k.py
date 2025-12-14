"""
python analyze_best_k.py --input_file evaluate_topk.jsonl
"""
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    print(f"Loading data from {args.input_file}...")
    records = []
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            for r in item.get("responses", []):
                # 收集所有相关指标
                entry = {
                    "is_correct": r.get("score", 0.0),
                    # Full Entropy Std
                    "std_full": r.get("std_entropy_full", 0.0),
                }
                
                # 自动寻找所有 top-k 字段
                for k_key in r.keys():
                    if k_key.startswith("std_entropy_top"):
                        entry[k_key] = r[k_key]
                
                records.append(entry)
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} response samples.")
    
    # 过滤掉无法计算的数据
    df = df.dropna()
    if df['is_correct'].nunique() < 2:
        print("Error: Data only has one class (all correct or all wrong). Cannot compute AUC.")
        return

    # === 分析核心：计算 AUROC ===
    results = []
    
    # 获取所有待测的列名
    metric_cols = [c for c in df.columns if c.startswith("std_entropy")]
    metric_cols.sort(key=lambda x: int(x.split('top')[-1]) if 'top' in x else 9999) # 简单排序

    print("\n" + "="*60)
    print(f"{'Metric (Std Dev)':<25} | {'AUROC':<10} | {'Correlation':<12}")
    print("-" * 60)

    for col in metric_cols:
        # AUROC 计算：
        # 我们的假设是：Std 越小（越稳定），越容易 Correct。
        # 所以我们取负值 -df[col]，这样值越大代表越稳定（越好）。
        # AUC > 0.5 表示该指标有效。
        auc = roc_auc_score(df['is_correct'], -df[col])
        
        # Point-Biserial 相关系数
        corr = df['is_correct'].corr(df[col])
        
        print(f"{col:<25} | {auc:.4f}     | {corr:.4f}")
        results.append((col, auc))
        
    print("="*60)
    
    # 找出最佳
    best_metric, best_auc = max(results, key=lambda x: x[1])
    print(f"\n🏆 Best Metric: {best_metric}")
    print(f"   Score: {best_auc:.4f}")
    
    if "top" in best_metric:
        best_k = best_metric.split("top")[-1]
        print(f"👉 Recommendation: Use Top-{best_k} Entropy Standard Deviation.")
    else:
        print("👉 Recommendation: Use Full Entropy Standard Deviation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    main(args)