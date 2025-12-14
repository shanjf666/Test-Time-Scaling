import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =====================================================
# 1. 基础清洗工具
# =====================================================
def last_boxed_only_string(string):
    if not string or not isinstance(string, str): return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]

def remove_boxed(s):
    if s is None: return None
    if "\\boxed " in s: return s.replace("\\boxed ", "")
    if s.startswith("\\boxed{") and s.endswith("}"): return s[len("\\boxed{") : -1]
    return s

def strip_string(string):
    if string is None: return ""
    string = str(string).replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5": string = "\\frac{1}{2}"
    return string

def check_correctness(pred_ans, gold_content):
    gold_boxed = last_boxed_only_string(gold_content)
    gold_ans = gold_content if gold_boxed is None else remove_boxed(gold_boxed)
    norm_pred = strip_string(pred_ans)
    norm_gold = strip_string(gold_ans)
    if not norm_pred and not norm_gold: return False
    return norm_pred == norm_gold

# =====================================================
# 2. 定义拟合函数
# =====================================================

def func_linear(x, a, b):
    return a * x + b

def func_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def func_cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func_sigmoid(x, L, k, x0, b):
    # L:峰值, k:斜率, x0:中心点, b:截距
    return L / (1 + np.exp(-np.clip(k * (x - x0), -100, 100))) + b

def func_exponential(x, a, b, c):
    return a * np.exp(np.clip(b * x, -50, 50)) + c

def func_power(x, a, b, c):
    return a * np.power(x + 1e-6, b) + c

# =====================================================
# 3. 辅助：将参数转为公式字符串
# =====================================================
def get_equation_string(name, p):
    """
    根据模型名称和参数 popt 生成可读的公式字符串
    """
    if name == "Linear":
        return f"y = {p[0]:.4f}x + {p[1]:.4f}"
    
    elif name == "Quadratic":
        return f"y = {p[0]:.4f}x² + {p[1]:.4f}x + {p[2]:.4f}"
    
    elif name == "Cubic":
        return f"y = {p[0]:.4f}x³ + {p[1]:.4f}x² + {p[2]:.4f}x + {p[3]:.4f}"
    
    elif name == "Sigmoid":
        # y = L / (1 + e^(-k(x-x0))) + b
        return f"y = {p[0]:.4f} / (1 + exp(-{p[1]:.4f} * (x - {p[2]:.4f}))) + {p[3]:.4f}"
    
    elif name == "Exponential":
        return f"y = {p[0]:.4f} * exp({p[1]:.4f}x) + {p[2]:.4f}"
    
    elif name == "Power":
        return f"y = {p[0]:.4f} * x^{p[1]:.4f} + {p[2]:.4f}"
    
    return "Unknown function"

# =====================================================
# 4. 核心处理逻辑
# =====================================================

def process_and_fit(df, n_bins=15):
    # 1. 分桶
    bins = np.linspace(0, 1, n_bins + 1)
    df['bin_id'] = pd.cut(df['sc_score'], bins, labels=False, include_lowest=True)
    
    # 2. 聚合
    bin_stats = df.groupby('bin_id').agg({
        'sc_score': 'mean',      
        'is_correct': 'mean',    
        'bin_id': 'count'        
    }).rename(columns={'bin_id': 'count', 'sc_score': 'avg_conf', 'is_correct': 'avg_acc'})
    
    bin_stats = bin_stats[bin_stats['count'] > 0].reset_index()
    
    x_data = bin_stats['avg_conf'].values
    y_data = bin_stats['avg_acc'].values
    weights = bin_stats['count'].values
    sigma = 1.0 / (np.sqrt(weights) + 1e-6)

    # 3. 定义要尝试的模型
    models = [
        {"name": "Linear", "func": func_linear, "p0": [1, 0]},
        {"name": "Quadratic", "func": func_quadratic, "p0": [1, 0, 0]},
        {"name": "Cubic", "func": func_cubic, "p0": [1, 0, 0, 0]},
        {"name": "Sigmoid", "func": func_sigmoid, "p0": [1.0, 5.0, 0.5, 0.0]},
        {"name": "Exponential", "func": func_exponential, "p0": [0.01, 5, 0]},
        {"name": "Power", "func": func_power, "p0": [1, 2, 0]}
    ]
    
    print("\n" + "="*50)
    print(f"CALCULATED FORMULAS (Based on {n_bins} bins weighted fit)")
    print("="*50)

    results = []

    # 4. 循环拟合
    for model in models:
        try:
            # 拟合
            popt, pcov = curve_fit(model["func"], x_data, y_data, p0=model["p0"], sigma=sigma, absolute_sigma=False, maxfev=10000)
            
            # 计算 R2
            y_pred = model["func"](x_data, *popt)
            r2 = r2_score(y_data, y_pred, sample_weight=weights)
            
            # 生成公式字符串
            eq_str = get_equation_string(model["name"], popt)
            
            results.append({
                "name": model["name"],
                "r2": r2,
                "equation": eq_str
            })
            
        except Exception as e:
            results.append({
                "name": model["name"],
                "r2": -1.0,
                "equation": f"Fit Failed: {str(e)}"
            })

    # 5. 按 R2 排序并打印
    results.sort(key=lambda x: x["r2"], reverse=True)
    
    for rank, res in enumerate(results, 1):
        print(f"\nRank {rank}: {res['name']}")
        print(f"  R² Score : {res['r2']:.4f}")
        print(f"  Formula  : {res['equation']}")
        
    print("\n" + "="*50)

# =====================================================
# 5. 主入口
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="qwen64.jsonl")
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    # 读取数据
    data = []
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                sc_score = item.get('sc_score', 0.0)
                sc_answer = item.get('sc_answer', "")
                gold_content = item.get('answer', item.get('solution', "")) 
                is_correct = check_correctness(sc_answer, gold_content)
                data.append({"sc_score": sc_score, "is_correct": is_correct})
    except FileNotFoundError:
        print("File not found.")
        return

    df = pd.DataFrame(data)
    if len(df) == 0: return

    # 执行计算
    process_and_fit(df, n_bins=args.n_bins)

if __name__ == "__main__":
    main()