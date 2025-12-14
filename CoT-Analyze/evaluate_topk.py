"""
python evaluate_topk.py --model_path Qwen/Qwen2.5-Math-1.5B --input_file qwen64.jsonl --output_file evaluate_topk.jsonl --batch_size 4
"""
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import tqdm
import re
from typing import List, Tuple, Dict, Any
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

# # =============================================================================
# # 1. 答案提取与清洗工具 (User Provided Logic)
# # =============================================================================

def clean_latex_format(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    cleaned_text = text.strip()
    cleaned_text = cleaned_text.replace("$", "").replace("\\$", "$")
    
    latex_whitespaces = {
        r'\\,': '', r'\\:': '', r'\\;': '', r'\\!': '',
        r'\\enspace': '', r'\\quad': '', r'\\qquad': '',
        r'\\hspace{[^}]*}': '', r'\\vspace{[^}]*}': '',
        r'\\phantom{[^}]*}': '', r'\\hfill': '', r'\\space': '',
        r'\\ ': '', r'\\mspace{[^}]*}': '', r'\\kern{[^}]*}': '',
    }
    for pattern, replacement in latex_whitespaces.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    useless_cmds = [
        r'\\left', r'\\right', r'\\big', r'\\Big', r'\\bigg', r'\\Bigg',
        r'\\text\s*', r'\\mathrm', r'\\displaystyle', r'\\rm', r'\\it',
        r'\\bf', r'\\cal', r'\\scriptstyle', r'\\scriptscriptstyle'
    ]
    for cmd in useless_cmds:
        cleaned_text = re.sub(cmd, '', cleaned_text)

    cleaned_text = cleaned_text.replace(r'\(', '(').replace(r'\)', ')')
    cleaned_text = cleaned_text.replace(r'\[', '[').replace(r'\]', ']')
    cleaned_text = cleaned_text.replace(r'\{', '{').replace(r'\}', '}')

    symbol_replacements = {
        r'\\pi': 'pi', r'\\theta': 'theta', r'\\sqrt': 'sqrt',
        r'\\frac{([^}]*?)}({[^}]*?})': r'\1/\2',  
        r'\\times': '*', r'\\div': '/', r'\\infty': 'oo',
        r'\\alpha': 'alpha', r'\\beta': 'beta', r'\\gamma': 'gamma',
        r'\\delta': 'delta', r'\\sum': 'sum', r'\\int': 'integrate',
        r'\\cdot': '*', r'\\pm': '+-', r'\\mp': '-+'
    }
    for pattern, replacement in symbol_replacements.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    np_pattern = re.compile(r'[\x00-\x1f]')
    cleaned_text = np_pattern.sub('', cleaned_text)

    cleaned_text = re.sub(r'\(\s+', '(', cleaned_text)
    cleaned_text = re.sub(r'\s+\)', ')', cleaned_text)
    cleaned_text = re.sub(r'\[\s+', '[', cleaned_text)
    cleaned_text = re.sub(r'\s+\]', ']', cleaned_text)
    cleaned_text = re.sub(r'\{\s+', '{', cleaned_text)
    cleaned_text = re.sub(r'\s+\}', '}', cleaned_text)
    cleaned_text = re.sub(r'\s*,\s*', ',', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if not cleaned_text:
        return ""
    return cleaned_text

def last_boxed_only_string(string):
    if not isinstance(string, str): return None
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
    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval

def remove_boxed(s):
    if s is None: return None
    if "\\boxed " in s: return s.replace("\\boxed ", "")
    if s.startswith("\\boxed{") and s.endswith("}"): return s[len("\\boxed{") : -1]
    return s

def strip_string(string):
    if string is None: return ""
    string = str(string)
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace(" ", "")
    if string == "0.5": string = "\\frac{1}{2}"
    return string

def extract_model_answer(response_text: str) -> str:
    boxed_part = last_boxed_only_string(response_text)
    if boxed_part is None:
        return None
    answer = remove_boxed(boxed_part)
    return answer.strip()

def is_correct_answer(predicted: str, true_answer: str) -> bool:
    if predicted is None or true_answer is None:
        return False
    return strip_string(predicted) == strip_string(true_answer)

# =============================================================================
# 2. 步骤解析工具
# =============================================================================

def parse_structured_steps(response_text: str, min_len: int = 5) -> List[Tuple[int, int, str]]:
    # 策略 1: 显式 Step 标记
    steps = []
    step_pattern = re.compile(r'(?:##\s*)?Step\s*(?:\d+)\s*[:.]?\s*(.*?)(?=(?:##\s*)?Step\s*\d+\s*[:.]?|$)', re.DOTALL | re.IGNORECASE)
    matches = list(step_pattern.finditer(response_text))
    valid_matches = []
    for m in matches:
        content = m.group(1).strip()
        if len(content) >= min_len:
            valid_matches.append((m.start(1), m.end(1), content))
    if valid_matches:
        return valid_matches

    # 策略 2: 段落划分
    para_pattern = re.compile(r'\S.*?(?:\n\s*\n|$)', re.DOTALL)
    paragraphs = []
    for m in para_pattern.finditer(response_text):
        content = m.group(0).strip()
        if len(content) >= min_len and not content.startswith("<|"):
            paragraphs.append((m.start(), m.end(), content))
    if len(paragraphs) >= 2:
        return paragraphs

    # 策略 3: 句子划分 (兜底)
    sentence_pattern = re.compile(r'[^\.!\?]+[\.!\?]+', re.DOTALL)
    sentences = []
    for m in sentence_pattern.finditer(response_text):
        content = m.group(0).strip()
        if len(content) >= min_len:
            sentences.append((m.start(), m.end(), content))
    if len(sentences) >= 2:
        return sentences

    return [(0, len(response_text), response_text)]

# =============================================================================
# 2. 核心指标计算逻辑 (修改重点)
# =============================================================================

def compute_metrics_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, k_list: List[int]):
    """
    计算 Full Entropy 以及 Top-K Entropy。
    """
    # Shift logits and labels (预测下一个token)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 1. 基础 LogProbs (用于计算 PPL 和 Full Entropy)
    probs_all = F.softmax(shift_logits, dim=-1) # [B, Seq, V]
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    
    # 2. 获取实际 Token 的 LogProb
    token_logprobs = torch.gather(log_probs_all, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # 3. 计算 Full Entropy (Baseline)
    full_entropy = -(probs_all * torch.log(probs_all + 1e-12)).sum(dim=-1)
    
    # Pad prefix (为了对齐 input_ids 长度，第一位补0)
    def pad_prefix(tensor, val=0.0):
        prefix = torch.full((tensor.size(0), 1), val, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([prefix, tensor], dim=1)

    metrics = {
        "logprobs": pad_prefix(token_logprobs, 0.0),
        "entropy_full": pad_prefix(full_entropy, 0.0),
    }

    # 4. 计算 Top-K Entropy (核心新增)
    # 针对每一个 K 值进行计算
    for k in k_list:
        if k == 'full': continue
        
        # 取 Top-K 概率和索引
        # values: [B, Seq, K]
        topk_probs, _ = torch.topk(probs_all, k, dim=-1)
        
        # --- 关键：重归一化 (Renormalization) ---
        # 计算 P(x | x in TopK)
        sum_probs = topk_probs.sum(dim=-1, keepdim=True)
        renorm_probs = topk_probs / sum_probs
        
        # 计算 Top-K 熵
        ent_k = -(renorm_probs * torch.log(renorm_probs + 1e-12)).sum(dim=-1)
        
        metrics[f"entropy_top{k}"] = pad_prefix(ent_k, 0.0)

    return metrics

def aggregate_metrics(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """计算均值和标准差"""
    if len(values) == 0:
        return {f"avg_{prefix}": 0.0, f"std_{prefix}": 0.0}
    
    return {
        f"avg_{prefix}": float(np.mean(values)),
        f"std_{prefix}": float(np.std(values)) # 重点关注这个 std
    }

# =============================================================================
# 3. 主流程
# =============================================================================

@torch.no_grad()
def main(args):
    # --- 0. 定义要研究的 K 值 ---
    # 建议包含小(5)、中(10,20)、大(50,100)
    TARGET_K_LIST = [1, 5, 10, 20]

    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    llm.eval()

    print(f"Reading data from {args.input_file}...")
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    if args.max_samples: data = data[:args.max_samples]

    f_out = open(args.output_file, 'w', encoding='utf-8')
    print("Starting Detailed Metric Calculation...")
    
    for item in tqdm.tqdm(data):
        problem_text = item.get("problem", "") or item.get("question", "")
        ground_truth = item.get("answer", "") or item.get("solution", "")
        responses = item.get("responses", [])
        if not responses: continue
            
        # 预处理 responses
        processed_responses = []
        extracted_answers_list = []
        
        for idx, r_text in enumerate(responses):
            ext_ans = extract_model_answer(r_text)
            # 这里调用你的清洗函数
            # is_corr = is_correct_answer(...) 
            # 暂时简写，请替换回原本逻辑
            is_corr = False 
            if ext_ans and ground_truth:
                is_corr = (ext_ans.strip() == ground_truth.strip())

            if ext_ans: extracted_answers_list.append(ext_ans)
            
            processed_responses.append({
                "response_id": idx,
                "raw_text": r_text,
                "extracted_answer": ext_ans,
                "score": 1.0 if is_corr else 0.0
            })

        # SC Calculation (简化版)
        sc_score = 0.0
        if extracted_answers_list:
            c = Counter(extracted_answers_list)
            sc_score = c.most_common(1)[0][1] / len(responses)

        # --- Batch Processing ---
        batch_size = args.batch_size
        num_responses = len(processed_responses)
        prompt_template = f"Question: {problem_text}\nAnswer the question step by step.\nAnswer:"
        prompt_len_char = len(prompt_template)

        for batch_start in range(0, num_responses, batch_size):
            batch_end = min(batch_start + batch_size, num_responses)
            batch_objs = processed_responses[batch_start:batch_end]
            
            full_texts = [prompt_template + r["raw_text"] for r in batch_objs]
            
            # Tokenize
            inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048, return_offsets_mapping=True).to(device)
            
            # Forward
            outputs = llm(inputs.input_ids, attention_mask=inputs.attention_mask)
            
            # Compute Metrics (Pass K List)
            batch_metrics = compute_metrics_from_logits(outputs.logits, inputs.input_ids, TARGET_K_LIST)
            
            # Transfer to CPU
            cpu_metrics = {k: v.float().cpu().numpy() for k, v in batch_metrics.items()}
            offset_mapping = inputs.offset_mapping.cpu().numpy()
            input_ids_cpu = inputs.input_ids.cpu().numpy()
            
            del outputs, batch_metrics, inputs
            torch.cuda.empty_cache()

            # Assign back to objects
            for i, p_res in enumerate(batch_objs):
                offsets = offset_mapping[i]
                
                # Find Start of Response
                res_start_idx = 0
                for t_idx, (start, end) in enumerate(offsets):
                    if start >= prompt_len_char and input_ids_cpu[i][t_idx] != tokenizer.pad_token_id:
                        res_start_idx = t_idx
                        break
                
                # Mask
                valid_mask = (input_ids_cpu[i] != tokenizer.pad_token_id)
                valid_mask[:res_start_idx] = False
                
                # 1. 提取基础 Logprob & Perplexity
                res_lp = cpu_metrics["logprobs"][i][valid_mask]
                p_res["perplexity"] = float(np.exp(-np.mean(res_lp))) if len(res_lp) > 0 else 0.0
                
                # 2. 提取 Full Entropy
                res_ent_full = cpu_metrics["entropy_full"][i][valid_mask]
                p_res.update(aggregate_metrics(res_ent_full, "entropy_full"))

                # 3. 提取各个 Top-K Entropy (重点！)
                for k in TARGET_K_LIST:
                    key = f"entropy_top{k}"
                    res_ent_k = cpu_metrics[key][i][valid_mask]
                    # 我们这里保存 avg 和 std
                    # std_entropy_topK 是后续分析的核心
                    p_res.update(aggregate_metrics(res_ent_k, f"entropy_top{k}"))

        # Save record
        final_record = {
            "problem": problem_text,
            "answer": ground_truth,
            "responses": processed_responses,
            "sc_score": sc_score
        }
        f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")

    f_out.close()
    print(f"Detailed Top-K metrics saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)