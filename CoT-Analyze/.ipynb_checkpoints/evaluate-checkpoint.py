"""
python evaluate.py --input_file qwen64.jsonl --output_file analyze.jsonl --model_path Qwen/Qwen2.5-Math-1.5B --max_samples 1
python evaluate.py --input_file 50step64.jsonl --output_file analyze50step.jsonl --model_path Qwen/Qwen2.5-Math-1.5B
python evaluate.py --input_file amc.jsonl --output_file analyze.jsonl --model_path Qwen/Qwen2.5-Math-1.5B
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

# =============================================================================
# 1. 答案提取与清洗工具 (User Provided Logic)
# =============================================================================

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
# 3. 核心指标计算逻辑
# =============================================================================

def compute_metrics_from_logits(logits: torch.Tensor, input_ids: torch.Tensor):
    """
    计算 Entropy, LogProb, Certainty 等基础 Token 级指标。
    """
    # Shift logits and labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 1. LogProbs & Probs
    # 注意：softmax 在 vocab 维度进行，Qwen vocab 很大，这步最耗显存
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    probs_all = F.softmax(shift_logits, dim=-1)
    
    # 2. Actual Token LogProb
    token_logprobs = torch.gather(log_probs_all, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # 3. Entropy
    entropy = -(probs_all * torch.log(probs_all + 1e-12)).sum(dim=-1)
    
    # 4. Certainty
    V = shift_logits.size(-1)
    logprob_sum = log_probs_all.sum(dim=-1)
    V_tensor = torch.tensor(V, dtype=shift_logits.dtype, device=shift_logits.device)
    custom_certainty = -1 / V_tensor * logprob_sum - torch.log(V_tensor)
    
    # Pad prefix to align with input_ids length
    def pad_prefix(tensor, val=0.0):
        prefix = torch.full((tensor.size(0), 1), val, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([prefix, tensor], dim=1)

    return {
        "logprobs": pad_prefix(token_logprobs, 0.0),
        "entropy": pad_prefix(entropy, 0.0),
        "certainty": pad_prefix(custom_certainty, 0.0)
    }

def aggregate_metrics(values: np.ndarray, prefix: str, include_min_max: bool = True) -> Dict[str, float]:
    """
    Helper to compute avg, std (and optionally min, max) for a list of values.
    Returns e.g. {"avg_logprob": -0.5, "std_logprob": ...}
    """
    if len(values) == 0:
        result = {
            f"avg_{prefix}": 0.0,
            f"std_{prefix}": 0.0
        }
        if include_min_max:
            result[f"min_{prefix}"] = 0.0
            result[f"max_{prefix}"] = 0.0
        return result
    
    result = {
        f"avg_{prefix}": float(np.mean(values)),
        f"std_{prefix}": float(np.std(values))
    }
    if include_min_max:
        result[f"min_{prefix}"] = float(np.min(values))
        result[f"max_{prefix}"] = float(np.max(values))
    return result

# =============================================================================
# 4. 主流程
# =============================================================================

@torch.no_grad()
def main(args):
    # --- Load Model ---
    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # 使用 torch_dtype 而不是 dtype，以兼容不同的库版本
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm.eval()

    # --- Load Data ---
    print(f"Reading data from {args.input_file}...")
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
            
    if args.max_samples:
        data = data[:args.max_samples]

    f_out = open(args.output_file, 'w', encoding='utf-8')
    print("Starting Detailed Metric Calculation...")
    
    for item in tqdm.tqdm(data):
        problem_text = item.get("problem", "") or item.get("question", "")
        ground_truth = item.get("answer", "") or item.get("solution", "")
        responses = item.get("responses", [])
        
        if not responses:
            continue
            
        # --- Pre-processing: Extract Answers & SC Score ---
        # 重构 responses 列表，保留原始文本，计算 extraction 和 correctness
        processed_responses = []
        extracted_answers_list = []
        
        for idx, r_text in enumerate(responses):
            ext_ans = extract_model_answer(r_text)
            # 使用 clean_latex_format 进行清洗后再对比
            is_corr = is_correct_answer(clean_latex_format(ext_ans), clean_latex_format(ground_truth))
            
            # 保存 extraction 结果供后续 SC 计算
            if ext_ans is not None:
                extracted_answers_list.append(strip_string(ext_ans))
            else:
                extracted_answers_list.append(None)

            processed_responses.append({
                "response_id": idx,
                "raw_text": r_text,
                "extracted_answer": ext_ans,
                "score": 1.0 if is_corr else 0.0,
                # Metrics placeholders
                "steps": [] 
            })

        # --- Calculate Self-Consistency ---
        valid_answers = [a for a in extracted_answers_list if a]
        sc_answer = None
        sc_score = 0.0
        if valid_answers:
            c = Counter(valid_answers)
            most_common = c.most_common(1)
            if most_common:
                sc_answer, count = most_common[0]
                sc_score = count / len(responses)

        # --- Model Forward Pass & Metric Calculation (Batched) ---
        prompt_template = f"Question: {problem_text}\nAnswer the question step by step.\nAnswer:"
        
        # [MODIFIED] Batching logic to prevent OOM
        batch_size = args.batch_size
        num_responses = len(processed_responses)
        
        for batch_start_idx in range(0, num_responses, batch_size):
            batch_end_idx = min(batch_start_idx + batch_size, num_responses)
            batch_processed_objs = processed_responses[batch_start_idx:batch_end_idx]
            
            full_texts = [prompt_template + r["raw_text"] for r in batch_processed_objs]
            
            # Tokenize
            inputs = tokenizer(
                full_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                return_offsets_mapping=True,
                max_length=2048
            ).to(device)
            
            # Forward
            outputs = llm(inputs.input_ids, attention_mask=inputs.attention_mask)
            batch_metrics = compute_metrics_from_logits(outputs.logits, inputs.input_ids)
            
            # Move to CPU immediately
            all_logprobs = batch_metrics['logprobs'].float().cpu().numpy()
            all_entropies = batch_metrics['entropy'].float().cpu().numpy()
            all_certainties = batch_metrics['certainty'].float().cpu().numpy()
            
            offset_mapping = inputs.offset_mapping.cpu().numpy()
            input_ids_cpu = inputs.input_ids.cpu().numpy()
            prompt_len_char = len(prompt_template)
            
            # Clear GPU memory
            del outputs, batch_metrics, inputs
            torch.cuda.empty_cache()

            # Assign metrics back to processed_responses
            # 注意：这里的循环变量 i 是 batch 内的相对索引
            for i, p_res in enumerate(batch_processed_objs):
                offsets = offset_mapping[i]
                
                # Find Response Start Index (Skip Prompt & Padding)
                res_start_idx = 0
                for t_idx, (start, end) in enumerate(offsets):
                    if start >= prompt_len_char and input_ids_cpu[i][t_idx] != tokenizer.pad_token_id:
                        res_start_idx = t_idx
                        break
                
                # Mask for valid response tokens
                valid_mask = (input_ids_cpu[i] != tokenizer.pad_token_id)
                valid_mask[:res_start_idx] = False
                
                # Extract Sequence Metrics
                res_lp = all_logprobs[i][valid_mask]
                res_ent = all_entropies[i][valid_mask]
                res_cert = all_certainties[i][valid_mask]
                
                # 1. Sequence Level Stats
                # EXCLUDE min/max for logprob
                p_res.update(aggregate_metrics(res_lp, "logprob", include_min_max=False))
                p_res.update(aggregate_metrics(res_ent, "entropy"))
                p_res.update(aggregate_metrics(res_cert, "certainty"))
                # PPL
                p_res["perplexity"] = float(np.exp(-p_res["avg_logprob"])) if len(res_lp) > 0 else 0.0

                # 2. Step Level Stats
                steps_data = parse_structured_steps(p_res["raw_text"])
                steps_list = []
                
                for s_id, (s_start, s_end, s_content) in enumerate(steps_data):
                    abs_start = prompt_len_char + s_start
                    abs_end = prompt_len_char + s_end
                    
                    # Identify tokens belonging to this step
                    step_indices = []
                    for t_idx, (os, oe) in enumerate(offsets):
                        if t_idx < res_start_idx: continue
                        if input_ids_cpu[i][t_idx] == tokenizer.pad_token_id: continue
                        # Token overlaps with step char range
                        if os < abs_end and oe > abs_start:
                            step_indices.append(t_idx)
                    
                    # Compute Step Metrics
                    step_obj = {
                        "step_id": s_id,
                        "text": s_content
                    }
                    
                    if step_indices:
                        s_lp = all_logprobs[i][step_indices]
                        s_ent = all_entropies[i][step_indices]
                        s_cert = all_certainties[i][step_indices]
                        
                        # EXCLUDE min/max for logprob here too
                        step_obj.update(aggregate_metrics(s_lp, "logprob", include_min_max=False))
                        step_obj.update(aggregate_metrics(s_ent, "entropy"))
                        step_obj.update(aggregate_metrics(s_cert, "certainty"))
                        step_obj["perplexity"] = float(np.exp(-step_obj["avg_logprob"]))
                    else:
                        # Empty step (no tokens found)
                        step_obj.update(aggregate_metrics([], "logprob", include_min_max=False))
                        step_obj.update(aggregate_metrics([], "entropy"))
                        step_obj.update(aggregate_metrics([], "certainty"))
                        step_obj["perplexity"] = 0.0

                    steps_list.append(step_obj)
                
                p_res["steps"] = steps_list

        # --- Construct Final Record ---
        final_record = {
            "problem": problem_text,
            "answer": ground_truth,
            "responses": processed_responses,
            "scoring": {
                "self_consistency_answer": sc_answer,
                "self_consistency_score": sc_score
            }
        }
        
        f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")

    f_out.close()
    print(f"Done! Detailed metrics saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--max_samples", type=int, default=None)
    # 增加 batch_size 参数，默认设为 4
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing responses")
    
    args = parser.parse_args()
    main(args)