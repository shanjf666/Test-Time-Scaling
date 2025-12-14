"""
python self-consistency.py --model_path Qwen/Qwen2.5-Math-1.5B --output_file qwen64.jsonl --num_return_sequences 64
python self-consistency.py --model_path meta-llama/Llama-3.2-1B-Instruct --output_file llama64.jsonl --num_return_sequences 64
python self-consistency.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/ttrl_step_50_qwen_1.5B --output_file 50step64.jsonl --num_return_sequences 64
python self-consistency.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/consistency_step_80 --output_file scresults/consistency_step_80_bo64.jsonl --num_return_sequences 64
python self-consistency.py --model_path Qwen/Qwen2.5-Math-1.5B --output_file try64_1.jsonl --num_return_sequences 64
python self-consistency.py --model_path Qwen/Qwen2.5-Math-1.5B --dataset_name AI-MO/aimo-validation-amc --split train --output_file amc.jsonl --num_return_sequences 64
python self-consistency.py --model_path Qwen/Qwen2.5-Math-1.5B --output_file qwen64.jsonl --num_return_sequences 64 --dataset_name  --split   --max_samples
"""
import json
import torch
import re
import argparse
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# =====================================================
# 1. 工具函数 (清洗、提取、标准化)
# =====================================================

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
    if s is None:
        return None
    if "\\boxed " in s:
        return s.replace("\\boxed ", "")
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    return s

def strip_string(string):
    if string is None:
        return ""
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
    """
    从模型输出中提取最终答案（去掉 \\boxed{}）。
    如果没有找到 \\boxed，返回 None。
    """
    boxed_part = last_boxed_only_string(response_text)
    if boxed_part is None:
        return None
    answer = remove_boxed(boxed_part)
    return answer.strip()

# =====================================================
# 2. 构造 prompt 的工厂函数
# =====================================================
def get_prompt_builder(model_path, tokenizer):
    name = model_path.lower()

    # Chat-template 模型
    if "instruct" in name or "llama" in name:
        def build_prompt(problem: str):
            return tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"}],
                tokenize=False,
                add_generation_prompt=True
            )
        return build_prompt

    # DeepSeek
    if "deepseek" in name:
        def build_prompt(problem: str):
            return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
        return build_prompt

    # 普通 Qwen
    if "qwen" in name:
        def build_prompt(problem: str):
            return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
        return build_prompt

    # 默认
    def build_prompt(problem: str):
        return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
    return build_prompt


# =====================================================
# 3. 主流程
# =====================================================
def main(args):
    # ----------------------------
    # 加载 tokenizer
    # ----------------------------
    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # ----------------------------
    # 创建 prompt 构造器
    # ----------------------------
    build_prompt = get_prompt_builder(args.model_path, tokenizer)

    # ----------------------------
    # 加载数据集
    # ----------------------------
    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Total problems to process: {len(dataset)}")

    # ----------------------------
    # 构造 prompts
    # ----------------------------
    prompts = [build_prompt(item["problem"]) for item in dataset]

    # ----------------------------
    # 生成参数
    # ----------------------------
    sampling_params = SamplingParams(
        n=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["<|eot_id|>", "</s>", "Q:"],
    )

    # ----------------------------
    # 初始化 vLLM
    # ----------------------------
    print(f"Initializing vLLM engine for model {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        trust_remote_code=True,
        dtype="auto",
    )

    # ----------------------------
    # 开始生成
    # ----------------------------
    print("Starting generation...")
    request_outputs = llm.generate(prompts, sampling_params)

    # ----------------------------
    # 保存输出 (包含 Self-Consistency 计算)
    # ----------------------------
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for i, request_output in enumerate(request_outputs):
            original_item = dataset[i]
            
            # 获取所有生成的文本
            responses = [out.text for out in request_output.outputs]
            
            # --- Self-Consistency 计算逻辑开始 ---
            extracted_answers = []
            
            for resp in responses:
                # 1. 提取 \boxed{} 内容
                raw_ans = extract_model_answer(resp)
                
                # 2. 标准化字符串 (去除空格、格式化 latex)
                # 使用 strip_string 确保例如 "5" 和 " 5 " 被视为同一个答案
                # 如果没有提取到答案，标记为 "[NO_ANSWER]" 以便统计
                if raw_ans is not None:
                    norm_ans = strip_string(raw_ans)
                else:
                    norm_ans = "[NO_ANSWER]"
                
                extracted_answers.append(norm_ans)
            
            # 3. 统计频率
            counter = Counter(extracted_answers)
            # 获取出现次数最多的 (answer, count)
            most_common = counter.most_common(1)
            
            if most_common:
                best_norm_answer, count = most_common[0]
    
                sc_answer = best_norm_answer
                sc_score = count / len(responses)
                
            else:
                sc_answer = None
                sc_score = 0.0
            # --- Self-Consistency 计算逻辑结束 ---

            record = {
                "problem": original_item["problem"],
                "answer": original_item.get("answer", original_item.get("solution", None)),
                "responses": responses,
                "extracted_answers": extracted_answers, # 可选：保存所有提取出的答案用于调试
                "sc_answer": sc_answer,                 # 最终选定的答案
                "sc_score": sc_score                    # 自洽性分数 (0.0 - 1.0)
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ----------------------------
    # 输出生成参数信息
    # ----------------------------
    print("=== Generation Finished ===")
    print(f"model_path       : {args.model_path}")
    print(f"max_tokens       : {args.max_tokens}")
    print(f"temperature      : {args.temperature}")
    print(f"num_return_sequences : {args.num_return_sequences}")
    print("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 高效生成 N 条候选回答（chat-template 版 + Self-Consistency）")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="math500_candidates.jsonl")
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)