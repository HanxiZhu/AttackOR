import argparse
import json
import os
import re
import sys
import datasets

from vllm import LLM, SamplingParams
from tqdm import tqdm


import transformers

# === 针对 vLLM Nightly 多模态 Bug 的终极补丁 ===
class DummyVisionConfig:
    spatial_merge_size = 2
    spatial_patch_size = 14
    temporal_patch_size = 2

# 强行把假配置挂载到基类上，完美欺骗 vLLM 的初始化检查
transformers.PretrainedConfig.vision_config = DummyVisionConfig()
# ============================================

# === 专门骗过 vLLM mRoPE 位置编码计算的补丁 ===
# 既然它一定要找这些多模态特殊 token，我们就随便给它几个整数
transformers.PretrainedConfig.video_token_id = 151654
transformers.PretrainedConfig.image_token_id = 151655
transformers.PretrainedConfig.vision_start_token_id = 151652
transformers.PretrainedConfig.vision_end_token_id = 151653



TEMPLATE_q2mc_en = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
"""

def load_data_robust(filepath):
    """
    鲁棒的数据加载函数：
    1. 优先尝试作为一个整体 JSON List 读取 (兼容标准 JSON 和 缩进格式)
    2. 如果失败，尝试作为 JSONL (逐行读取)
    """
    print(f"正在加载文件: {filepath} ...")
    data = []
    
    # 尝试方法 A: 整体读取 (适用于标准 JSON List, 即使后缀写错成 jsonl)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f) # 尝试一次性解析整个文件
            if isinstance(content, list):
                print(f" -> 识别为标准 JSON 列表格式 (List)，加载成功。")
                return content
            elif isinstance(content, dict):
                print(f" -> 识别为单条 JSON 对象 (Dict)，加载成功。")
                return [content]
    except json.JSONDecodeError:
        # 如果整体解析失败，说明可能是真正的 JSONL 或者格式有问题，进入方法 B
        pass
    except Exception as e:
        print(f"Warning: 整体读取尝试失败 ({e})，尝试逐行读取...")

    # 尝试方法 B: 逐行读取 (适用于标准的 JSONL / NDJSON)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {i+1}: {e}")
                    # 如果前几行就报错，通常说明文件完全不对，直接抛出
                    if i < 5: raise e
        print(f" -> 识别为 JSONL (逐行) 格式，加载成功。")
        return data
    except Exception as e:
        raise ValueError(f"无法解析数据文件 {filepath}。请检查文件是否为有效的 JSON List 或 JSONL 格式。\n详细错误: {e}")

def main(args):
    # 基础参数校验
    assert isinstance(args.topk, int)
    assert args.decoding_method in ["greedy", "sampling"]
    
    if args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required!")
    assert os.path.exists(args.model_name_or_path), f"Model path does not exist: {args.model_name_or_path}"
    assert args.save_dir is not None

    os.makedirs(args.save_dir, exist_ok=True)

    # ==========================
    # 1. 更加鲁棒的数据加载逻辑
    # ==========================
    sample = []
    
    # 优先加载本地文件
    if args.dataset_name and os.path.exists(args.dataset_name):
        sample = load_data_robust(args.dataset_name)
    else:
        # Fallback 到 HuggingFace
        if args.dataset_split is None:
             # 如果本地没找到，也没给 split，就报错
            if args.dataset_name is None:
                raise ValueError("请提供 --dataset_name (本地路径或HF数据集名)")
            print(f"本地未找到文件 '{args.dataset_name}'，尝试从 HuggingFace 加载...")
            if args.dataset_split is None:
                 raise ValueError("从 HuggingFace 加载需要指定 --dataset_split")
        
        ds = datasets.load_dataset(args.dataset_name)
        ds = ds[args.dataset_split]
        for example in ds:
            sample.append(example)
            
    print(f"总计加载样本数: {len(sample)}")
    
    # ==========================
    # 2. 数据预处理与Prompt构建
    # ==========================
    processed_sample = []
    skipped_count = 0
    
    for example in sample:
        # 兼容攻击数据集 ('question') 和 原始数据集 ('en_question')
        q_text = example.get("en_question")
        if q_text is None:
            q_text = example.get("question")
            
        if q_text is None:
            skipped_count += 1
            continue

        prompt = TEMPLATE_q2mc_en.replace("{Question}", q_text.strip()).strip()
        
        # 保留原始 metadata (id, type, answer 等)
        example_t = {k: v for k, v in example.items() if k not in ["prompt"]}
        example_t["prompt"] = prompt
        processed_sample.append(example_t)
    
    if skipped_count > 0:
        print(f"Warning: 跳过了 {skipped_count} 条缺少 question/en_question 字段的样本。")
        
    sample = processed_sample
    if len(sample) == 0:
        raise ValueError("有效样本数为 0，请检查数据文件内容！")

    # ==========================
    # 3. 模型初始化
    # ==========================
    print(f"Loading model from {args.model_name_or_path} ...")
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True,reasoning_parser="qwen3", enable_prefix_caching=True, language_model_only=True)
    print("Init model done.")
    
    stop_tokens = ["</s>"]
    if args.decoding_method == "greedy":
        print(f"Mode: Greedy Decoding (temp=0)")
        sampling_params = SamplingParams(n=args.topk, temperature=0, top_p=1, max_tokens=model.llm_engine.model_config.max_model_len if args.max_tokens is None else args.max_tokens, stop=stop_tokens)
    elif args.decoding_method == "sampling":
        print(f"Mode: Sampling (temp={args.temperature})")
        sampling_params = SamplingParams(n=args.topk, temperature=args.temperature, top_p=args.top_p, max_tokens=model.llm_engine.model_config.max_model_len if args.max_tokens is None else args.max_tokens, stop=stop_tokens)
    else:
        raise ValueError("Invalid decoding method")

    # ==========================
    # 4. 执行推理
    # ==========================
    prompts = [example["prompt"] for example in sample]
    print(f"Starting generation for {len(prompts)} prompts...")
    generations = model.generate(prompts, sampling_params)

    # ==========================
    # 5. 保存结果
    # ==========================
    save_file = os.path.join(args.save_dir, "generated_attacks.jsonl")
    print(f"Saving results to {save_file} ...")
    
    fw = open(save_file, "w", encoding='utf-8')
    num_total = 0
    
    for example, prompt, generation in zip(sample, prompts, generations):
        outputs = generation.outputs
        # 去重逻辑
        unique_outputs = set([o.text for o in outputs])
        
        for output_text in unique_outputs:
            num_total += 1
            
            # 构造结果对象
            example_t = example.copy() # 浅拷贝，保留原始 id, type, answer
            example_t["q2mc_en_prompt"] = prompt
            example_t["en_math_model_coptpy_code"] = output_text
            
            if args.verbose:
                print(f"--- Output for ID {example.get('id', 'N/A')} ---")
                print(output_text[:100] + "...") 

            fw.write(json.dumps(example_t, ensure_ascii=False) + "\n")
            
    fw.close()
    print(f"Generation finished. Total outputs written: {num_total}")

def parse_args():
    parser = argparse.ArgumentParser()
    # model path
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="CardinalOperations/ORLM-LLaMA-3-8B",
        help="Hugging Face model id or local path",
    )
    parser.add_argument("--dataset_name", type=str, default="./datasets_local/NL4OPT_test_adversarial_phantom_llm.jsonl", help="Path to local file or HF dataset name") 
    parser.add_argument("--dataset_split", type=str, default="test", help="Split name if using HF dataset") 
    parser.add_argument("--save_dir", type=str, default="./results_attack_phantom_llm")  
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # num_gpus
    parser.add_argument("--topk", type=int, default=1)  
    parser.add_argument("--temperature", type=float, default=0.7) 
    parser.add_argument("--top_p", type=float, default=0.95) 
    parser.add_argument("--max_tokens", type=int, default=14000) 
    parser.add_argument("--decoding_method", type=str, default="greedy")  
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)