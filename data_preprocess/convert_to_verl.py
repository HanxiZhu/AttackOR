#!/usr/bin/env python3
"""
将攻击数据集转换为verl框架标准格式
- 读取JSON数据集
- 调用OpenAI API对比baseline和attack题目，提取排雷逻辑
- 转换为verl格式
- 保存为parquet文件
"""

import json
import os
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import pandas as pd
from tqdm import tqdm


# API 配置（推荐通过环境变量提供，不在代码中硬编码）
# OPENAI_API_BASE: OpenAI 兼容 API base URL（默认官方）
# OPENAI_API_KEY: API key（必须由用户提供）
# OPENAI_MODEL: 模型名
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2")

# 初始化OpenAI客户端（延迟初始化，在需要时创建）
_client = None

def get_client() -> OpenAI:
    """获取OpenAI客户端（单例模式）"""
    global _client
    if _client is None:
        if not API_KEY:
            raise ValueError("缺少 OPENAI_API_KEY：请设置环境变量 OPENAI_API_KEY 或在运行前注入密钥")
        _client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )
    return _client


def extract_defense_logic(baseline_question: str, attack_question: str, max_retries: int = 3) -> str:
    """
    Call OpenAI API to compare baseline and attack questions, extract defense logic
    
    Args:
        baseline_question: baseline question
        attack_question: attack question
        max_retries: maximum retry attempts
    
    Returns:
        One-sentence summary of defense logic
    """
    prompt = f"""Compare the following two optimization problems. Identify the specific interference injected in the attack question and summarize the correct defense logic in one sentence.

Baseline Question:
{baseline_question}

Attack Question:
{attack_question}

Please analyze:
1. What interference was injected in the attack question?
2. What is the correct defense logic to handle this interference? (Summarize in one sentence)

Return only the one-sentence defense logic summary, nothing else."""
    
    for attempt in range(max_retries):
        try:
            client = get_client()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            defense_logic = response.choices[0].message.content.strip()
            return defense_logic
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避
                time.sleep(wait_time)
                continue
            else:
                print(f"API call failed after {max_retries} retries: {e}")
                return "Failed to extract defense logic"  # Default value
    
    return "Failed to extract defense logic"


def get_baseline_defense_logic(question: str) -> str:
    """
    Generate defense_logic for baseline questions (normal solving)
    
    Args:
        question: baseline question
    
    Returns:
        Normal solving prompt
    """
    return "Solve the optimization problem normally without special defense logic"


def process_single_item(item: Dict, baseline_map: Dict[str, Dict], 
                       executor: Optional[ThreadPoolExecutor] = None) -> Dict:
    """
    处理单个数据项，转换为verl格式
    
    Args:
        item: 原始数据项
        baseline_map: baseline映射 {index: baseline_item}
        executor: 线程池执行器（用于并发API调用）
    
    Returns:
        verl格式的数据项
    """
    item_id = item["id"]
    item_type = item["type"]
    question = item["question"]
    answer = item["answer"]
    
    # 提取index（id的前缀部分）
    index = item_id.split("_", 1)[0]  # 从左边分割，确保正确提取index
    
    # 转换为float类型
    try:
        answer_float = float(answer)
    except (ValueError, TypeError):
        answer_float = 0.0
    
    # 构建verl格式
    verl_item = {
        "data_source": "Attack_OR",
        "prompt": [{"role": "user", "content": question}],
        "ability": "Attack_OR",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer_float
        },
        "extra_info": {
            "attack_type": item_type,
            "defense_logic": "",  # 稍后填充
            "index": item_id
        }
    }
    
    # 获取defense_logic
    if item_type == "baseline":
        defense_logic = get_baseline_defense_logic(question)
    else:
        # 找到对应的baseline
        baseline_item = baseline_map.get(index)
        if baseline_item:
            baseline_question = baseline_item["question"]
            # 如果提供了executor，使用异步调用
            if executor:
                future = executor.submit(extract_defense_logic, baseline_question, question)
                verl_item["_future"] = future
            else:
                defense_logic = extract_defense_logic(baseline_question, question)
                verl_item["extra_info"]["defense_logic"] = defense_logic
        else:
            defense_logic = "Baseline question not found"
            verl_item["extra_info"]["defense_logic"] = defense_logic
    
    if "_future" not in verl_item:
        verl_item["extra_info"]["defense_logic"] = defense_logic
    
    return verl_item


def load_data(json_path: str) -> List[Dict]:
    """加载JSON数据集"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_baseline_map(data: List[Dict]) -> Dict[str, Dict]:
    """建立baseline映射 {index: baseline_item}"""
    baseline_map = {}
    for item in data:
        if item["type"] == "baseline":
            index = item["id"].split("_", 1)[0]  # 从左边分割，确保正确提取index
            baseline_map[index] = item
    return baseline_map


def convert_to_verl_format(json_path: str, output_path: str, max_workers: int = 10):
    """
    主函数：将数据集转换为verl格式并保存为parquet
    
    Args:
        json_path: 输入JSON文件路径
        output_path: 输出parquet文件路径
        max_workers: 并发API调用的最大线程数
    """
    print("正在加载数据...")
    data = load_data(json_path)
    print(f"已加载 {len(data)} 条数据")
    
    print("正在建立baseline映射...")
    baseline_map = build_baseline_map(data)
    print(f"找到 {len(baseline_map)} 个baseline题目")
    
    print("正在处理数据并调用API...")
    verl_data = []
    
    # 分离baseline和attack题目
    baseline_items = [item for item in data if item["type"] == "baseline"]
    attack_items = [item for item in data if item["type"] != "baseline"]
    
    print(f"Baseline题目: {len(baseline_items)} 条")
    print(f"攻击题目: {len(attack_items)} 条")
    
    # 先处理baseline题目（不需要API调用）
    print("\n处理baseline题目...")
    for item in tqdm(baseline_items, desc="Baseline"):
        verl_item = process_single_item(item, baseline_map)
        verl_data.append(verl_item)
    
    # 使用线程池并发处理attack题目
    print("\n处理攻击题目（并发调用API）...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {}
        for item in attack_items:
            verl_item = process_single_item(item, baseline_map, executor)
            if "_future" in verl_item:
                future = verl_item.pop("_future")
                futures[future] = verl_item
            else:
                verl_data.append(verl_item)
        
        # 等待所有API调用完成
        for future in tqdm(as_completed(futures), total=len(futures), desc="API调用"):
            verl_item = futures[future]
            try:
                defense_logic = future.result()
                verl_item["extra_info"]["defense_logic"] = defense_logic
            except Exception as e:
                print(f"API call error: {e}")
                verl_item["extra_info"]["defense_logic"] = "API call failed"
            verl_data.append(verl_item)
    
    print(f"\n已处理 {len(verl_data)} 条数据")
    
    # 转换为DataFrame并保存为parquet
    print("正在保存为parquet文件...")
    df = pd.DataFrame(verl_data)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"已保存到: {output_path}")
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"总数据量: {len(df)}")
    print(f"攻击类型分布:")
    if len(df) > 0:
        attack_type_counts = df['extra_info'].apply(lambda x: x.get('attack_type', 'unknown') if isinstance(x, dict) else 'unknown').value_counts()
        print(attack_type_counts.to_string())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将攻击数据集转换为verl格式")
    parser.add_argument(
        "--input",
        type=str,
        default="MAMO_easy_lp_adversarial_merged.jsonl",
        help="输入JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="MAMO_easy.parquet",
        help="输出parquet文件路径"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="并发API调用的最大线程数"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI API基础URL（覆盖环境变量）"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API密钥（覆盖环境变量）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="使用的模型名称（覆盖环境变量）"
    )
    
    args = parser.parse_args()
    
    # 如果提供了命令行参数，更新配置
    if args.api_base:
        globals()['API_BASE_URL'] = args.api_base
    if args.api_key:
        globals()['API_KEY'] = args.api_key
    if args.model:
        globals()['MODEL'] = args.model
    # 重置客户端以使用新配置
    globals()['_client'] = None
    
    convert_to_verl_format(args.input, args.output, args.workers)
