#!/usr/bin/env python3
"""
将verl格式的parquet转换为评测用的json格式
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def convert_to_eval_format(input_path: str, output_path: str):
    """
    将parquet转换为评测json格式
    
    Args:
        input_path: 输入parquet文件
        output_path: 输出json文件
    """
    df = pd.read_parquet(input_path)
    print(f"加载 {len(df)} 条数据")
    
    results = []
    for _, row in df.iterrows():
        # 提取question - 从prompt数组中取content
        question = ""
        prompt = row['prompt']
        if hasattr(prompt, '__iter__') and len(prompt) > 0:
            first_item = list(prompt)[0] if not isinstance(prompt, list) else prompt[0]
            if isinstance(first_item, dict):
                question = first_item.get('content', '')
        elif isinstance(prompt, list) and len(prompt) > 0:
            question = prompt[0].get('content', '')
        
        # 提取answer
        reward_model = row['reward_model']
        if isinstance(reward_model, dict):
            answer = str(int(reward_model.get('ground_truth', 0)))
        else:
            answer = "0"
        
        # 提取id和type
        extra_info = row['extra_info']
        if isinstance(extra_info, dict):
            item_id = extra_info.get('index', '')
            item_type = extra_info.get('attack_type', 'unknown')
        else:
            item_id = ''
            item_type = 'unknown'
        
        results.append({
            "id": item_id,
            "type": item_type,
            "question": question,
            "answer": answer
        })
    
    # 保存为json数组
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"已保存 {len(results)} 条到 {output_path}")
    
    # 统计
    types = {}
    for item in results:
        t = item['type']
        types[t] = types.get(t, 0) + 1
    print(f"\n类型分布:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将parquet转换为评测json")
    parser.add_argument("--input", "-i", default="MAMO_easy_test.parquet", help="输入parquet文件")
    parser.add_argument("--output", "-o", default="MAMO_easy_test.jsonl", help="输出json文件")
    args = parser.parse_args()
    convert_to_eval_format(args.input, args.output)
