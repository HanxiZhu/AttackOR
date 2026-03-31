#!/usr/bin/env python3
"""
合并多个verl格式的parquet数据集，用于训练
"""

import argparse
import pandas as pd
from pathlib import Path


def merge_datasets(input_files: list[str], output_path: str, shuffle: bool = True, seed: int = 42):
    """
    合并多个parquet文件

    Args:
        input_files: 输入parquet文件列表
        output_path: 输出文件路径
        shuffle: 是否打乱数据
        seed: 随机种子
    """
    dfs = []
    total = 0

    for f in input_files:
        p = Path(f)
        if not p.exists():
            print(f"跳过不存在的文件: {f}")
            continue
        df = pd.read_parquet(f)
        dfs.append(df)
        print(f"加载 {p.name}: {len(df)} 条")
        total += len(df)

    if not dfs:
        print("没有有效数据！")
        return

    merged = pd.concat(dfs, ignore_index=True)

    if shuffle:
        merged = merged.sample(frac=1, random_state=seed).reset_index(drop=True)

    merged.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\n合并完成: {total} 条 -> {output_path}")

    # 统计
    if 'extra_info' in merged.columns:
        attack_types = merged['extra_info'].apply(
            lambda x: x.get('attack_type', 'unknown') if isinstance(x, dict) else 'unknown'
        )
        print(f"\nattack_type分布:\n{attack_types.value_counts().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并verl格式数据集")
    parser.add_argument("--inputs", nargs="+", required=True, help="输入parquet文件")
    parser.add_argument("--output", "-o", required=True, help="输出parquet文件")
    parser.add_argument("--no-shuffle", action="store_true", help="不打乱数据")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    merge_datasets(args.inputs, args.output, shuffle=not args.no_shuffle, seed=args.seed)
