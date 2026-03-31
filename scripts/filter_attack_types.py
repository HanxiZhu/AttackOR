#!/usr/bin/env python3
"""
过滤 attack_or 数据集，根据 attack_type 选择特定样本用于消融实验

用法:
    python scripts/filter_attack_types.py --attack_types unit_conversion,llm_attribute
    python scripts/filter_attack_types.py --attack_types "unit_conversion,llm_attribute,baseline" --output_dir ./filtered_data
"""

import argparse
import os
from collections import Counter

import pandas as pd


def filter_by_attack_types(
    input_path: str,
    output_path: str,
    attack_types: list[str],
    verbose: bool = True,
) -> int:
    """
    根据 attack_type 过滤数据

    Args:
        input_path: 输入 parquet 文件路径
        output_path: 输出 parquet 文件路径
        attack_types: 允许的 attack_type 列表
        verbose: 是否打印详细信息

    Returns:
        过滤后的样本数量
    """
    # 读取数据
    df = pd.read_parquet(input_path)

    if verbose:
        print(f"加载数据: {input_path}")
        print(f"原始样本数: {len(df)}")

        # 统计原始 attack_type 分布
        attack_type_counts = Counter()
        for _, row in df.iterrows():
            ei = row.get("extra_info", {})
            if isinstance(ei, dict):
                attack_type_counts[ei.get("attack_type", "unknown")] += 1
        print(f"原始 attack_type 分布:")
        for k, v in sorted(attack_type_counts.items()):
            print(f"  {k}: {v}")

    # 过滤数据
    def should_keep(row):
        ei = row.get("extra_info", {})
        if not isinstance(ei, dict):
            return False
        attack_type = ei.get("attack_type")
        return attack_type in attack_types

    mask = df.apply(should_keep, axis=1)
    filtered_df = df[mask]

    if verbose:
        print(f"\n过滤后样本数: {len(filtered_df)}")
        # 统计过滤后的 attack_type 分布
        filtered_counts = Counter()
        for _, row in filtered_df.iterrows():
            ei = row.get("extra_info", {})
            if isinstance(ei, dict):
                filtered_counts[ei.get("attack_type", "unknown")] += 1
        print(f"过滤后 attack_type 分布:")
        for k, v in sorted(filtered_counts.items()):
            print(f"  {k}: {v}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存过滤后的数据
    filtered_df.to_parquet(output_path, index=False)
    if verbose:
        print(f"\n保存到: {output_path}")

    return len(filtered_df)


def main():
    parser = argparse.ArgumentParser(
        description="过滤 attack_or 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 过滤 unit_conversion 和 llm_attribute
    python scripts/filter_attack_types.py --attack_types unit_conversion,llm_attribute

    # 指定输出目录
    python scripts/filter_attack_types.py --attack_types unit_conversion,llm_attribute --output_dir ./filtered_data

    # 使用自定义后缀
    python scripts/filter_attack_types.py --attack_types unit_conversion --output_suffix "_uc_only"

可用的 attack_type:
    - unit_conversion
    - llm_attribute
    - llm_phantom
    - arithmetic
    - baseline
        """,
    )

    parser.add_argument(
        "--attack_types",
        type=str,
        required=True,
        help="要保留的 attack_type，用逗号分隔，如: unit_conversion,llm_attribute",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="输入数据目录，默认为项目根目录下的 data 目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出数据目录，默认与输入目录相同",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="输出文件名后缀，默认自动根据 attack_types 生成，如: _unit_conversion_llm_attribute",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="merged_train.parquet",
        help="训练数据文件名",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="merged_val.parquet",
        help="验证数据文件名",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，不打印详细信息",
    )

    args = parser.parse_args()

    # 解析 attack_types
    attack_types = [t.strip() for t in args.attack_types.split(",")]

    # 确定项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 确定输入输出目录
    input_dir = args.input_dir or os.path.join(project_root, "data")
    output_dir = args.output_dir or input_dir

    # 生成输出文件后缀
    if args.output_suffix:
        suffix = args.output_suffix
    else:
        suffix = "_" + "_".join(attack_types)

    # 处理训练数据
    train_input = os.path.join(input_dir, args.train_file)
    train_output = os.path.join(
        output_dir,
        args.train_file.replace(".parquet", f"{suffix}.parquet"),
    )

    print("=" * 60)
    print(f"过滤 attack_types: {attack_types}")
    print("=" * 60)

    print("\n[训练数据]")
    filter_by_attack_types(
        train_input,
        train_output,
        attack_types,
        verbose=not args.quiet,
    )

    # 处理验证数据
    val_input = os.path.join(input_dir, args.val_file)
    if os.path.exists(val_input):
        val_output = os.path.join(
            output_dir,
            args.val_file.replace(".parquet", f"{suffix}.parquet"),
        )
        print("\n[验证数据]")
        filter_by_attack_types(
            val_input,
            val_output,
            attack_types,
            verbose=not args.quiet,
        )

    print("\n" + "=" * 60)
    print("过滤完成!")
    print(f"训练数据: {train_output}")
    if os.path.exists(val_input):
        print(f"验证数据: {val_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
