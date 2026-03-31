#!/usr/bin/env python3
"""将数据集划分为训练集、验证集和测试集"""
import pandas as pd
import argparse

def split_dataset(input_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    划分数据集
    
    Args:
        input_file: 输入parquet文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    """
    # 验证比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须等于1.0，当前为 {train_ratio + val_ratio + test_ratio}")
    
    print(f"正在读取数据集: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"总数据量: {len(df)}")
    
    # 显示原始数据分布
    print("\n原始数据分布:")
    attack_type = df['extra_info'].apply(lambda x: x.get('attack_type', '') if isinstance(x, dict) else '')
    type_counts = attack_type.value_counts()
    print(type_counts.to_string())
    
    # 随机打乱（如果还没有打乱的话）
    print(f"\n使用随机种子 {random_state} 划分数据集...")
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 计算划分点
    total = len(df_shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 划分数据集
    train_df = df_shuffled.iloc[:train_end].reset_index(drop=True)
    val_df = df_shuffled.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df_shuffled.iloc[val_end:].reset_index(drop=True)
    
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/total*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 条 ({len(val_df)/total*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/total*100:.1f}%)")
    
    # 显示各数据集的类型分布
    print("\n训练集类型分布:")
    train_type = train_df['extra_info'].apply(lambda x: x.get('attack_type', '') if isinstance(x, dict) else '')
    print(train_type.value_counts().to_string())
    
    print("\n验证集类型分布:")
    val_type = val_df['extra_info'].apply(lambda x: x.get('attack_type', '') if isinstance(x, dict) else '')
    print(val_type.value_counts().to_string())
    
    print("\n测试集类型分布:")
    test_type = test_df['extra_info'].apply(lambda x: x.get('attack_type', '') if isinstance(x, dict) else '')
    print(test_type.value_counts().to_string())
    
    # 保存数据集
    train_file = input_file.replace('.parquet', '_train.parquet')
    val_file = input_file.replace('.parquet', '_val.parquet')
    test_file = input_file.replace('.parquet', '_test.parquet')
    
    print(f"\n正在保存数据集...")
    train_df.to_parquet(train_file, index=False, engine='pyarrow')
    print(f"  训练集已保存: {train_file}")
    
    val_df.to_parquet(val_file, index=False, engine='pyarrow')
    print(f"  验证集已保存: {val_file}")
    
    test_df.to_parquet(test_file, index=False, engine='pyarrow')
    print(f"  测试集已保存: {test_file}")
    
    print("\n完成！")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="划分数据集为训练集、验证集和测试集")
    parser.add_argument(
        "--input",
        type=str,
        default="MAMO_easy.parquet",
        help="输入parquet文件路径"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认0.8）"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="验证集比例（默认0.1）"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="测试集比例（默认0.1）"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子（默认42）"
    )
    
    args = parser.parse_args()
    
    split_dataset(
        args.input,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_state
    )
