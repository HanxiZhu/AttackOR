#!/usr/bin/env python3
"""打乱parquet文件的数据顺序"""
import pandas as pd

# 读取parquet文件
print("正在读取parquet文件...")
df = pd.read_parquet('MAMO_easy.parquet')
print(f"原始数据量: {len(df)}")

# 显示打乱前的顺序（前10条）
print("\n打乱前的顺序（前10条数据的ID）:")
for i in range(min(10, len(df))):
    idx = df.iloc[i]['extra_info']['index']
    atype = df.iloc[i]['extra_info']['attack_type']
    print(f"  {i}: {idx} ({atype})")

# 随机打乱数据顺序
print("\n正在随机打乱数据顺序...")
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存回原文件
print("正在保存打乱后的数据...")
df_shuffled.to_parquet('MAMO_easy.parquet', index=False, engine='pyarrow')
print("完成！数据已打乱并保存。")

# 验证打乱效果
print("\n打乱后的顺序（前10条数据的ID）:")
for i in range(min(10, len(df_shuffled))):
    idx = df_shuffled.iloc[i]['extra_info']['index']
    atype = df_shuffled.iloc[i]['extra_info']['attack_type']
    print(f"  {i}: {idx} ({atype})")

# 统计打乱后的类型分布（验证数据完整性）
print("\n打乱后的类型分布:")
attack_type = df_shuffled['extra_info'].apply(lambda x: x.get('attack_type', '') if isinstance(x, dict) else '')
type_counts = attack_type.value_counts()
print(type_counts.to_string())
