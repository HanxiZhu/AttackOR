#!/usr/bin/env python3
"""详细检查parquet文件的数据"""
import pandas as pd
import json

df = pd.read_parquet('MAMO_easy.parquet')

print("=" * 80)
print("数据概览")
print("=" * 80)
print(f"总数据量: {len(df)}")
print(f"列名: {list(df.columns)}")
print()

# 检查defense_logic的情况
defense_logic = df['extra_info'].apply(lambda x: x.get('defense_logic', '') if isinstance(x, dict) else '')
attack_type = df['extra_info'].apply(lambda x: x.get('attack_type', '') if isinstance(x, dict) else '')

print("=" * 80)
print("Defense Logic 统计")
print("=" * 80)
print(f"正常解答（baseline）: {(defense_logic == '正常解答优化问题，无需特殊排雷逻辑').sum()}")
print(f"无法提取/API失败: {(defense_logic.str.contains('无法提取|API调用失败|未找到|Failed', case=False, na=False)).sum()}")
success_count = len(df) - (defense_logic == '正常解答优化问题，无需特殊排雷逻辑').sum() - (defense_logic.str.contains('无法提取|API调用失败|未找到|Failed', case=False, na=False)).sum()
print(f"成功提取: {success_count}")
print()

# 按攻击类型统计
print("=" * 80)
print("按攻击类型统计 Defense Logic")
print("=" * 80)
for atype in attack_type.unique():
    mask = attack_type == atype
    type_df = df[mask]
    type_defense = type_df['extra_info'].apply(lambda x: x.get('defense_logic', '') if isinstance(x, dict) else '')
    success = ~type_defense.str.contains('无法提取|API调用失败|未找到|Failed|正常解答', case=False, na=False)
    print(f"{atype}:")
    print(f"  总数: {len(type_df)}")
    print(f"  成功: {success.sum()}")
    print(f"  失败: {(~success).sum()}")
print()

# 查看一些成功提取的示例
print("=" * 80)
print("成功提取的示例（前3条）")
print("=" * 80)
success_mask = ~defense_logic.str.contains('无法提取|API调用失败|未找到|Failed|正常解答', case=False, na=False)
success_df = df[success_mask]
if len(success_df) > 0:
    for idx, row in success_df.head(3).iterrows():
        print(f"\n示例 {idx + 1}:")
        print(f"  ID: {row['extra_info']['index']}")
        print(f"  攻击类型: {row['extra_info']['attack_type']}")
        print(f"  Defense Logic: {row['extra_info']['defense_logic']}")
        print(f"  Question (前100字符): {row['prompt'][0]['content'][:100]}...")
else:
    print("没有成功提取的示例")
print()

# 查看一些失败的示例
print("=" * 80)
print("失败的示例（前3条）")
print("=" * 80)
failed_mask = defense_logic.str.contains('无法提取|API调用失败|未找到|Failed', case=False, na=False)
failed_df = df[failed_mask]
if len(failed_df) > 0:
    for idx, row in failed_df.head(3).iterrows():
        print(f"\n示例 {idx + 1}:")
        print(f"  ID: {row['extra_info']['index']}")
        print(f"  攻击类型: {row['extra_info']['attack_type']}")
        print(f"  Defense Logic: {row['extra_info']['defense_logic']}")
        print(f"  Question (前100字符): {row['prompt'][0]['content'][:100]}...")
else:
    print("没有失败的示例")
print()

# 检查数据格式
print("=" * 80)
print("数据格式检查")
print("=" * 80)
sample = df.iloc[0]
print("第一条数据的所有字段:")
for col in df.columns:
    print(f"\n{col}:")
    if col == 'prompt':
        print(f"  类型: {type(sample[col])}")
        print(f"  内容: {sample[col]}")
    elif col == 'reward_model':
        print(f"  类型: {type(sample[col])}")
        print(f"  内容: {sample[col]}")
    elif col == 'extra_info':
        print(f"  类型: {type(sample[col])}")
        print(f"  内容: {json.dumps(sample[col], indent=2, ensure_ascii=False)}")
    else:
        print(f"  类型: {type(sample[col])}")
        print(f"  内容: {sample[col]}")
