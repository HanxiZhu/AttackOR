"""
下载ORLM评估数据集到本地并查看内容
"""
import datasets
import json
import os
import argparse
from pathlib import Path

# 数据集配置
DATASETS = {
    "NL4OPT": {
        "name": "CardinalOperations/NL4OPT",
        "split": "test"
    },
    "MAMO_EasyLP": {
        "name": "CardinalOperations/MAMO",
        "split": "easy_lp"
    },
    "MAMO_ComplexLP": {
        "name": "CardinalOperations/MAMO",
        "split": "complex_lp"
    },
    "IndustryOR": {
        "name": "CardinalOperations/IndustryOR",
        "split": "test"
    }
}

def download_dataset(dataset_name: str, split: str, output_dir: str = "./datasets_local"):
    """
    下载数据集并保存为JSONL格式
    
    Args:
        dataset_name: HuggingFace数据集名称
        split: 数据集分割
        output_dir: 输出目录
    """
    print(f"\n正在下载: {dataset_name} ({split})")
    
    try:
        # 加载数据集
        ds = datasets.load_dataset(dataset_name)
        ds_split = ds[split]
        
        print(f"数据集大小: {len(ds_split)} 个样本")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        dataset_short_name = dataset_name.split("/")[-1]
        output_file = os.path.join(output_dir, f"{dataset_short_name}_{split}.jsonl")
        
        # 保存为JSONL
        print(f"正在保存到: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for example in ds_split:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        print(f"✓ 已保存到: {output_file}")
        
        # 显示前3个样本
        print(f"\n前3个样本预览:")
        for i, example in enumerate(ds_split[:3]):
            print(f"\n--- 样本 {i+1} ---")
            # 确保example是字典类型
            if not isinstance(example, dict):
                print(f"警告: 样本不是字典类型，而是 {type(example)}")
                print(f"内容: {str(example)[:200]}")
                continue
            print(f"Keys: {list(example.keys())}")
            if "en_question" in example:
                question_preview = example['en_question'][:200]
                if len(example['en_question']) > 200:
                    question_preview += "..."
                print(f"问题: {question_preview}")
            if "en_answer" in example:
                answer_preview = str(example['en_answer'])[:200]
                if len(str(example['en_answer'])) > 200:
                    answer_preview += "..."
                print(f"答案: {answer_preview}")
        
        return output_file
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None

def explore_dataset(jsonl_file: str, num_samples: int = 5):
    """
    查看数据集内容
    
    Args:
        jsonl_file: JSONL文件路径
        num_samples: 查看的样本数量
    """
    print(f"\n{'='*60}")
    print(f"查看数据集: {jsonl_file}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(jsonl_file):
        print(f"错误: 文件不存在 {jsonl_file}")
        return
    
    samples = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"总样本数: {len(samples)}")
    if samples:
        print(f"\n字段列表: {list(samples[0].keys())}")
        
        # 显示前N个样本
        for i, sample in enumerate(samples[:num_samples]):
            print(f"\n{'='*60}")
            print(f"样本 {i+1}/{min(num_samples, len(samples))}")
            print(f"{'='*60}")
            
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载和查看ORLM数据集")
    parser.add_argument("--dataset", type=str, default="all", 
                       choices=["all", "NL4OPT", "MAMO_EasyLP", "MAMO_ComplexLP", "IndustryOR"],
                       help="要下载的数据集")
    parser.add_argument("--output_dir", type=str, default="./datasets_local",
                       help="输出目录")
    parser.add_argument("--explore", type=str, default=None,
                       help="查看指定JSONL文件的内容")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="查看的样本数量")
    
    args = parser.parse_args()
    
    if args.explore:
        # 查看指定文件
        explore_dataset(args.explore, args.num_samples)
    else:
        # 下载数据集
        if args.dataset == "all":
            downloaded = []
            for name, config in DATASETS.items():
                result = download_dataset(config["name"], config["split"], args.output_dir)
                if result:
                    downloaded.append(result)
            
            print(f"\n{'='*60}")
            print(f"所有数据集已下载到: {args.output_dir}")
            print(f"{'='*60}")
            print("\n下载的文件:")
            for f in downloaded:
                print(f"  - {f}")
            print("\n使用方法:")
            print("  python download_datasets.py --explore <jsonl_file>")
        else:
            config = DATASETS[args.dataset]
            download_dataset(config["name"], config["split"], args.output_dir)

