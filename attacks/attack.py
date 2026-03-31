#!/usr/bin/env python3
"""
攻击生成器：支持 Attribute、Phantom、Arithmetic 攻击，可指定攻击类型或全部运行。
"""
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import Counter

from attack_with_arithmetic import attack as arithmetic_attack
from attack_with_unit_conversion import attack as unit_conversion_attack

def load_model(model_path):
    """加载本地模型（仅在需要 attribute/phantom 时调用）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return tokenizer, model

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Attribute攻击生成器 - 基于属性库的Attribute攻击",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置
  python attack.py
  
  # 指定输入数据集
  python attack.py --input ./datasets_local/IndustryOR_test.jsonl
  python attack.py --input ./datasets_local/MAMO_complex_lp.jsonl
  python attack.py --input ./datasets_local/MAMO_easy_lp.jsonl
  python attack.py --input ./datasets_local/NL4OPT_test.jsonl

  
  # 指定输入和输出
  python attack.py \\
      --input ./datasets_local/IndustryOR_test.jsonl \\
      --output ./datasets_local/IndustryOR_test_adversarial_attribute.jsonl
  
  # 使用API模型
  python attack.py \\
      --input ./datasets_local/IndustryOR_test.jsonl \\
      --api-model Pro/deepseek-ai/DeepSeek-V3.2

  # 仅 Phantom 攻击
  python attack.py --attacks phantom --input ./data.jsonl

  # 仅 Attribute 攻击（与之前行为一致）
  python attack.py --attacks attribute --input ./data.jsonl

  # 两种攻击都跑（默认）
  python attack.py --attacks all --input ./data.jsonl
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='./datasets_local/IndustryOR_test.jsonl',
        help='输入数据集路径（默认: ./datasets_local/IndustryOR_test.jsonl）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出数据集路径（默认: 自动根据输入文件名生成，如 xxx_adversarial_attribute.jsonl）'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='本地模型路径或 Hugging Face 模型 ID（用于 attribute/phantom 的本地推理）'
    )
    
    parser.add_argument(
        '--api-model',
        type=str,
        default="Pro/deepseek-ai/DeepSeek-V3.2",
        help='API模型名称（如: Pro/deepseek-ai/DeepSeek-V3.2）。如果不指定此参数，将使用本地模型。指定后将使用API而不是本地模型'
    )
    
    parser.add_argument(
        '--api-base-url',
        type=str,
        default=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        help='OpenAI 兼容 API 的 base URL（可通过环境变量 OPENAI_API_BASE 覆盖）'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('OPENAI_API_KEY', ''),
        help='API 密钥（推荐通过环境变量 OPENAI_API_KEY 设置）'
    )
    
    parser.add_argument(
        '--question-field',
        type=str,
        default='en_question',
        help='问题字段名（默认: en_question）'
    )
    
    parser.add_argument(
        '--answer-field',
        type=str,
        default='en_answer',
        help='答案字段名（默认: en_answer）'
    )
    
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='LLM批处理大小（默认: 32）。保留接口，attribute攻击为逐条处理'
    )
    
    parser.add_argument(
        '--concurrency',
        type=int,
        default=100,
        help='并发数（默认: 10）。用于API调用时的并发请求数，可以加快处理速度。注意：过高的并发可能导致API限流'
    )
    
    parser.add_argument(
        '--attribute-library',
        type=str,
        default='./attribute_library.json',
        help='属性库文件路径（默认: ./attribute_library.json）。如果文件不存在，攻击将失败'
    )
    
    parser.add_argument(
        '--attacks',
        type=str,
        nargs='+',
        choices=['attribute', 'phantom', 'arithmetic', 'unit', 'all'],
        default=['all'],
        help='要使用的攻击类型：attribute、phantom、arithmetic、unit（单位换算）、all（四种都跑）。默认: phantom'
    )
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 归一化攻击类型
    if "all" in args.attacks:
        attack_types = ["attribute", "phantom", "arithmetic", "unit"]
    else:
        attack_types = list(dict.fromkeys(args.attacks))  # 去重且保序
    
    # 如果没有指定输出路径，自动生成
    if args.output is None:
        input_path = Path(args.input)
        if len(attack_types) == 1:
            suffix = "_adversarial_" + attack_types[0]
        else:
            suffix = "_adversarial_unified"
        output_name = input_path.stem + suffix + input_path.suffix
        args.output = str(input_path.parent / output_name)
    
    # 1. 加载资源
    print("=" * 60)
    print("攻击生成器 (Attribute + Phantom)")
    print("=" * 60)
    print(f"输入数据集: {args.input}")
    print(f"输出数据集: {args.output}")
    print(f"攻击类型: {attack_types}")
    
    # 判断使用API还是本地模型（仅 attribute/phantom 需要模型）
    need_llm = "attribute" in attack_types or "phantom" in attack_types
    if need_llm and args.api_model:
        from openai import OpenAI
        print(f"模式: API调用")
        print(f"API模型: {args.api_model}")
        print(f"API地址: {args.api_base_url}")
        print(f"问题字段: {args.question_field}")
        print(f"答案字段: {args.answer_field}")
        print("=" * 60)
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("请通过 --api-key 参数或 OPENAI_API_KEY 环境变量提供API密钥")
        api_client = OpenAI(api_key=api_key, base_url=args.api_base_url)
        tokenizer, model = None, None
    elif need_llm:
        print(f"模式: 本地模型")
        print(f"模型路径: {args.model}")
        print(f"问题字段: {args.question_field}")
        print(f"答案字段: {args.answer_field}")
        print(f"批处理大小: {args.batch_size}")
        print("=" * 60)
        tokenizer, model = load_model(args.model)
        api_client = None
    else:
        print(f"模式: 仅规则攻击 (arithmetic)，无需模型")
        print("=" * 60)
        api_client, tokenizer, model = None, None, None
    
    # 2. 初始化攻击器（按需；仅 attribute/phantom 时加载属性库与模型相关依赖）
    library_path = Path(args.attribute_library)
    attr_attacker = None
    phantom_attacker = None
    if "attribute" in attack_types or "phantom" in attack_types:
        if not library_path.exists():
            raise FileNotFoundError(f"属性库文件不存在: {args.attribute_library}\n请先运行 build_attribute_library.py 构建属性库")
        print(f"\n加载属性库: {args.attribute_library}")
        from attack_with_attribute_library import AttributeLibraryAttacker
        from attack_with_phantom_library import PhantomLibraryAttacker
        try:
            if "attribute" in attack_types:
                if args.api_model:
                    attr_attacker = AttributeLibraryAttacker(
                        str(library_path), tokenizer=None, model=None,
                        api_client=api_client, api_model=args.api_model
                    )
                else:
                    attr_attacker = AttributeLibraryAttacker(
                        str(library_path), tokenizer=tokenizer, model=model,
                        api_client=None, api_model=None
                    )
                print("Attribute 攻击器初始化成功")
            if "phantom" in attack_types:
                if args.api_model:
                    phantom_attacker = PhantomLibraryAttacker(
                        str(library_path), tokenizer=None, model=None,
                        api_client=api_client, api_model=args.api_model
                    )
                else:
                    phantom_attacker = PhantomLibraryAttacker(
                        str(library_path), tokenizer=tokenizer, model=model,
                        api_client=None, api_model=None
                    )
                print("Phantom 攻击器初始化成功")
        except Exception as e:
            raise RuntimeError(f"无法初始化攻击器: {e}")
    
    # 3. 加载数据
    print(f"\n加载数据从 {args.input}...")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"输入文件为空: {args.input}")
            
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split('\n') if line.strip()]
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}\n请检查输入文件格式是否正确")
    except Exception as e:
        raise RuntimeError(f"加载数据时出错: {e}")
    
    if not data:
        raise ValueError(f"输入文件中没有有效数据: {args.input}")
    
    print(f"已加载 {len(data)} 个样本\n")
    
    # 4. 生成攻击
    print("=" * 60)
    print("生成攻击 (" + ", ".join(attack_types) + ")")
    print("=" * 60)
    print(f"总样本数: {len(data)}")
    print(f"并发数: {args.concurrency}\n")
    
    tasks = []
    for idx, item in enumerate(data):
        q = item.get(args.question_field, item.get('question', ''))
        gt = item.get(args.answer_field, item.get('answer', ''))
        tasks.append({"idx": idx, "question": q, "answer": gt})
    
    def process_attack(task):
        idx = task["idx"]
        q = task["question"]
        gt = task["answer"]
        out = {"idx": idx, "question": q, "answer": gt, "attribute": None, "phantom": None, "arithmetic": None, "unit": None}
        if "unit" in attack_types:
            q_unit, reason_unit = unit_conversion_attack(q, max_unit_conversions=2, expand_unit_map=True, return_failure_reason=True)
            out["unit"] = (q_unit, reason_unit)
        if "arithmetic" in attack_types:
            q_arith, _ = arithmetic_attack(q, max_replacements=1, return_failure_reason=True)
            out["arithmetic"] = (q_arith, None if q_arith else "无可替换系数数值")
        if "attribute" in attack_types:
            if args.api_model:
                att = attr_attacker
            else:
                att = AttributeLibraryAttacker(
                    str(library_path), tokenizer=tokenizer, model=model,
                    api_client=None, api_model=None
                )
            q_attr, reason = att.attack(q, return_failure_reason=True)
            out["attribute"] = (q_attr, reason)
        if "phantom" in attack_types:
            if args.api_model:
                ph = phantom_attacker
            else:
                ph = PhantomLibraryAttacker(
                    str(library_path), tokenizer=tokenizer, model=model,
                    api_client=None, api_model=None
                )
            q_ph, reason = ph.attack(q, return_failure_reason=True)
            out["phantom"] = (q_ph, reason)
        return out
    
    results = []
    attr_stats = {"success": 0, "failed": 0}
    phantom_stats = {"success": 0, "failed": 0}
    arithmetic_stats = {"success": 0, "failed": 0}
    unit_stats = {"success": 0, "failed": 0}
    attr_failure_reasons = Counter()
    phantom_failure_reasons = Counter()
    stats_lock = Lock()
    
    def _record_failure(reason, counter):
        if not reason:
            return
        if "无法识别领域" in reason:
            counter["无法识别领域"] += 1
        elif "无法匹配Category" in reason or "无法匹配 Category" in reason:
            counter["无法匹配Category"] += 1
        elif "没有generic_attributes" in reason or "无可用属性" in reason:
            counter["无可用属性"] += 1
        elif "无法选择属性" in reason:
            counter["无法选择属性"] += 1
        elif "安全检查未通过" in reason:
            counter["安全检查未通过"] += 1
        elif "LLM 未返回" in reason:
            counter["LLM未返回"] += 1
        else:
            counter["其他"] += 1
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_task = {executor.submit(process_attack, t): t for t in tasks}
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Attacks"):
            try:
                r = future.result()
                idx, q, gt = r["idx"], r["question"], r["answer"]
                results.append({"id": f"{idx}_baseline", "type": "baseline", "question": q, "answer": gt})
                if "attribute" in attack_types:
                    q_attr, reason = r["attribute"]
                    if q_attr:
                        results.append({"id": f"{idx}_llm_attribute", "type": "llm_attribute", "question": q_attr, "answer": gt})
                        with stats_lock:
                            attr_stats["success"] += 1
                    else:
                        with stats_lock:
                            attr_stats["failed"] += 1
                            _record_failure(reason, attr_failure_reasons)
                if "phantom" in attack_types:
                    q_ph, reason = r["phantom"]
                    if q_ph:
                        results.append({"id": f"{idx}_llm_phantom", "type": "llm_phantom", "question": q_ph, "answer": gt})
                        with stats_lock:
                            phantom_stats["success"] += 1
                    else:
                        with stats_lock:
                            phantom_stats["failed"] += 1
                            _record_failure(reason, phantom_failure_reasons)
                if "arithmetic" in attack_types:
                    q_arith, _ = r["arithmetic"]
                    if q_arith:
                        results.append({"id": f"{idx}_arithmetic", "type": "arithmetic", "question": q_arith, "answer": gt})
                        with stats_lock:
                            arithmetic_stats["success"] += 1
                    else:
                        with stats_lock:
                            arithmetic_stats["failed"] += 1
                if "unit" in attack_types:
                    q_unit, _ = r["unit"]
                    if q_unit:
                        results.append({"id": f"{idx}_unit_conversion", "type": "unit_conversion", "question": q_unit, "answer": gt})
                        with stats_lock:
                            unit_stats["success"] += 1
                    else:
                        with stats_lock:
                            unit_stats["failed"] += 1
            except Exception as e:
                t = future_to_task[future]
                print(f"\n警告: 处理样本 {t['idx']} 时出错: {e}")
                with stats_lock:
                    if "attribute" in attack_types:
                        attr_stats["failed"] += 1
                        attr_failure_reasons["异常错误"] += 1
                    if "phantom" in attack_types:
                        phantom_stats["failed"] += 1
                        phantom_failure_reasons["异常错误"] += 1
                    if "arithmetic" in attack_types:
                        arithmetic_stats["failed"] += 1
                    if "unit" in attack_types:
                        unit_stats["failed"] += 1
    
    type_order = {"baseline": 0, "llm_attribute": 1, "llm_phantom": 2, "arithmetic": 3, "unit_conversion": 4}
    results.sort(key=lambda x: (int(x["id"].split("_")[0]), type_order.get(x["type"], 3)))
    
    if "attribute" in attack_types:
        total = len(data)
        print(f"\n📊 Attribute 攻击:")
        print(f"   ✅ 成功: {attr_stats['success']}/{total} ({100*attr_stats['success']/total:.1f}%)")
        print(f"   ❌ 失败: {attr_stats['failed']}/{total}")
        for reason, count in attr_failure_reasons.most_common():
            print(f"      - {reason}: {count}")
    if "phantom" in attack_types:
        total = len(data)
        print(f"\n📊 Phantom 攻击:")
        print(f"   ✅ 成功: {phantom_stats['success']}/{total} ({100*phantom_stats['success']/total:.1f}%)")
        print(f"   ❌ 失败: {phantom_stats['failed']}/{total}")
        for reason, count in phantom_failure_reasons.most_common():
            print(f"      - {reason}: {count}")
    if "arithmetic" in attack_types:
        total = len(data)
        print(f"\n📊 Arithmetic 攻击:")
        print(f"   ✅ 成功: {arithmetic_stats['success']}/{total} ({100*arithmetic_stats['success']/total:.1f}%)")
        print(f"   ❌ 失败: {arithmetic_stats['failed']}/{total}")
    if "unit" in attack_types:
        total = len(data)
        print(f"\n📊 Unit Conversion 攻击:")
        print(f"   ✅ 成功: {unit_stats['success']}/{total} ({100*unit_stats['success']/total:.1f}%)")
        print(f"   ❌ 失败: {unit_stats['failed']}/{total}")

    # 5. 保存结果
    print(f"\n保存结果到 {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 6. 统计信息
    print("\n" + "=" * 60)
    print("生成完成!")
    print("=" * 60)
    print(f"总样本数: {len(results)}")
    
    type_counts = Counter([x['type'] for x in results])
    print("\n攻击类型分布:")
    for attack_type, count in sorted(type_counts.items()):
        print(f"  - {attack_type}: {count}")
    
    print(f"\n输出已保存到: {args.output}")

if __name__ == "__main__":
    main()
