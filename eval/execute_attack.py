import subprocess
import os
import json
import tempfile
import concurrent.futures
import argparse
from collections import Counter, defaultdict

ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best solution: {model.objval}")\nelse:\n    print("No Best Solution")'

def load_data_robust(filepath):
    """
    鲁棒的数据加载函数：兼容 JSON List 和 JSONL
    """
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # 尝试整体读取
            content = json.load(f)
            if isinstance(content, list):
                return content
            elif isinstance(content, dict):
                return [content]
    except:
        pass # 失败则回退到逐行读取

    # 逐行读取 (JSONL)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return []

def majority_voting(pred_answers):
    # Filter out None values before counting
    valid_answers = [ans for ans in pred_answers if ans is not None]
    if not valid_answers:
        return None
        
    count = Counter(valid_answers)
    max_count = max(count.values())
    possible_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    return possible_answers[0]

def compile_script(script_content, timeout=300):
    target_dir = './eval_execute'
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.py', dir=target_dir) as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(script_content.encode())

    try:
        process = subprocess.run(['python', tmp_file_name], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
        execution_result = process.stdout
        
        # 提取逻辑优化
        execution_best_solution = None
        execution_state = "Execution Suceessful but Out of Expectation"
        
        # 寻找标准输出标记
        marker = "Just print the best solution:"
        pos = execution_result.find(marker)
        
        if pos != -1:
            # 提取数字部分
            raw_sol = execution_result[pos + len(marker):].strip().split('\n')[0].strip()
            execution_best_solution = raw_sol
            execution_state = "Execution Successful and Best Solution Found"
        elif "No Best Solution" in execution_result:
            execution_best_solution = "No Best Solution"
            execution_state = "Execution Successful but No Best Solution Found"
            
    except subprocess.TimeoutExpired as e:
        execution_result = e.stdout if e.stdout else "Timeout"
        execution_best_solution = None
        execution_state = "Execution Failed: Timeout"
    except subprocess.CalledProcessError as e:
        execution_result = e.stdout if e.stdout else str(e.stderr)
        execution_best_solution = None
        execution_state = f"Execution Failed: Process Error"
    except Exception as e:
        execution_result = str(e)
        execution_best_solution = None
        execution_state = f"Execution Failed: {str(e)}"
    finally:
        if os.path.exists(tmp_file_name):
            os.remove(tmp_file_name)

    return {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution, 
        "execution_state": execution_state
    }

def is_answer_correct(pred, gt, tolerance=0.05):
    """独立的判断逻辑函数"""
    if pred is None: 
        return False
    
    # 归一化 Ground Truth
    gt_val = str(gt).strip()
    pred_val = str(pred).strip()

    # 情况1: 无解情况
    if "No Best Solution" in gt_val:
        return "No Best Solution" in pred_val

    # 情况2: 数值比较
    try:
        gt_float = float(gt_val)
        pred_float = float(pred_val)
        
        # 避免除以0错误
        if abs(gt_float) < 1e-9:
            return abs(pred_float) <= tolerance
        else:
            return abs((pred_float - gt_float) / gt_float) <= tolerance
    except:
        # 如果转浮点失败，回退到字符串精确匹配
        return gt_val == pred_val

def main(args):
    # 环境检查
    try:
        subprocess.run(['python', '-c', 'import coptpy'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Check: coptpy is installed.")
    except:
        print("Warning: coptpy might not be installed properly.")

    # 1. 加载数据
    examples = load_data_robust(args.input_file)
    print(f"Loaded {len(examples)} examples.")

    # 2. 准备运行脚本
    early_failed = []
    to_run = []
    
    for example in examples:
        # 寻找代码字段
        code_field = next((k for k in example.keys() if "coptpy_code" in k), None)
        
        example_t = example.copy()
        
        if not code_field or not example.get(code_field):
            example_t.update({"execution_state": "Failed: No code field"})
            early_failed.append(example_t)
            continue

        output_code = example[code_field]
        
        # 提取代码块
        start = output_code.find("```python")
        if start == -1:
            start = output_code.find("```") # 尝试不带 python 标记的代码块
        
        if start != -1:
            end = output_code.find("```", start + 3)
            if end != -1:
                script = output_code[start:end].replace("```python", "").replace("```", "").strip()
            else:
                script = output_code[start:].replace("```python", "").replace("```", "").strip()
        else:
            # 没找到代码块标记，尝试直接运行整个输出（极少情况）
            script = output_code
            
        if not script:
            example_t.update({"execution_state": "Failed: Empty script"})
            early_failed.append(example_t)
            continue

        # 注入 Trace 代码
        script += ADD_SCRIPT
        example_t["to_run_script"] = script
        to_run.append(example_t)

    print(f"Valid scripts to run: {len(to_run)}")

    # 3. 并行执行
    def process_example(ex):
        res = compile_script(ex["to_run_script"], timeout=args.timeout)
        ex.update(res)
        return json.dumps(ex, ensure_ascii=False)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_example, ex): ex for ex in to_run}
        
        # 实时写入结果
        with open(args.output_file, "w", encoding='utf-8') as fw:
            for ex in early_failed:
                line = json.dumps(ex, ensure_ascii=False)
                fw.write(line + "\n")
                results.append(ex) # 记录下来用于评估

            for future in concurrent.futures.as_completed(futures):
                try:
                    res_str = future.result()
                    fw.write(res_str + "\n")
                    results.append(json.loads(res_str))
                except Exception as e:
                    print(f"Error executing task: {e}")

    print("Execution execution completed. Starting evaluation...")

    # 4. 评估逻辑 (支持按 Type 分类)
    if args.question_field and args.answer_field:
        
        # 存储结构: type -> list of is_correct (0 or 1)
        # 如果是 pass@k (k>1)，这里简化处理：同一个ID只要有一个对就算对
        scores_by_type = defaultdict(list)
        
        # 为了处理 k>1 的情况，我们先按 ID 聚合
        # id -> {type: "...", gt: "...", preds: []}
        grouped_results = {} 
        
        for res in results:
            # 兼容：有些样本可能没有 ID，用 question 做 key
            qid = res.get("id", res.get(args.question_field))
            qtype = res.get("type", "unknown") # 获取攻击类型
            gt = res.get(args.answer_field)
            pred = res.get("execution_best_solution")
            
            if qid not in grouped_results:
                grouped_results[qid] = {
                    "type": qtype,
                    "gt": gt,
                    "preds": []
                }
            grouped_results[qid]["preds"].append(pred)

        # 计算得分
        for qid, info in grouped_results.items():
            gt = info["gt"]
            preds = info["preds"]
            qtype = info["type"]
            
            # Pass@k 逻辑: 只要有一个 pred 是对的，该样本就算 Pass
            is_pass = 0
            for p in preds:
                if is_answer_correct(p, gt, args.numerical_err_tolerance):
                    is_pass = 1
                    break
            
            scores_by_type[qtype].append(is_pass)
            scores_by_type["overall"].append(is_pass) # 总分

        # 5. 生成报告
        metrics = {}
        print("\n" + "="*20 + " Evaluation Report " + "="*20)
        for cat, scores in scores_by_type.items():
            acc = sum(scores) / len(scores) if scores else 0
            metrics[f"acc_{cat}"] = acc
            print(f"Type: {cat:<15} | Count: {len(scores):<5} | Accuracy: {acc:.2%}")
        print("="*60 + "\n")

        # 保存 Metrics
        metrics_file = args.output_file.replace(".jsonl", "").replace(".json", "") + ".metrics.json"
        with open(metrics_file, "w") as fw:
            json.dump(metrics, fw, indent=4)
            print(f"Metrics saved to {metrics_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./results_attack_phantom_llm/NL4OPT_test_adversarial_phantom_llm.jsonl", help="Path to local .json/.jsonl file or HF dataset name") 
    parser.add_argument("--output_file", type=str, default="./results_attack_phantom_llm/NL4OPT_test_adversarial_phantom_llm.metrics.jsonl", help="Path to local .json/.jsonl file or HF dataset name") 
    parser.add_argument("--timeout", type=int, default=60) 
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--majority_voting", action="store_true") # 暂未详细实现，本脚本主要关注 Pass@1
    parser.add_argument("--question_field", type=str, default="question") # 兼容原逻辑
    parser.add_argument("--answer_field", type=str, default="answer") 
    parser.add_argument("--numerical_err_tolerance", type=float, default=0.05)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)