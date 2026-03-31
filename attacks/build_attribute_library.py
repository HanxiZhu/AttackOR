#!/usr/bin/env python3
"""
构建属性库（Attribute Skill Library）
从数据集中综合提取常见领域场景、实体和属性，构建外挂知识库
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
from tqdm import tqdm

def load_model(model_path):
    """加载本地模型"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return tokenizer, model

def parse_llm_output(output_text):
    """解析LLM输出的JSON"""
    try:
        output_text = output_text.strip()
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        json_str = output_text[start:end]
        return json.loads(json_str)
    except:
        return None

def call_llm(prompt, tokenizer=None, model=None, api_client=None, api_model=None):
    """调用LLM生成响应"""
    if api_client:
        try:
            response = api_client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return None
    else:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7
            )
        response_text = tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        return response_text

def extract_categories(datasets, tokenizer=None, model=None, api_client=None, api_model=None):
    """从数据集中提取常见的领域类别"""
    print("\n步骤1: 提取领域类别...")
    
    # 收集所有问题（使用全部数据）
    all_questions = []
    total_count = 0
    for dataset_path, data in datasets.items():
        questions = [item.get('en_question', item.get('question', '')) 
                     for item in data if item.get('en_question') or item.get('question')]
        all_questions.extend(questions)
        total_count += len(data)
        print(f"  从 {Path(dataset_path).name} 收集了 {len(questions)} 个问题")
    
    print(f"  总共收集了 {len(all_questions)} 个问题")
    
    # 如果问题太多，需要分批处理或总结
    # 先尝试全部发送，如果token太多，LLM会自动截断或我们可以分批
    # 为了节省token，可以只发送问题的开头部分
    if len(all_questions) > 100:
        # 如果问题太多，使用代表性样本（每个数据集采样，但采样更多）
        print(f"  问题数量较多({len(all_questions)})，使用代表性样本进行分析...")
        sample_questions = []
        samples_per_dataset = max(50, len(all_questions) // len(datasets) // 2)  # 每个数据集至少50个，或总数的1/2N
        for dataset_path, data in datasets.items():
            questions = [item.get('en_question', item.get('question', '')) 
                        for item in data if item.get('en_question') or item.get('question')]
            # 均匀采样
            if len(questions) > samples_per_dataset:
                step = len(questions) // samples_per_dataset
                sample_questions.extend(questions[::step][:samples_per_dataset])
            else:
                sample_questions.extend(questions)
        questions_text = "\n\n".join([f"Question {i+1}: {q[:300]}..." 
                                     for i, q in enumerate(sample_questions)])
        print(f"  使用 {len(sample_questions)} 个代表性样本进行分析")
    else:
        # 问题不多，使用全部
        questions_text = "\n\n".join([f"Question {i+1}: {q[:300]}..." 
                                     for i, q in enumerate(all_questions)])
        print(f"  使用全部 {len(all_questions)} 个问题进行分析")
    
    prompt = f"""You are analyzing a collection of optimization problems to identify common domain categories.

Here are some sample questions from the dataset:

{questions_text}

Your task is to identify the most common domain categories (e.g., production, transportation, investment, scheduling, resource_allocation) that appear in these optimization problems.

Output a JSON list of category names and their descriptions:
{{
    "categories": [
        {{
            "name": "production",
            "description": "生产制造领域 - problems involving factories, workers, manufacturing, production planning"
        }},
        {{
            "name": "transportation",
            "description": "运输物流领域 - problems involving trucks, warehouses, ports, logistics, shipping"
        }}
    ]
}}

Only include categories that are clearly represented in the sample questions. Output ONLY the JSON, no explanations."""
    
    response = call_llm(prompt, tokenizer, model, api_client, api_model)
    result = parse_llm_output(response)
    
    if result and "categories" in result:
        categories = {cat["name"]: cat["description"] for cat in result["categories"]}
        print(f"提取到 {len(categories)} 个领域类别: {list(categories.keys())}")
        return categories
    else:
        # 默认类别
        return {
            "production": "生产制造领域",
            "transportation": "运输物流领域",
            "investment": "投资金融领域",
            "scheduling": "调度规划领域",
            "resource_allocation": "资源分配领域"
        }

def extract_entities_for_category(category_name, category_desc, datasets, 
                                  tokenizer=None, model=None, api_client=None, api_model=None):
    """为特定类别提取常见实体"""
    print(f"\n步骤2: 为类别 '{category_name}' 提取实体...")
    
    # 收集所有相关问题的样本（使用全部数据）
    relevant_questions = []
    for dataset_path, data in datasets.items():
        for item in data:
            q = item.get('en_question', item.get('question', ''))
            if q and any(keyword in q.lower() for keyword in category_name.split('_')):
                relevant_questions.append(q)
    
    print(f"  找到 {len(relevant_questions)} 个相关问题")
    
    # 如果相关问题太多，采样代表性样本
    if len(relevant_questions) > 100:
        # 均匀采样
        step = len(relevant_questions) // 100
        sample_questions = relevant_questions[::step][:100]
        print(f"  使用 {len(sample_questions)} 个代表性样本进行分析")
    else:
        sample_questions = relevant_questions
        print(f"  使用全部 {len(sample_questions)} 个相关问题进行分析")
    
    if not sample_questions:
        # 如果没有找到相关问题的，使用所有问题的样本
        print(f"  未找到相关问题，使用全部问题的样本...")
        all_questions = []
        for dataset_path, data in datasets.items():
            questions = [item.get('en_question', item.get('question', '')) 
                        for item in data if item.get('en_question') or item.get('question')]
            all_questions.extend(questions)
        
        if len(all_questions) > 100:
            step = len(all_questions) // 100
            sample_questions = all_questions[::step][:100]
        else:
            sample_questions = all_questions
    
    questions_text = "\n\n".join([f"Question {i+1}: {q[:400]}..." 
                                 for i, q in enumerate(sample_questions)])
    
    prompt = f"""You are analyzing optimization problems in the "{category_name}" domain ({category_desc}).

Here are sample questions from this domain:

{questions_text}

Your task is to identify the most common entity types (e.g., factory, worker, product, truck, warehouse) that appear in these problems.

Output a JSON list of entity names and their descriptions:
{{
    "entities": [
        {{
            "name": "factory",
            "description": "工厂"
        }},
        {{
            "name": "worker",
            "description": "工人"
        }}
    ]
}}

Only include entities that are clearly represented in the sample questions. Output ONLY the JSON, no explanations."""
    
    response = call_llm(prompt, tokenizer, model, api_client, api_model)
    result = parse_llm_output(response)
    
    if result and "entities" in result:
        entities = {ent["name"]: ent["description"] for ent in result["entities"]}
        print(f"  提取到 {len(entities)} 个实体: {list(entities.keys())}")
        return entities
    else:
        return {}

def extract_attributes_for_entity(category_name, entity_name, entity_desc, 
                                 tokenizer=None, model=None, api_client=None, api_model=None):
    """为特定实体提取相关属性"""
    print(f"  步骤3: 为实体 '{entity_name}' 提取属性...")
    
    prompt = f"""You are an expert in the "{category_name}" domain. You need to identify realistic attributes that could be associated with "{entity_name}" ({entity_desc}) in optimization problems.

These attributes should be:
1. Realistic industry standards or requirements
2. Per-unit attributes (not global constraints)
3. Could be used as distracting information in optimization problems
4. Should NOT include value ranges (we only need the attribute name, description, unit, and example usage)

Output a JSON list of attributes:
{{
    "attributes": [
        {{
            "name": "quality certification level",
            "description": "质量认证等级",
            "unit": "level",
            "example_usage": "Each {entity_name} must maintain a quality certification level"
        }},
        {{
            "name": "safety training certification",
            "description": "安全培训认证",
            "unit": "certification",
            "example_usage": "Each {entity_name} requires safety training certification"
        }}
    ]
}}

Generate 3-5 realistic attributes. Output ONLY the JSON, no explanations."""
    
    response = call_llm(prompt, tokenizer, model, api_client, api_model)
    result = parse_llm_output(response)
    
    if result and "attributes" in result:
        attributes = result["attributes"]
        print(f"    提取到 {len(attributes)} 个属性")
        return attributes
    else:
        return []

def extract_generic_attributes(category_name, tokenizer=None, model=None, 
                              api_client=None, api_model=None):
    """为类别提取通用属性（用于未匹配的实体）"""
    prompt = f"""You are an expert in the "{category_name}" domain. Generate generic attributes that could apply to any entity in this domain.

These should be:
1. Generic enough to apply to various entities
2. Per-unit attributes (not global constraints)
3. Realistic industry standards

Output a JSON list:
{{
    "attributes": [
        {{
            "name": "operating standard",
            "description": "操作标准",
            "unit": "standard",
            "example_usage": "Each entity must meet operating standard requirements"
        }}
    ]
}}

Generate 2-3 generic attributes. Output ONLY the JSON, no explanations."""
    
    response = call_llm(prompt, tokenizer, model, api_client, api_model)
    result = parse_llm_output(response)
    
    if result and "attributes" in result:
        return result["attributes"]
    else:
        return []

def load_datasets(dataset_paths):
    """加载所有数据集"""
    datasets = {}
    for path_str in dataset_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping")
            continue
        
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split('\n') if line.strip()]
        
        datasets[str(path)] = data
        print(f"  Loaded {len(data)} samples")
    
    return datasets

def main():
    parser = argparse.ArgumentParser(description="构建属性库（Attribute Skill Library）")
    parser.add_argument('--datasets', '-d', nargs='+', 
                       default=[
                           './datasets_local/IndustryOR_test.jsonl',
                           './datasets_local/NL4OPT_test.jsonl',
                           './datasets_local/MAMO_complex_lp.jsonl',
                           './datasets_local/MAMO_easy_lp.jsonl'
                       ],
                       help='数据集路径列表')
    parser.add_argument('--output', '-o', default='./attribute_library.json',
                       help='输出属性库文件路径')
    import os
    parser.add_argument('--model', '-m',
                       default='Qwen/Qwen2.5-7B-Instruct',
                       help='本地模型路径或 Hugging Face 模型 ID')
    parser.add_argument('--api-model', type=str, default='Pro/deepseek-ai/DeepSeek-V3.2',
                       help='API模型名称（如: gpt-4o-mini）。如果不指定，将使用本地模型')
    parser.add_argument('--api-base-url', type=str,
                       default=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                       help='OpenAI 兼容 API base URL')
    parser.add_argument('--api-key', type=str,
                       default=os.getenv('OPENAI_API_KEY', ''),
                       help='API 密钥（推荐 OPENAI_API_KEY 环境变量）')
    
    args = parser.parse_args()
    
    # 初始化LLM
    if args.api_model:
        print(f"使用API模式: {args.api_model}")
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('请通过 --api-key 或环境变量 OPENAI_API_KEY 提供 API 密钥')
        api_client = OpenAI(api_key=api_key, base_url=args.api_base_url)
        tokenizer, model = None, None
    else:
        print(f"使用本地模型: {args.model}")
        tokenizer, model = load_model(args.model)
        api_client = None
    
    # 加载数据集
    print("\n" + "="*60)
    print("加载数据集...")
    print("="*60)
    datasets = load_datasets(args.datasets)
    
    if not datasets:
        print("错误: 没有成功加载任何数据集")
        return
    
    # 构建属性库
    print("\n" + "="*60)
    print("构建属性库...")
    print("="*60)
    
    # 步骤1: 提取领域类别
    categories = extract_categories(datasets, tokenizer, model, api_client, args.api_model)
    
    # 步骤2-4: 为每个类别提取实体和属性
    library = {"categories": {}}
    
    for cat_name, cat_desc in tqdm(categories.items(), desc="处理类别"):
        # 提取实体
        entities_dict = extract_entities_for_category(
            cat_name, cat_desc, datasets, tokenizer, model, api_client, args.api_model
        )
        
        # 为每个实体提取属性
        entities = {}
        for ent_name, ent_desc in entities_dict.items():
            attributes = extract_attributes_for_entity(
                cat_name, ent_name, ent_desc, tokenizer, model, api_client, args.api_model
            )
            entities[ent_name] = {
                "description": ent_desc,
                "attributes": attributes
            }
        
        # 提取通用属性
        generic_attributes = extract_generic_attributes(
            cat_name, tokenizer, model, api_client, args.api_model
        )
        
        library["categories"][cat_name] = {
            "description": cat_desc,
            "entities": entities,
            "generic_attributes": generic_attributes
        }
    
    # 保存属性库
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("属性库构建完成!")
    print("="*60)
    print(f"输出文件: {output_path}")
    print(f"类别数: {len(library['categories'])}")
    
    total_entities = sum(len(cat['entities']) for cat in library['categories'].values())
    total_attributes = sum(
        sum(len(ent['attributes']) for ent in cat['entities'].values()) 
        for cat in library['categories'].values()
    )
    print(f"实体总数: {total_entities}")
    print(f"属性总数: {total_attributes}")

if __name__ == "__main__":
    main()
