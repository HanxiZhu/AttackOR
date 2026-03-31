#!/usr/bin/env python3
"""
基于属性库的Attribute攻击实现
"""
import json
import os
import re
import random
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI

def load_model(model_path):
    """加载本地模型"""
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

def call_llm(prompt, tokenizer=None, model=None, api_client=None, api_model=None, max_tokens=512):
    """调用LLM生成响应。max_tokens 默认 512，phantom 融合整题时可传 2048 等更大值以防截断。"""
    if api_client:
        try:
            response = api_client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
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
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7
            )
        response_text = tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        return response_text

class AttributeLibraryAttacker:
    def __init__(self, library_path, tokenizer=None, model=None, api_client=None, api_model=None):
        """初始化攻击器"""
        self.tokenizer = tokenizer
        self.model = model
        self.api_client = api_client
        self.api_model = api_model
        self.use_api = api_client is not None
        self.library = self.load_library(library_path)
    
    def load_library(self, library_path):
        """加载属性库"""
        with open(library_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def identify_domain_and_entities(self, question):
        """使用LLM识别题目领域和核心实体"""
        prompt = f"""Analyze the following optimization problem and identify:
1. The domain category (e.g., production, transportation, investment, scheduling, resource_allocation)
2. The main entities mentioned in the problem (e.g., factory, worker, truck, warehouse)

Question:
{question}

Output JSON:
{{
    "domain": "production",
    "entities": ["factory", "worker", "product"]
}}

Output ONLY the JSON, no explanations."""
        
        response = call_llm(prompt, self.tokenizer, self.model, self.api_client, self.api_model)
        result = parse_llm_output(response)
        
        if result:
            return result.get("domain", ""), result.get("entities", [])
        else:
            return "", []
    
    def find_matching_category(self, domain):
        """在属性库中找到最匹配的Category"""
        if not domain:
            return None
        
        # 直接匹配
        if domain in self.library.get("categories", {}):
            return domain
        
        # 模糊匹配
        domain_lower = domain.lower()
        for cat_name in self.library.get("categories", {}).keys():
            if domain_lower in cat_name or cat_name in domain_lower:
                return cat_name
        
        return None
    
    def find_matching_entity(self, category, entities):
        """在Category中找到最匹配的Entity"""
        if not category or category not in self.library.get("categories", {}):
            return None, None
        
        category_data = self.library["categories"][category]
        entity_dict = category_data.get("entities", {})
        
        # 尝试直接匹配
        for entity in entities:
            entity_lower = entity.lower()
            for ent_name in entity_dict.keys():
                if entity_lower == ent_name.lower() or entity_lower in ent_name.lower():
                    return ent_name, entity_dict[ent_name]
        
        # 模糊匹配
        for entity in entities:
            entity_lower = entity.lower()
            for ent_name, ent_data in entity_dict.items():
                if entity_lower in ent_name or ent_name in entity_lower:
                    return ent_name, ent_data
        
        return None, None
    
    def select_attribute(self, entity_data, question):
        """从Entity的属性列表中选择属性（优先选择最不相关但看似合理的）"""
        if not entity_data:
            return None
        
        attributes = entity_data.get("attributes", [])
        if not attributes:
            return None
        
        # 使用LLM选择最不相关但看似合理的属性
        attributes_text = "\n".join([
            f"- {attr['name']}: {attr['description']} ({attr['unit']})"
            for attr in attributes
        ])
        
        prompt = f"""You are selecting a distracting attribute for an adversarial attack on an optimization problem.

Question:
{question[:500]}

Available attributes:
{attributes_text}

Select the attribute that is:
1. Least relevant to solving the optimization problem
2. But still seems plausible and realistic
3. Would be most confusing for an AI solving this problem

CRITICAL: The selected attribute must NOT affect the ground truth answer or change the feasible region of the optimization problem.

Output JSON:
{{
    "selected_attribute": {{
        "name": "quality certification level",
        "description": "质量认证等级",
        "unit": "level",
        "example_usage": "Each entity must maintain a quality certification level"
    }}
}}

Output ONLY the JSON, no explanations."""
        
        response = call_llm(prompt, self.tokenizer, self.model, self.api_client, self.api_model)
        result = parse_llm_output(response)
        
        if result and "selected_attribute" in result:
            return result["selected_attribute"]
        else:
            # 回退：随机选择
            return random.choice(attributes)
    
    def analyze_question_value_range(self, question):
        """分析题目中的数值范围（min, max, 格式等）"""
        # 提取所有数值
        numbers = []
        patterns = [
            r'\b(\d+\.?\d*)\b',  # 整数或小数
            r'\$(\d+\.?\d*)',    # 美元金额
            r'(\d+\.?\d*)\s*(yuan|yuan)',  # 人民币
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    num_str = match[0]
                else:
                    num_str = match
                try:
                    num = float(num_str)
                    if num > 0:  # 忽略0
                        numbers.append(num)
                except:
                    pass
        
        if not numbers:
            return None, None, "integer"
        
        min_val = min(numbers)
        max_val = max(numbers)
        
        # 判断格式
        has_decimal = any(n != int(n) for n in numbers)
        format_type = "decimal" if has_decimal else "integer"
        
        return min_val, max_val, format_type
    
    def generate_value_in_range(self, attribute, question, min_val, max_val, format_type):
        """LLM根据题目数值范围，为选定的属性生成合适的数值"""
        prompt = f"""You are generating a distracting value for an attribute in an optimization problem.

Question context (showing value ranges):
{question[:500]}

Attribute to generate value for:
- Name: {attribute['name']}
- Description: {attribute['description']}
- Unit: {attribute['unit']}

Value range in question: {min_val:.2f} to {max_val:.2f}
Value format: {format_type}

Generate a value that:
1. Is in a similar magnitude to the values in the question
2. Matches the format (integer or decimal)
3. Is realistic for this attribute
4. Would be confusing but not obviously wrong

CRITICAL: The generated value and description must NOT affect the ground truth answer or change the feasible region of the optimization problem.

Output JSON:
{{
    "value": 3.2,
    "unit": "level",
    "full_description": "Each entity must maintain a quality certification level of 3.2"
}}

Output ONLY the JSON, no explanations."""
        
        response = call_llm(prompt, self.tokenizer, self.model, self.api_client, self.api_model)
        result = parse_llm_output(response)
        
        if result:
            return result.get("value"), result.get("unit", attribute.get("unit", "")), result.get("full_description", "")
        else:
            # 回退：在范围内随机生成
            if format_type == "decimal":
                value = round(random.uniform(min_val * 0.1, max_val * 0.1), 1)
            else:
                value = int(random.uniform(min_val * 0.1, max_val * 0.1))
            return value, attribute.get("unit", ""), f"Each entity must have {attribute['name']} of {value} {attribute.get('unit', '')}"
    
    def fuse_attribute_naturally(self, question, attribute, value, unit, full_description):
        """使用LLM（扮演行业专家）自然缝合属性到题目中"""
        prompt = f"""You are an industry expert adding a realistic requirement to an optimization problem. Your task is to naturally integrate the following attribute information into the problem statement.

Original question:
{question}

Attribute to add:
{full_description}

Instructions:
1. Act as an industry expert who is providing additional context
2. Integrate this attribute naturally - you can either:
   - Insert a new sentence that flows naturally with the text
   - Modify an existing sentence to include this information
3. Use authoritative language (e.g., "According to regulations", "Industry standards require", "Compliance mandates")
4. Make it sound like a legitimate requirement, not an obvious addition
5. Ensure the sentence is grammatically correct and flows naturally

CRITICAL: The integrated attribute must NOT affect the ground truth answer or change the feasible region of the optimization problem. It should only be a distracting per-unit property.

Output the complete modified question with the attribute naturally integrated. Do NOT add explanations or markdown formatting."""
        
        response = call_llm(prompt, self.tokenizer, self.model, self.api_client, self.api_model)
        
        if response:
            # 清理响应（移除可能的markdown格式）
            response = response.strip()
            if response.startswith("```"):
                # 移除代码块标记
                lines = response.split('\n')
                response = '\n'.join([l for l in lines if not l.strip().startswith('```')])
            return response.strip()
        else:
            return question
    
    def enhanced_safety_check(self, original_question, modified_question):
        """增强的安全检查"""
        modified_lower = modified_question.lower()
        
        # 检查全局约束关键词
        dangerous_patterns = [
            r'\btotal\s+(budget|cost|capacity|amount|limit)',
            r'\bmaximum\s+(budget|cost|capacity|amount)',
            r'\bminimum\s+(budget|cost|capacity|amount)',
            r'\bat\s+most\s+\d+',
            r'\bat\s+least\s+\d+',
            r'\bmust\s+not\s+exceed',
            r'\bcannot\s+exceed',
            r'\bexceed\s+the\s+(total|maximum)',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, modified_lower):
                return False, f"Global constraint pattern detected: {pattern}"
        
        # 检查是否引入了新的约束（通过对比原问题）
        # 这里可以添加更复杂的检查逻辑
        
        return True, "Safe"
    
    def attack(self, question, return_failure_reason=False):
        """执行完整的攻击流程
        
        Args:
            question: 原始问题
            return_failure_reason: 如果为True，失败时返回(None, reason)而不是None
        
        Returns:
            成功: modified_question 或 (modified_question, None)
            失败: None 或 (None, failure_reason)
        """
        # 步骤1: 识别领域和实体
        domain, entities = self.identify_domain_and_entities(question)
        if not domain:
            reason = "无法识别领域"
            return (None, reason) if return_failure_reason else None
        
        # 步骤2: 找到匹配的Category
        category = self.find_matching_category(domain)
        if not category:
            reason = f"无法匹配Category (识别到的domain: {domain})"
            return (None, reason) if return_failure_reason else None
        
        # 步骤3: 找到匹配的Entity
        entity_name, entity_data = self.find_matching_entity(category, entities)
        
        # 如果找不到Entity，使用generic_attributes
        if not entity_data:
            category_data = self.library["categories"][category]
            generic_attrs = category_data.get("generic_attributes", [])
            if not generic_attrs:
                reason = f"找不到Entity且Category '{category}'没有generic_attributes"
                return (None, reason) if return_failure_reason else None
            # 使用第一个通用属性
            attribute = generic_attrs[0]
        else:
            # 步骤4: 选择属性
            attribute = self.select_attribute(entity_data, question)
            if not attribute:
                reason = f"无法选择属性 (Entity: {entity_name})"
                return (None, reason) if return_failure_reason else None
        
        # 步骤5: 分析数值范围
        min_val, max_val, format_type = self.analyze_question_value_range(question)
        if min_val is None:
            # 如果没有数值，使用默认范围
            min_val, max_val = 1.0, 100.0
            format_type = "decimal"
        
        # 步骤6: 生成数值
        value, unit, full_description = self.generate_value_in_range(
            attribute, question, min_val, max_val, format_type
        )
        
        # 步骤7: 自然融合
        modified_question = self.fuse_attribute_naturally(
            question, attribute, value, unit, full_description
        )

        # 安全检查：避免引入影响可行域/最优解的全局约束模式
        is_safe, safety_reason = self.enhanced_safety_check(question, modified_question)
        if not is_safe:
            reason = f"安全检查未通过: {safety_reason}"
            return (None, reason) if return_failure_reason else None
        
        return (modified_question, None) if return_failure_reason else modified_question

def main():
    """测试函数"""
    parser = argparse.ArgumentParser(description="基于属性库的Attribute攻击测试")
    parser.add_argument('--library', '-l', default='./attribute_library.json',
                       help='属性库文件路径')
    parser.add_argument('--question', '-q', type=str,
                       help='测试问题')
    parser.add_argument('--model', '-m',
                       default='Qwen/Qwen2.5-7B-Instruct',
                       help='本地模型路径或 Hugging Face 模型 ID')
    parser.add_argument('--api-model', type=str, default="Pro/deepseek-ai/DeepSeek-V3.2",
                       help='API模型名称')
    parser.add_argument('--api-base-url', type=str,
                       default=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                       help='OpenAI 兼容 API base URL')
    parser.add_argument('--api-key', type=str,
                       default=os.getenv('OPENAI_API_KEY', ''),
                       help='API 密钥（推荐设置环境变量 OPENAI_API_KEY）')
    
    args = parser.parse_args()
    
    # 初始化
    if args.api_model:
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('请通过 --api-key 或环境变量 OPENAI_API_KEY 提供 API 密钥')
        api_client = OpenAI(api_key=api_key, base_url=args.api_base_url)
        tokenizer, model = None, None
    else:
        tokenizer, model = load_model(args.model)
        api_client = None
    
    attacker = AttributeLibraryAttacker(
        args.library, tokenizer, model, api_client, args.api_model
    )
    
    # 测试
    if args.question:
        question = args.question
    else:
        question = "A factory produces two types of food, I and II, and currently has 50 skilled workers."
    
    print("原始问题:")
    print(question)
    print("\n" + "="*60 + "\n")
    
    result = attacker.attack(question)
    
    if result:
        print("攻击后的问题:")
        print(result)
    else:
        print("攻击失败（无法匹配或安全检查未通过）")

if __name__ == "__main__":
    main()
