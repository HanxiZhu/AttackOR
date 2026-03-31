#!/usr/bin/env python3
"""
Phantom 攻击：按与 Attribute 类似的多步流程实现。
- Step 1: identify_domain_and_entities (LLM) 复用
- Step 2: find_matching_category (规则)
- Step 3: find_matching_entity (规则)
- Step 4: select_attribute_for_phantom (LLM) 选「最适合作为稀缺资源 XX」的属性
- Step 5: generate_phantom_option (LLM) 只生成虚假选项文本（模仿题目已有选项）
- Step 6: generate_constraint_sentence (LLM) 只生成一句「该选项需要 XX，但 XX 为 0/受限」
- Step 7: fuse_phantom_naturally (LLM) 将 phantom 选项 + 约束句融合进题目，输出完整题目
"""
import json
import random
from pathlib import Path

from attack_with_attribute_library import (
    AttributeLibraryAttacker,
    call_llm,
    parse_llm_output,
)


def _pick_constraint_attribute_fallback(entity_data, category_data):
    """回退：从 entity 或 category 的 generic_attributes 中随机选一个作为 XX。"""
    if entity_data:
        attrs = entity_data.get("attributes", [])
        if attrs:
            return random.choice(attrs)
    generic = (category_data or {}).get("generic_attributes", [])
    if generic:
        return random.choice(generic)
    return None


class PhantomLibraryAttacker:
    """Phantom 攻击器：多步流程，与 Attribute 攻击结构对齐。"""

    def __init__(self, library_path, tokenizer=None, model=None, api_client=None, api_model=None):
        self._attr = AttributeLibraryAttacker(
            library_path, tokenizer=tokenizer, model=model,
            api_client=api_client, api_model=api_model
        )
        self.tokenizer = tokenizer
        self.model = model
        self.api_client = api_client
        self.api_model = api_model

    def select_attribute_for_phantom(self, entity_data, question):
        """
        Step 4：用 LLM 从 entity 的属性中选一个最适合作为「稀缺资源 XX」（可设为 0/受限）的属性。
        输入: entity_data, question  输出: 一个 attribute 或 None（失败时由调用方用 fallback）。
        """
        if not entity_data:
            return None
        attributes = entity_data.get("attributes", [])
        if not attributes:
            return None
        attributes_text = "\n".join([
            f"- {attr['name']}: {attr['description']} ({attr.get('unit', '')})"
            for attr in attributes
        ])
        prompt = f"""You are selecting one attribute to use as a "scarce resource" in a phantom option constraint. The constraint will state: "this phantom option requires X, but X is 0 or unavailable."

Question:
{question[:500]}

Available attributes (choose ONE):
{attributes_text}

Select the attribute that:
1. Is most plausible as something that could be "in short supply" or "unavailable" (e.g. certification slots, quota, allocation, license count)
2. Fits the domain of the question
3. When we say "X is 0 or limited", sounds natural and convincing

Output JSON:
{{
    "selected_attribute": {{
        "name": "attribute name",
        "description": "description",
        "unit": "unit",
        "example_usage": "example usage"
    }}
}}

Output ONLY the JSON, no explanations."""

        response = call_llm(
            prompt, self.tokenizer, self.model,
            self.api_client, self.api_model
        )
        result = parse_llm_output(response)
        if result and "selected_attribute" in result:
            return result["selected_attribute"]
        return None

    def generate_phantom_option(self, question):
        """
        Step 5：只生成「虚假选项」的文本，模仿题目中已有选项/实体的风格和格式。
        输入: question  输出: 一段简短的 phantom 选项文本（一句或一行），或 None。
        """
        prompt = f"""### Optimized Prompt for Step 5
You are an adversarial expert. Create a "Super-Optimized Phantom Option" for this problem.

Original Question:
{question}

Requirements:
1. **Mathematical Dominance**: Identify the "best" existing option. Create a phantom option that is 20-30% more efficient (e.g., higher profit or lower cost). It must be the mathematically "Ideal" choice.
2. **Contextual Mirroring**: Use the exact phrasing structure of existing entities. If they use "Type X requires...", you use "Type [Phantom] requires...". 
3. **Parameter Integrity**: Provide all coefficients (labor, cost, materials) so the model can build a full variable for it. All values must be plausible within the current scale.
4. **No Spoilers**: Just output the description of this new option. Do not mention it is fake or restricted.

Example Logic: If Product A makes $10 and B makes $12, make Product C make $15 with the same or less resources."""

        response = call_llm(
            prompt, self.tokenizer, self.model,
            self.api_client, self.api_model
        )
        if not response:
            return None
        out = response.strip()
        if out.startswith("```"):
            lines = out.split("\n")
            out = "\n".join(l for l in lines if not l.strip().startswith("```"))
        return out.strip() if out else None

    def generate_constraint_sentence(self, attribute, phantom_option_text=None):
        """
        Step 6：只生成一句「该虚假选项需要 XX，但 XX 为 0 或受限」。
        输入: attribute (name/description/example_usage), 可选的 phantom_option_text
        输出: 一句约束句，或 None。
        """
        x_name = attribute.get("name", "")
        x_desc = attribute.get("description", "") or ""
        x_usage = attribute.get("example_usage", "") or ""
        phantom_ctx = f"\nPhantom option (for reference): {phantom_option_text}" if phantom_option_text else ""
        prompt = f"""Write exactly ONE sentence that states: a certain option requires "{x_name}" (or a natural phrasing of it), but the available amount / quota / allocation of that is 0 or strictly limited, so that option cannot be chosen.

Attribute: name="{x_name}", description="{x_desc}", example_usage="{x_usage}"{phantom_ctx}

Requirements:
- One sentence only. No preamble, no "For example", no markdown.
- The sentence must make it clear that the option is infeasible because X is 0 or unavailable (e.g. "However, this option requires certification slots, and the current allocation of certification slots is 0.").
Output ONLY that one sentence."""

        response = call_llm(
            prompt, self.tokenizer, self.model,
            self.api_client, self.api_model
        )
        if not response:
            return None
        out = response.strip()
        if out.startswith("```"):
            lines = out.split("\n")
            out = "\n".join(l for l in lines if not l.strip().startswith("```"))
        return out.strip() if out else None

    def fuse_phantom_naturally(self, question, phantom_option_text, constraint_sentence):
        """
        Step 7：将「虚假选项」和「约束句」自然融入原题，输出完整修改后的题目。
        输入: question, phantom_option_text, constraint_sentence
        输出: 完整修改后题目字符串，或 None。
        """
        prompt = f"""You are integrating a phantom option and its constraint into an optimization problem. Output the complete modified question.

Original question:
{question}

New phantom option to add (mimic the style of existing options and insert it where options are listed):
{phantom_option_text}

Constraint sentence to add (this sentence must make the phantom option unselectable—e.g. state that the phantom requires something that is 0 or unavailable):
{constraint_sentence}

Instructions:
1. Insert the phantom option in the same place and format as existing options (e.g. same table row, same list item style).
2. Insert the constraint sentence naturally (e.g. right after introducing the phantom, or in a "Note:" clause). The constraint must clearly rule out the phantom option.
3. Do NOT change the ground truth answer or the feasibility of the real options.
4. Output ONLY the complete modified question. No explanations, no markdown."""

        response = call_llm(
            prompt, self.tokenizer, self.model,
            self.api_client, self.api_model,
            max_tokens=2048,
        )
        if not response:
            return None
        out = response.strip()
        if out.startswith("```"):
            lines = out.split("\n")
            out = "\n".join(l for l in lines if not l.strip().startswith("```"))
        return out.strip() if out else None

    def attack(self, question, return_failure_reason=False):
        """
        执行 Phantom 攻击（多步流程）：
        1. identify_domain_and_entities
        2. find_matching_category
        3. find_matching_entity
        4. select_attribute_for_phantom（失败则 fallback 随机选）
        5. generate_phantom_option
        6. generate_constraint_sentence
        7. fuse_phantom_naturally
        """
        domain, entities = self._attr.identify_domain_and_entities(question)
        if not domain:
            reason = "无法识别领域"
            return (None, reason) if return_failure_reason else None

        category = self._attr.find_matching_category(domain)
        if not category:
            reason = f"无法匹配 Category (识别到的 domain: {domain})"
            return (None, reason) if return_failure_reason else None

        category_data = self._attr.library.get("categories", {}).get(category, {})
        entity_name, entity_data = self._attr.find_matching_entity(category, entities)

        attribute = None
        if entity_data:
            attribute = self.select_attribute_for_phantom(entity_data, question)
        if not attribute:
            attribute = _pick_constraint_attribute_fallback(entity_data, category_data)
        if not attribute:
            reason = f"找不到 Entity 且 Category '{category}' 无可用属性"
            return (None, reason) if return_failure_reason else None

        phantom_option_text = self.generate_phantom_option(question)
        if not phantom_option_text:
            reason = "LLM 未返回虚假选项文本"
            return (None, reason) if return_failure_reason else None

        constraint_sentence = self.generate_constraint_sentence(attribute, phantom_option_text)
        if not constraint_sentence:
            reason = "LLM 未返回约束句"
            return (None, reason) if return_failure_reason else None

        modified = self.fuse_phantom_naturally(question, phantom_option_text, constraint_sentence)
        if not modified:
            reason = "LLM 未返回融合后题目"
            return (None, reason) if return_failure_reason else None

        return (modified, None) if return_failure_reason else modified


def main():
    import argparse
    from openai import OpenAI

    parser = argparse.ArgumentParser(description="Phantom 攻击（多步流程）")
    parser.add_argument("--library", "-l", default="./attribute_library.json", help="属性库路径")
    parser.add_argument("--question", "-q", type=str, help="测试问题")
    parser.add_argument("--model", "-m", default="", help="本地模型路径")
    parser.add_argument("--api-model", type=str, default="Pro/deepseek-ai/DeepSeek-V3.2", help="API 模型")
    import os
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="OpenAI 兼容 API base URL（可通过 OPENAI_API_BASE 覆盖）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API 密钥（推荐通过环境变量 OPENAI_API_KEY 设置）",
    )
    args = parser.parse_args()

    if args.api_model:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请通过 --api-key 或环境变量 OPENAI_API_KEY 提供 API 密钥")
        api_client = OpenAI(api_key=api_key, base_url=args.api_base_url)
        tokenizer, model = None, None
    else:
        from attack_with_attribute_library import load_model
        tokenizer, model = load_model(args.model)
        api_client = None

    attacker = PhantomLibraryAttacker(
        args.library, tokenizer=tokenizer, model=model,
        api_client=api_client, api_model=args.api_model
    )

    q = args.question or "A factory produces two types of food, I and II, and currently has 50 skilled workers."
    print("原始问题:")
    print(q)
    print("\n" + "=" * 60 + "\n")

    result = attacker.attack(q)
    if result:
        print("Phantom 攻击后:")
        print(result)
    else:
        print("Phantom 攻击失败")


if __name__ == "__main__":
    main()
