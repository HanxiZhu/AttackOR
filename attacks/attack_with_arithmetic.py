#!/usr/bin/env python3
"""
Arithmetic 攻击：纯规则实现，将题目中的系数数值等值拆解为等价算术表达，不改变约束与答案。
- 排除：紧邻 $、¥、% 的数字（非系数语义）
- 严格等价模板：N = (N+M)-M、N = N*K/K、half of 2N、sum of a and (N-a) 等
- 类型：整数与浮点数分别处理，保持原题精度
"""
import re
import random
from typing import Tuple, Optional, List


# 数值范围：只替换 1~100 的整数或常见小数，避免过大数拆解冗长
MAX_VAL = 100000
MIN_VAL = 1


def _is_excluded_position(text: str, start: int, end: int, num_str: str) -> bool:
    """排除：前缀 $、¥，后缀 %，以及小数中的整数部分（如 2.5 里的 2）。"""
    if num_str.lstrip("0").replace(".", "") == "" or float(num_str) == 0:
        return True
    # 前缀：前一个非空白字符是 $ 或 ¥
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i >= 0 and text[i] in "$¥":
        return True
    # 后缀：后接 %（允许空格）
    j = end
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "%":
        return True
    # 小数中的整数部分：当前匹配为纯整数且紧跟小数点（如 2.5 中的 2）
    if "." not in num_str and end < len(text) and text[end] == ".":
        return True
    return False


def _integer_templates(n: int) -> List[str]:
    """整数 N 的动态等价表达，适配 0 到 1,000,000。"""
    if n < 0 or n > MAX_VAL:
        return []
    templates = []
    # 计算当前量级 (Magnitude) 以确定合适的拆解步长
    mag = 10 ** (len(str(n)) - 1) if n > 0 else 1
    steps = [mag, mag * 2, mag // 2] if mag >= 1 else [1]
    steps = [s for s in steps if s > 0]

    # N = (N+M) - M (减法模板)
    for m in steps:
        templates.append(f"the difference between {n + m} and {m}")
    
    # N = sum of a and (N-a) (加法模板)
    for m in steps:
        if n > m:
            templates.append(f"the sum of {m} and {n - m}")
            templates.append(f"a total of {n - m} plus {m}")

    # N = N/K * K (倍数模板)
    if n > 0:
        for k in [2, 5, 10]:
            if n % k == 0:
                templates.append(f"the product of {n // k} and {k}")
            templates.append(f"the result of {n * k} divided by {k}")
            
    # 特殊表达：一半
    if n % 2 == 0 and n > 0:
        templates.append(f"half of {2 * n}")
        
    return list(set(templates))


def _float_templates(f: float) -> List[str]:
    """浮点数 F 的等价表达，保持小数位数并适配量级。"""
    templates = []
    # 确定精度
    decimals = len(str(f).split(".")[-1]) if "." in str(f) else 0
    if decimals == 0:
        return _integer_templates(int(f))

    # 动态步长：根据小数点位数确定加减的颗粒度
    precision_step = 10 ** -decimals
    
    # F = (F+M) - M
    for factor in [1, 10, 100]:
        m = round(precision_step * factor, decimals)
        s = round(f + m, decimals)
        templates.append(f"the difference between {s} and {m}")

    # F = sum of a and (F-a)
    a = round(f / 2, decimals)
    b = round(f - a, decimals)
    if abs((a + b) - f) < 1e-10:
        templates.append(f"the sum of {a} and {b}")

    # 倍数表达
    twice = round(2 * f, decimals)
    templates.append(f"half of {twice}")
    
    return list(set(templates))


def _pick_replacement(num_str: str) -> Optional[str]:
    """根据数值字符串选一个等价替换（整数或浮点）。"""
    try:
        if "." in num_str:
            v = float(num_str)
            if v < MIN_VAL or v > MAX_VAL:
                return None
            opts = _float_templates(v)
        else:
            v = int(num_str)
            if v < MIN_VAL or v > MAX_VAL:
                return None
            opts = _integer_templates(v)
        return random.choice(opts) if opts else None
    except (ValueError, TypeError):
        return None


def attack(
    text: str,
    max_replacements: int = 1,
    random_seed: Optional[int] = None,
    return_failure_reason: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    对文本中的系数数值做等值算术拆解，不改变 $、¥、% 相关数字。
    返回 (新文本, 失败原因)；成功时失败原因为 None。
    """
    if random_seed is not None:
        random.seed(random_seed)

    # 匹配数字：整数或小数，且不匹配小数中的整数部分（整词）
    # \b\d+\b 会匹配 "2" 在 "2.5" 中，所以改用：匹配 \d+(?:\.\d+)? 且前面不是 \.
    pattern = re.compile(r"(?<![.\d])(\d+(?:\.\d+)?)(?![.\d])\b")
    matches = list(pattern.finditer(text))
    if not matches:
        if return_failure_reason:
            return None, "未找到可替换的数值"
        return None, None

    candidates = []
    for m in matches:
        num_str = m.group(1)
        if _is_excluded_position(text, m.start(), m.end(), num_str):
            continue
        repl = _pick_replacement(num_str)
        if repl is not None:
            candidates.append((m, repl))

    if not candidates:
        if return_failure_reason:
            return None, "无可替换的系数数值（均已排除或超出范围）"
        return None, None

    n = min(max_replacements, len(candidates))
    chosen = random.sample(candidates, n)
    chosen.sort(key=lambda x: x[0].start(), reverse=True)

    new_text = text
    for m, repl in chosen:
        new_text = new_text[: m.start()] + repl + new_text[m.end() :]

    if return_failure_reason:
        return new_text, None
    return new_text, None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Arithmetic 攻击：等值算术拆解")
    parser.add_argument("--text", type=str, default="A boat takes 20 minutes and a canoe 40 minutes. Budget is $100.")
    parser.add_argument("--max-replacements", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    out, reason = attack(args.text, max_replacements=args.max_replacements, random_seed=args.seed, return_failure_reason=True)
    if out is None:
        print("Failed:", reason)
    else:
        print("Original:", args.text)
        print("Attacked:", out)


if __name__ == "__main__":
    main()
