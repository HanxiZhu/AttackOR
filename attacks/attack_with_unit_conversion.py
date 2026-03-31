#!/usr/bin/env python3
"""
Unit Conversion 攻击（独立脚本，与 generate_unified_attacks.py 解耦）
改进点：
1. 支持逗号数字（千分位），避免“10,000 units”→ 0 milli-units
2. 扩展 unit_map 覆盖更多单位（yuan、years、km/h 等）
3. 改进 last_sentence 逻辑，保证至少保留一条锚定
4. 禁止 0 milli-units / 0 cents 的错误插入
5. factor=1 的映射单独标注，便于评估时分层
"""
import re
from pathlib import Path
from typing import Optional, Tuple, List


# ============ 配置项：可在 attack.py 中通过参数覆盖 ============
UNIT_MAP_EXPANSION = {
    # IndustryOR/MAMO 常见
    "yuan": (100, "cents"),
    "Yuan": (100, "cents"),
    "years": (12, "months"),
    "kilometers": (1000, "meters"),
    "km/h": (60, "minutes"),  # 简化，每 1 km/h = 60 min/km
    "kg/h": (60, "minutes"),
    "MW": (1e6, "kW"),  # 1 MW = 1000 kW
    "TB": (1e12, "GB"),
    "Tbps": (1e9, "Gbps"),
}


def _is_invalid_zero_insertion(unit: str, new_val: float) -> bool:
    """禁止 0 milli-units / 0 cents 的错误插入"""
    return (unit == "milli-units" and abs(new_val) < 1e-6) or (unit == "cents" and abs(new_val) < 0.01)


def _build_anchor_constraints(attacked_units: List[Tuple[str, str]], baseline_question: str) -> List[str]:
    """
    B 方案：至少保留一条锚定。优先选「最后一句中未出现 in X」的 (orig, dist)，
    若全被过滤则仍取第一条加入约束。
    """
    if not attacked_units:
        return []
    parts = baseline_question.strip().split(".")
    last_sentence = parts[-2].strip().lower() if len(parts) >= 2 else baseline_question.lower()
    constraints = []
    for orig, dist in attacked_units:
        dist_core = (dist.split()[-1] if dist else "").lower()
        if dist_core and f"in {dist_core}" not in last_sentence:
            orig_display = "percentages" if orig == "%" else orig
            constraints.append(f"use {orig_display} (not {dist})")
    if not constraints:
        orig, dist = attacked_units[0]
        orig_display = "percentages" if orig == "%" else orig
        constraints.append(f"use {orig_display} (not {dist})")
    return constraints


# 默认 unit_map / blacklist（与 generate_unified_attacks 一致 + 扩展）
DEFAULT_UNIT_MAP = {
    "gram": (1000, "mg"), "grams": (1000, "mg"),
    "kg": (1000, "grams"), "kilogram": (1000, "grams"), "kilograms": (1000, "grams"),
    "lb": (0.4536, "kg"), "lbs": (0.4536, "kg"), "pound": (0.4536, "kg"), "pounds": (0.4536, "kg"),
    "mg": (0.001, "grams"), "milligram": (0.001, "grams"), "milligrams": (0.001, "grams"),
    "ton": (1000, "kg"), "tons": (1000, "kg"),
    "minute": (60, "seconds"), "minutes": (60, "seconds"),
    "hour": (60, "minutes"), "hours": (60, "minutes"),
    "day": (24, "hours"), "days": (24, "hours"),
    "week": (7, "days"), "weeks": (7, "days"),
    "calorie": (4.184, "kJ"), "calories": (4.184, "kJ"),
    "kwh": (3600, "kJ"), "kilowatt-hour": (3600, "kJ"),
    "ml": (0.001, "liters"), "milliliter": (0.001, "liters"), "milliliters": (0.001, "liters"),
    "liter": (1000, "ml"), "liters": (1000, "ml"),
    "gallon": (3.785, "liters"), "gallons": (3.785, "liters"),
    "km": (1000, "meters"), "kilometer": (1000, "meters"),
    "m": (100, "cm"), "meter": (100, "cm"), "meters": (100, "cm"),
    "cm": (10, "mm"),
    "percent": (0.01, "ratio"), "percentage": (0.01, "ratio"),
    "unit": (1000, "milli-units"), "units": (1000, "milli-units"),
    "person": (8, "work-hours"), "people": (8, "work-hours"),
    "worker": (8, "work-hours"), "workers": (8, "work-hours"),
    "employee": (8, "work-hours"), "employees": (8, "work-hours"),
    "container": (1, "storage-unit"), "containers": (1, "storage-unit"),
    "crate": (1, "box"), "crates": (1, "box"),
    "box": (1, "package"), "boxes": (1, "package"),
    "item": (1, "piece"), "items": (1, "piece"),
    "coin": (1, "token"), "coins": (1, "token"),
    "pill": (1, "tablet"), "pills": (1, "tablet"),
    "sandwich": (1, "serving"), "sandwiches": (1, "serving"),
    "cone": (1, "serving"), "cones": (1, "serving"),
    "duck": (1, "bird"), "ducks": (1, "bird"),
    "grape": (1, "fruit"), "grapes": (1, "fruit"),
    "wire": (1, "cable"), "wires": (1, "cable"),
}
DEFAULT_UNIT_BLACKLIST = {
    "and", "or", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "produces", "requires", "causes", "provides", "contains", "needs", "takes",
    "hot", "cold", "small", "large", "cheap", "expensive", "big", "tiny",
    "red", "blue", "green", "yellow", "black", "white",
    "first", "second", "third", "last", "next", "previous",
}


# ============ 主攻击函数 ============
def _attack_impl(
    text: str,
    unit_map: dict,
    unit_blacklist: set,
    max_unit_conversions: int = 2,
    expand_unit_map: bool = False,
    return_failure_reason: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    单位换算攻击，支持扩展 unit_map 与逗号数字。

    返回： (new_text, failure_reason)
    """
    new_text = text
    attacked_units = []
    conversion_count = 0

    # 1. 扩展 unit_map
    if expand_unit_map:
        unit_map = {**unit_map, **UNIT_MAP_EXPANSION}

    # 2. 金钱处理（$ + 数字）- 保持原有逻辑
    money_pattern = re.compile(r"(\$)\s*(\d+(?:\.\d+)?)\b")
    for match in reversed(list(money_pattern.finditer(new_text))):
        try:
            val = float(match.group(2))
            cents = int(val * 100)
            insertion = f" (i.e., {cents} cents)"
            new_text = new_text[:match.end()] + insertion + new_text[match.end():]
        except (ValueError, ZeroDivisionError):
            continue
        attacked_units.append(("dollars", "cents"))

    # 3. 匹配 number+unit（支持逗号，但跳过 $ 后的数字）
    # 改进正则：支持千分位（\d{1,3}(?:,\d{3})*(?:\.\d+)?），或连续数字
    # 排除紧邻 $ 的数字（避免重复金钱处理）
    number_unit_pattern = re.compile(r"(?<!\$)\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*([a-zA-Z%]+)")
    matches = list(number_unit_pattern.finditer(new_text))
    matches.sort(key=lambda m: m.start(), reverse=True)

    for match in matches:
        if conversion_count >= max_unit_conversions:
            break

        val_str = match.group(1).replace(",", "")  # 去掉逗号
        unit_raw = match.group(2)
        unit_lower = unit_raw.lower() if unit_raw != "%" else "percent"

        if unit_lower in unit_blacklist:
            continue

        if unit_lower == "%":
            unit_lower = "percent"

        if unit_lower not in unit_map:
            continue

        try:
            val = float(val_str)
            factor, distraction_unit = unit_map[unit_lower]

            # 禁止 0 milli-units / 0 cents 的错误插入
            if _is_invalid_zero_insertion(distraction_unit, val * factor):
                if return_failure_reason:
                    return None, f"0 {distraction_unit} insertion blocked (value={val})"
                continue

            new_val = round(val * factor, 4)
            if new_val.is_integer():
                new_val = int(new_val)

            if unit_lower in ["percent", "percentage"]:
                insertion = f" (or {new_val} ratio)"
            else:
                insertion = f" (equivalent to {new_val} {distraction_unit})"

            end_pos = match.end()
            new_text = new_text[:end_pos] + insertion + new_text[end_pos:]

            if (unit_raw, distraction_unit) not in attacked_units:
                attacked_units.append((unit_raw, distraction_unit))
            conversion_count += 1
        except (ValueError, TypeError):
            continue

    # 4. 若未有任何转换，返回失败
    if not attacked_units:
        reason = "No units could be converted"
        return (None, reason) if return_failure_reason else (None, reason)

    # 5. 生成锚定（B 方案：至少保留一条）
    constraints = _build_anchor_constraints(attacked_units, text)
    anchor_msg = " Important: Please ensure the final calculations " + " and ".join(constraints) + "."
    return (new_text + anchor_msg, None)


def attack(
    text: str,
    max_unit_conversions: int = 2,
    expand_unit_map: bool = True,
    return_failure_reason: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """对外接口：使用默认 unit_map/blacklist，返回 (new_text, failure_reason)。"""
    return _attack_impl(
        text,
        unit_map=DEFAULT_UNIT_MAP,
        unit_blacklist=DEFAULT_UNIT_BLACKLIST,
        max_unit_conversions=max_unit_conversions,
        expand_unit_map=expand_unit_map,
        return_failure_reason=return_failure_reason,
    )


# ============ 命令行入口（可独立运行） ============
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Unit Conversion 攻击（支持逗号数字、扩展 unit_map、改进锚定逻辑）"
    )
    parser.add_argument("--text", type=str, help="待攻击的文本")
    parser.add_argument("--max-unit-conversions", type=int, default=2, help="最大单位转换次数")
    parser.add_argument("--expand-unit-map", action="store_true", help="扩展 unit_map（增加 yuan、years 等）")
    parser.add_argument("--output", "-o", type=str, help="输出文件路径（可选）")
    parser.add_argument("--return-failure-reason", action="store_true", help="返回失败原因")
    args = parser.parse_args()

    input_text = args.text or ""
    if args.output and not input_text:
        try:
            input_text = Path(args.output).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"Error: Input file not found: {args.output}")
            return

    result_text, failure_reason = attack(
        input_text,
        max_unit_conversions=args.max_unit_conversions,
        expand_unit_map=args.expand_unit_map,
        return_failure_reason=True,
    )

    if failure_reason is None:
        print("=== Attack Result ===\nSuccess: True")
        print(result_text[:500] + "..." if len(result_text) > 500 else result_text)
        if args.output:
            Path(args.output).write_text(result_text, encoding="utf-8")
    else:
        print("=== Attack Result ===\nSuccess: False")
        print(f"Reason: {failure_reason}")


if __name__ == "__main__":
    main()
