import json
import re
import os
import tempfile
import subprocess
import time
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

# 本地 LLM 模型路径（用于防御逻辑评估）
# 用户需要设置此路径指向本地的 llama3-7b 模型
# 优先从环境变量 LOCAL_LLM_MODEL_PATH_ENV 获取（由训练脚本设置）
# 其次从 LOCAL_LLM_MODEL_PATH 获取
LOCAL_LLM_MODEL_PATH = os.environ.get("LOCAL_LLM_MODEL_PATH_ENV") or os.environ.get("LOCAL_LLM_MODEL_PATH", "/path/to/llama3-7b")

# 本地 LLM 服务地址（vllm OpenAI 兼容 API）
# 如果设置了此环境变量，将使用指定的服务地址
# 否则将尝试从 Ray 获取 rollout 服务地址
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", None)


def _maybe_debug_log_line(payload: dict) -> None:
    p = os.environ.get("ATTACK_OR_DEBUG_LOG")
    if not p:
        return
    try:
        with open(p, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# Tokenizer缓存（lazy loading）
_tokenizer_cache = None
_tokenizer_model_path = None

def extract_python_code(solution_str):
    """提取标准 Markdown 格式的 python 代码块"""
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, solution_str, re.DOTALL)
    return match.group(1) if match else None

def extract_reasoning(solution_str):
    """剔除代码块后，获取思维链/建模分析素材"""
    return re.sub(r"```python.*?```", "", solution_str, flags=re.DOTALL)

def compute_length_reward(total_tokens, config=None, is_chars=False):
    """
    长度惩罚：使用渐进式惩罚，超过阈值越多惩罚越大
    注意：max_response_length=4000 tokens，使用更低的阈值和更强的惩罚
    来更有效地阻止模型生成长文本
    
    Args:
        total_tokens: 总文本token数（如果is_chars=True，则是字符数）
        config: 忽略（保留参数以兼容旧代码）
        is_chars: 如果True，表示total_tokens实际上是字符数，需要调整阈值
    
    Returns:
        float: 长度惩罚值（负值，超过阈值越多惩罚越大）
    """
    # 如果is_chars=True，说明tokenizer获取失败，回退到字符数
    # 对于中文，1字符≈1-2 tokens，所以3000字符≈1500-3000 tokens
    if is_chars:
        threshold = 4000  # 字符数阈值（降低以更早惩罚）
        max_threshold = 6000  # 最大阈值（对应约3000-6000 tokens）
    else:
        threshold = 2000  # token数阈值（从2000降低到1500，更早惩罚）
        max_threshold = 4000  # 最大阈值（超过此值使用最大惩罚）
    
    if total_tokens <= threshold:
        return 0.0
    elif total_tokens <= max_threshold:
        # 渐进式惩罚：在threshold和max_threshold之间线性增加
        # 从-2.0到-3.0
        penalty = -2.0 - (total_tokens - threshold) / (max_threshold - threshold) * 1.0
        return penalty
    else:
        # 超过max_threshold使用最大惩罚-3.0
        return -3.0

def get_tokenizer(model_path=None, **kwargs):
    """
    获取tokenizer（lazy loading）
    
    Args:
        model_path: 模型路径（可选，优先从kwargs获取）
        **kwargs: 可能包含 tokenizer 或 actor_model_path
    
    Returns:
        tokenizer实例
    """
    global _tokenizer_cache, _tokenizer_model_path
    
    # 如果kwargs中有tokenizer，直接使用
    if 'tokenizer' in kwargs and kwargs['tokenizer'] is not None:
        return kwargs['tokenizer']
    
    # 确定模型路径
    if model_path is None:
        model_path = kwargs.get('actor_model_path') or kwargs.get('model_path')
    
    if model_path is None:
        # 尝试从环境变量获取（actor模型路径）
        model_path = os.environ.get('ACTOR_MODEL_PATH') or os.environ.get('MODEL_PATH')
    
    if model_path is None:
        raise ValueError("无法获取模型路径，请通过kwargs传递tokenizer或model_path")
    
    # 如果已经加载过相同路径的tokenizer，直接返回
    if _tokenizer_cache is not None and _tokenizer_model_path == model_path:
        return _tokenizer_cache
    
    # 加载tokenizer
    try:
        _tokenizer_cache = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _tokenizer_model_path = model_path
        return _tokenizer_cache
    except Exception as e:
        raise ValueError(f"无法加载tokenizer from {model_path}: {e}")

def execute_solver_code(code_str):
    """在沙箱执行代码，精准适配 coptpy 日志特征以抓取数值"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code_str)
        temp_path = f.name

    try:
        # 必须设置超时时间 (5秒)
        result = subprocess.run(
            ["python3", temp_path], 
            capture_output=True, 
            text=True, 
            timeout=5.0
        )
        os.remove(temp_path)
        
        if result.returncode != 0:
            return None
            
        # 针对 coptpy 标准输出格式： "Best solution   : 2190.000000000"
        copt_match = re.search(r"Best solution\s*:\s*([-+]?\d*\.\d+)", result.stdout)
        if copt_match:
            return float(copt_match.group(1))
            
        # 兜底抓取
        nums = re.findall(r"[-+]?(?:\d*\.*\d+)", result.stdout)
        return float(nums[-1]) if nums else None
        
    except subprocess.TimeoutExpired:
        os.remove(temp_path)
        return None
    except Exception:
        os.remove(temp_path)
        return None

async def audit_defense_logic(reasoning_str, defense_logic, local_llm_base_url=None, local_llm_model_path=None):
    """
    语义审计：判断推理文本是否体现了正确的排雷逻辑
    使用本地部署的 LLM（通过 vllm）进行评估
    
    Args:
        reasoning_str: 模型推理文本
        defense_logic: 需要的防御逻辑
        local_llm_base_url: 本地 LLM 服务地址（可选，优先使用此参数）
        local_llm_model_path: 本地 LLM 模型路径（用于指定模型名称）
        max_reasoning_chars: 最大推理文本长度（字符数），默认2000，防止超过vLLM上下文限制
    
    Returns:
        bool: True 如果推理体现了正确的防御逻辑，False 否则
    """
    prompt = f"""You are an OR auditing expert.
Defense Logic Required: {defense_logic}
Model Reasoning: {reasoning_str}

Did the Model Reasoning explicitly identify and correctly handle the required defense logic? 
Answer purely with [YES] or [NO]."""

    # 确定服务地址：优先使用参数，其次环境变量
    # 注意：空字符串被视为未设置
    base_url = local_llm_base_url if local_llm_base_url else (LOCAL_LLM_BASE_URL if LOCAL_LLM_BASE_URL else None)
    model_name = local_llm_model_path if local_llm_model_path else LOCAL_LLM_MODEL_PATH
    
    # 必须指定 base_url（使用独立的 llama3-7b 服务）
    if base_url is None:
        print("Error: LOCAL_LLM_BASE_URL must be set when using independent llama3-7b service")
        print("Please set LOCAL_LLM_BASE_URL environment variable or pass local_llm_base_url parameter")
        print("Example: export LOCAL_LLM_BASE_URL='http://localhost:8000/v1'")
        return False

    try:
        # 使用 AsyncOpenAI 客户端连接本地 vllm 服务
        client = AsyncOpenAI(
            api_key="not-needed",  # vllm 不需要真实的 API key
            base_url=base_url
        )
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        reply = response.choices[0].message.content.strip()
        return "[YES]" in reply.upper()
    except Exception as e:
        print(f"Local LLM API Error: {e}")
        return False

async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    verl 调用的主入口函数：采用平滑累加机制，使用相对误差判断准确性
    支持异步调用，可以使用本地 LLM 评估防御逻辑
    
    Args:
        data_source: 数据源名称
        solution_str: 模型生成的解决方案字符串
        ground_truth: 正确答案
        extra_info: 额外信息字典，包含 attack_type 和 defense_logic
        **kwargs: 额外参数，可能包含 local_llm_base_url 和 local_llm_model_path
    """
    # 获取tokenizer并计算token数（用于长度惩罚）
    is_chars_fallback = False
    try:
        tokenizer = get_tokenizer(**kwargs)
        total_tokens = len(tokenizer.encode(solution_str, add_special_tokens=False))
        # Debug logging: tokenizer获取成功
        try:
            debug_log = {
                "message": "tokenizer_success",
                "data": {
                    "has_tokenizer": True,
                    "total_tokens": total_tokens,
                    "solution_length_chars": len(solution_str),
                    "kwargs_keys": list(kwargs.keys())
                }
            }
            _maybe_debug_log_line(debug_log)
        except Exception:
            pass
    except Exception as e:
        # 如果无法获取tokenizer，回退到字符数（但会给出警告）
        print(f"Warning: 无法获取tokenizer，回退到字符数计算: {e}")
        total_tokens = len(solution_str)  # 临时回退，但这不是准确的token数
        is_chars_fallback = True
        # Debug logging: tokenizer获取失败
        try:
            debug_log = {
                "message": "tokenizer_fallback",
                "data": {
                    "has_tokenizer": False,
                    "error": str(e)[:200],
                    "total_tokens_fallback": total_tokens,
                    "solution_length_chars": len(solution_str),
                    "kwargs_keys": list(kwargs.keys())
                }
            }
            _maybe_debug_log_line(debug_log)
        except Exception:
            pass
    
    # 长度惩罚：超过阈值扣1分（根据是token数还是字符数使用不同阈值）
    length_penalty = compute_length_reward(total_tokens, is_chars=is_chars_fallback)
    
    # 初始化 reward 为长度惩罚（可以为负，这样失败的长输出会被惩罚）
    reward = length_penalty
    
    if extra_info is None:
        return reward

    attack_type = extra_info.get('attack_type', 'none')
    defense_logic = extra_info.get('defense_logic', '')
    
    # 阶梯 1: 格式分 (+0.2)
    code_str = extract_python_code(solution_str)
    if not code_str:
        return reward  # 现在失败也会带着长度惩罚返回
    reward += 0.2
        
    # 阶梯 2: 运行分 (+0.3)
    model_result = execute_solver_code(code_str)
    if model_result is None:
        return reward  # 现在失败也会带着长度惩罚返回
    reward += 0.3
        
    # 阶梯 3: 准确分 (+1.0) -> 使用相对误差 (Relative Error)
    # 与ORLM评测脚本保持一致：使用5%的相对误差容忍度
    gt_val_str = str(ground_truth).strip()
    model_result_str = str(model_result).strip() if model_result is not None else None
    
    # 处理 "No Best Solution" 情况（与 execute_attack.py 一致）
    if "No Best Solution" in gt_val_str:
        is_correct = (model_result_str is not None and "No Best Solution" in model_result_str)
    else:
        # 数值比较：使用5%的相对误差容忍度（与ORLM评测脚本一致）
        try:
            gt_float = float(gt_val_str)
            model_float = float(model_result_str) if model_result_str else None
            
            if model_float is None:
                is_correct = False
            elif abs(gt_float) < 1e-9:
                # 避免除以0错误：如果ground truth接近0，使用绝对误差
                is_correct = abs(model_float) <= 0.05
            else:
                # 使用相对误差：容忍5%的偏差
                relative_error = abs((model_float - gt_float) / gt_float)
                is_correct = relative_error <= 0.05
        except (ValueError, TypeError):
            # 如果转浮点失败，回退到字符串精确匹配
            is_correct = (gt_val_str == model_result_str)
    
    # Debug logging: Accuracy check
    try:
        # 计算相对误差用于日志（如果可能）
        relative_error_for_log = None
        if model_result is not None and "No Best Solution" not in gt_val_str:
            try:
                gt_float = float(gt_val_str)
                model_float = float(str(model_result))
                if abs(gt_float) >= 1e-9:
                    relative_error_for_log = abs((model_float - gt_float) / gt_float)
            except (ValueError, TypeError):
                pass
        
        debug_log = {}  # 初始化 debug_log（如果之前未定义）
        debug_log["message"] = "accuracy_check"
        debug_log["data"] = {
            "model_result": model_result_str,
            "ground_truth": gt_val_str,
            "relative_error": relative_error_for_log,
            "tolerance": 0.05,
            "is_correct": is_correct,
            "reward_after_acc": reward + (1.0 if is_correct else 0.0)
        }
        _maybe_debug_log_line(debug_log)
    except Exception:
        pass
    
    if not is_correct:
        return reward  # 现在失败也会带着长度惩罚返回
    reward += 1.0
        
    # 阶梯 4: 逻辑防守溢价分 (+1.0)
    if attack_type != 'baseline':
        reasoning_str = extract_reasoning(solution_str)
        # 异步调用本地 LLM 评估防御逻辑
        local_llm_base_url = kwargs.get('local_llm_base_url')
        local_llm_model_path = kwargs.get('local_llm_model_path')
        is_defended = await audit_defense_logic(
            reasoning_str, 
            defense_logic,
            local_llm_base_url=local_llm_base_url,
            local_llm_model_path=local_llm_model_path
        )
        
        if is_defended:
            reward += 1.0
    
    # 注意：长度奖励已在函数开头计算并添加到 reward 中
    # 不再需要在这里重复计算，也不再使用 max(0.0, ...) 限制
    # 允许负 reward，这样失败的长输出会被有效惩罚
    
    # Debug logging: Length penalty
    try:
        debug_log = {}  # 初始化 debug_log（如果之前未定义）
        debug_log["message"] = "length_penalty"
        debug_log["data"] = {
            "total_tokens": total_tokens,
            "is_chars_fallback": is_chars_fallback,
            "length_penalty": length_penalty,
            "reward_without_length": reward - length_penalty,  # 当前 reward 减去长度惩罚，得到不含长度的部分
            "final_reward": reward,
            "threshold": 3000 if is_chars_fallback else 1500,
            "max_threshold": 6000 if is_chars_fallback else 3000
        }
        _maybe_debug_log_line(debug_log)
    except Exception:
        pass
    
    # Debug logging: Final result
    try:
        debug_log = {}  # 初始化 debug_log（如果之前未定义）
        debug_log["message"] = "final"
        debug_log["data"] = {
            "final_reward": reward,
            "attack_type": attack_type,
            "total_tokens": total_tokens
        }
        _maybe_debug_log_line(debug_log)
    except Exception:
        pass
            
    return reward