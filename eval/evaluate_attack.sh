#!/bin/bash
# ============================================================================
# 攻击数据集评测脚本
# ============================================================================
# 
# 用法:
#   bash evaluate_attack.sh <攻击数据集路径> [输出目录] [模型路径]
#
# 示例:
#   # 使用自动生成的输出目录
#   CUDA_VISIBLE_DEVICES=0 bash evaluate_attack.sh ./datasets_local/NL4OPT_test_adversarial_merged.jsonl
#   CUDA_VISIBLE_DEVICES=1 bash evaluate_attack.sh ./datasets_local/IndustryOR_test_adversarial_merged.jsonl
#   CUDA_VISIBLE_DEVICES=3 bash evaluate_attack.sh ./datasets_local/MAMO_complex_lp_adversarial_merged.jsonl
#   CUDA_VISIBLE_DEVICES=2 bash evaluate_attack.sh ./datasets_local/MAMO_easy_lp_adversarial_merged.jsonl

#   CUDA_VISIBLE_DEVICES=0 bash evaluate_attack.sh ./datasets_local/attack_or_verl_test.jsonl
#   CUDA_VISIBLE_DEVICES=1 bash evaluate_attack.sh ./datasets_local/MAMO_easy_test.jsonl
#   CUDA_VISIBLE_DEVICES=2 bash evaluate_attack.sh ./datasets_local/IndustryOR_test.jsonl
#   CUDA_VISIBLE_DEVICES=3 bash evaluate_attack.sh ./datasets_local/MAMO_complex_test.jsonl


#   
#   # 指定输出目录
#   bash evaluate_attack.sh ./datasets_local/IndustryOR_test_adversarial_unified.jsonl ./results_IndustryOR
#   
#   # 指定所有参数
#   bash evaluate_attack.sh \
#       ./datasets_local/MAMO_complex_lp_adversarial_unified.jsonl \
#       ./results_MAMO_complex \
#       /path/to/custom/model
#
# ============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 解析参数 ====================
if [ $# -lt 1 ]; then
    echo -e "${RED}错误: 缺少必需参数${NC}"
    echo ""
    echo "用法: bash evaluate_attack.sh <攻击数据集路径> [输出目录] [模型路径]"
    echo ""
    echo "示例:"
    echo "  bash evaluate_attack.sh ./datasets_local/NL4OPT_test_adversarial_unified.jsonl"
    echo "  bash evaluate_attack.sh ./datasets_local/IndustryOR_test_adversarial_unified.jsonl ./results_IndustryOR"
    exit 1
fi

# 参数1: 攻击数据集路径（必需）
ATTACK_DATASET="$1"

# 参数2: 输出目录（可选，默认根据数据集名称自动生成）
if [ -n "$2" ]; then
    OUTPUT_DIR="$2"
else
    # 从数据集文件名自动生成输出目录
    BASENAME=$(basename "$ATTACK_DATASET" .jsonl)
    OUTPUT_DIR="./results_${BASENAME}"
fi

# 参数3: 模型路径（可选）
# 参数3: 模型路径（可选）
MODEL_PATH="${3:-./outputs/grpo}"

# 派生的文件路径
GENERATED_CODE_FILE="${OUTPUT_DIR}/generated_attacks.jsonl"
EXECUTED_RESULTS_FILE="${OUTPUT_DIR}/executed.jsonl"
METRICS_FILE="${OUTPUT_DIR}/executed.metrics.json"

# ==================== 检查输入文件 ====================
if [ ! -f "$ATTACK_DATASET" ]; then
    echo -e "${RED}错误: 攻击数据集文件不存在: $ATTACK_DATASET${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ==================== 显示配置 ====================
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}攻击数据集评测工具${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "攻击数据集: ${GREEN}$ATTACK_DATASET${NC}"
echo -e "输出目录: $OUTPUT_DIR"
echo -e "模型路径: $MODEL_PATH"
echo -e "${BLUE}============================================${NC}"
echo ""

# ==================== 步骤1: 使用LLM生成代码 ====================
echo -e "${YELLOW}[1/2] 使用LLM为攻击数据集生成代码...${NC}"

python eval/generate_attack.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$ATTACK_DATASET" \
    --dataset_split "test" \
    --decoding_method "greedy" \
    --topk 1 \
    --save_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 代码生成失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 代码生成完成: $GENERATED_CODE_FILE${NC}"
echo ""

# ==================== 步骤2: 执行代码并计算指标 ====================
echo -e "${YELLOW}[2/2] 执行代码并按攻击类型统计指标...${NC}"

python eval/execute_attack.py \
    --input_file "$GENERATED_CODE_FILE" \
    --output_file "$EXECUTED_RESULTS_FILE" \
    --question_field "question" \
    --answer_field "answer" \
    --timeout 60 \
    --max_workers 128 \
    --numerical_err_tolerance 0.05

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 代码执行失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 执行完成${NC}"
echo ""

# ==================== 显示结果 ====================
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}评估结果${NC}"
echo -e "${BLUE}============================================${NC}"

if [ -f "$METRICS_FILE" ]; then
    cat "$METRICS_FILE"
    echo ""
else
    echo -e "${YELLOW}警告: 未找到指标文件${NC}"
    echo ""
fi

echo -e "${BLUE}============================================${NC}"
echo -e "结果文件:"
echo -e "  - 生成代码: ${GREEN}$GENERATED_CODE_FILE${NC}"
echo -e "  - 执行结果: ${GREEN}$EXECUTED_RESULTS_FILE${NC}"
echo -e "  - 评估指标: ${GREEN}$METRICS_FILE${NC}"
echo -e "${BLUE}============================================${NC}"

