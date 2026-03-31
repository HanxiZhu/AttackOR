#!/bin/bash
# ORLM Training Script
# 基于 README.md 中的训练配方创建
# 使用方法: ./train.sh 或 bash train.sh

set -e  # 遇到错误立即退出

export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${RANDOM}"
# ==========================================
# 配置参数 - 请根据你的实际情况修改
# ==========================================

# 基础模型路径（可以是 HuggingFace 模型 ID 或本地路径）
# 示例: "meta-llama/Llama-3-8B" 或 "Qwen/Qwen2.5-9B" 或 "/path/to/your/model"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-9B}"

# 训练数据路径（JSONL 格式）
# 可以是单个文件或包含多个 jsonl 文件的目录
DATA_PATH="${DATA_PATH:-./datasets_local/train_data.jsonl}"

# 模型保存路径
SAVE_PATH="${SAVE_PATH:-./outputs/sft}"

# ==========================================
# GPU 和批次配置
# ==========================================

# GPU 数量
NUM_GPUS=8

# 每个 GPU 的 batch size
BATCH_SIZE_PER_GPU=2

# 总 batch size（梯度累积后的有效 batch size）
# 建议值: 16-64，根据你的数据集大小调整
TOTAL_BATCH_SIZE=128

# 数据预处理进程数
# 0 表示实时处理，>0 表示多进程预处理（更快但占用更多内存）
PREPROCESSING_NUM_WORKERS=8

# 计算梯度累积步数
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

# ==========================================
# 训练超参数
# ==========================================

# 最大序列长度
# LLaMA-3-8B 支持 8192，Qwen2.5-9B 支持更长序列
MAX_SEQ_LENGTH=14096

# 学习率

LEARNING_RATE=2e-5

# 训练轮数
NUM_TRAIN_EPOCHS=3

# 预热比例
WARMUP_RATIO=0.03

# ==========================================
# DeepSpeed 配置
# ==========================================

# 根据显存情况选择配置:
# - stage3_no_offloading_bf16.json: 显存充足时使用（最快，推荐）
# - stage3_offloading_bf16.json: 显存不足时使用（会慢一些）
# - stage3_no_offloading_fp16.json: 不支持 bf16 的 GPU 使用
# - stage3_offloading_fp16.json: 显存不足且不支持 bf16 时使用
DEEPSPEED_CONFIG="sft/configs/stage3_no_offloading_bf16.json"

# ==========================================
# 可选: LoRA 配置（参数高效微调）
# ==========================================

# 是否使用 LoRA（设置为 true 启用）
USE_LORA=false

# LoRA 参数（仅在 USE_LORA=true 时生效）
LORA_RANK=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
# LoRA target modules（不同模型可能需要调整）
# LLaMA/Qwen: "[q_proj,k_proj,v_proj,o_proj]"
# 其他模型请查看模型结构
LORA_TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"

# ==========================================
# 其他配置
# ==========================================

# 限制训练样本数量（用于快速测试，设为 null 或注释掉表示使用全部数据）
# MAX_TRAIN_SAMPLES=1000

# 随机种子（用于可复现性）
SEED=42

# ==========================================
# 打印配置信息
# ==========================================

echo "=========================================="
echo "ORLM 训练配置"
echo "=========================================="
echo "模型: $MODEL_NAME_OR_PATH"
echo "数据: $DATA_PATH"
echo "输出: $SAVE_PATH"
echo "GPU 数量: $NUM_GPUS"
echo "Batch Size: $TOTAL_BATCH_SIZE (每GPU: $BATCH_SIZE_PER_GPU, 累积: $GRADIENT_ACC_STEPS)"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_TRAIN_EPOCHS"
echo "最大序列长度: $MAX_SEQ_LENGTH"
echo "DeepSpeed 配置: $DEEPSPEED_CONFIG"
if [ "$USE_LORA" = true ]; then
    echo "LoRA: 启用 (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
else
    echo "LoRA: 禁用 (全量微调)"
fi
echo "=========================================="
echo ""

# ==========================================
# 检查必要文件
# ==========================================

if [ ! -f "$DATA_PATH" ] && [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    echo "请检查 DATA_PATH 配置是否正确"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "错误: DeepSpeed 配置文件不存在: $DEEPSPEED_CONFIG"
    echo "请检查 DEEPSPEED_CONFIG 配置是否正确"
    exit 1
fi

# ==========================================
# 进入项目目录
# ==========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "工作目录: $(pwd)"
echo "开始训练..."
echo ""

# ==========================================
# 构建训练命令
# ==========================================

# 1. 声明 TRAIN_CMD 变量，并使用当前环境的 python 启动分布式训练
TRAIN_CMD="python -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS \
    -m sft.finetune \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_dataset_name_or_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --eval_strategy \"no\" \
    --save_strategy \"no\" \
    --save_total_limit 1 \
    --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS \
    --ddp_timeout 14400 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio $WARMUP_RATIO \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --logging_steps 1 \
    --report_to \"tensorboard\" \
    --gradient_checkpointing True \
    --deepspeed $DEEPSPEED_CONFIG \
    --bf16 True"

# 添加 LoRA 参数（如果启用）
if [ "$USE_LORA" = true ]; then
    TRAIN_CMD="$TRAIN_CMD \
    --use_lora True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES"
fi

# 添加 max_train_samples（如果设置了）
if [ ! -z "$MAX_TRAIN_SAMPLES" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_train_samples $MAX_TRAIN_SAMPLES"
fi

# ==========================================
# 执行训练
# ==========================================

echo "执行训练命令..."
echo ""
eval $TRAIN_CMD