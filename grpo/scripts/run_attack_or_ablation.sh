#!/bin/bash
# GRPO 消融实验训练脚本 - 支持指定 attack_type 进行训练
# 用法: ATTACK_TYPES="llm_attribute,llm_phantom,arithmetic,baseline" bash run_attack_or_ablation.sh [ENGINE] [ACTOR_MODEL_PATH] [LOCAL_LLM_MODEL_PATH]
# 
# 示例:
#   # 使用 unit_conversion 和 llm_attribute 两种攻击类型
#   ATTACK_TYPES="baseline,arithmetic,unit_conversion,llm_attribute" bash run_attack_or_ablation.sh
#   ATTACK_TYPES="baseline,llm_phantom,unit_conversion,llm_attribute" bash run_attack_or_ablation.sh
#
#   # 使用所有攻击类型（完整训练）
#   ATTACK_TYPES="all" bash run_attack_or_ablation.sh
#
#   # 使用 baseline 和 unit_conversion
#   ATTACK_TYPES="baseline,unit_conversion" bash run_attack_or_ablation.sh

set -x

# ============================================================================
# 消融实验配置
# ============================================================================

# 要使用的攻击类型，逗号分隔
# 设置为 "all" 表示使用所有攻击类型（等同于原始完整训练）
# 可选值: unit_conversion, llm_attribute, llm_phantom, arithmetic, baseline
ATTACK_TYPES=${ATTACK_TYPES:-"all"}

# 是否在训练完成后保留过滤后的数据文件
# 设为 "true" 保留，设为 "false" 删除
KEEP_FILTERED_DATA=${KEEP_FILTERED_DATA:-"true"}

# ============================================================================
# 模型路径配置 - 用户需要根据实际情况修改以下路径
# ============================================================================

# 引擎类型
ENGINE=${1:-vllm}

# Actor 模型路径（orlm，基于 llama3-7b 的运筹学模型）
ACTOR_MODEL_PATH=${2:-"./outputs/sft"}

# 本地 LLM 模型路径（用于防御逻辑评估）
LOCAL_LLM_MODEL_PATH=${3:-"meta-llama/Llama-3.1-8B-Instruct"}

# ============================================================================
# 项目路径配置
# ============================================================================

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# 数据处理
# ============================================================================

# 原始数据路径
ORIGINAL_TRAIN_DATA="$PROJECT_ROOT/data/merged_train.parquet"
ORIGINAL_VAL_DATA="$PROJECT_ROOT/data/merged_val.parquet"

# 根据攻击类型生成后缀
if [ "$ATTACK_TYPES" = "all" ]; then
    # 使用所有攻击类型，不需要过滤
    TRAIN_DATA="$ORIGINAL_TRAIN_DATA"
    VAL_DATA="$ORIGINAL_VAL_DATA"
    DATA_SUFFIX=""
    FILTERED_TRAIN_DATA=""
    FILTERED_VAL_DATA=""
else
    # 生成数据文件后缀
    DATA_SUFFIX="_$(echo $ATTACK_TYPES | tr ',' '_')"
    
    # 过滤后的数据路径
    FILTERED_TRAIN_DATA="$PROJECT_ROOT/data/merged_train${DATA_SUFFIX}.parquet"
    FILTERED_VAL_DATA="$PROJECT_ROOT/data/merged_val${DATA_SUFFIX}.parquet"
    
    # 检查是否需要重新过滤（如果原始数据比过滤数据新，或不存）
    NEED_FILTER=false
    
    if [ ! -f "$FILTERED_TRAIN_DATA" ]; then
        NEED_FILTER=true
        echo "过滤后的训练数据不存在，需要过滤..."
    elif [ "$ORIGINAL_TRAIN_DATA" -nt "$FILTERED_TRAIN_DATA" ]; then
        NEED_FILTER=true
        echo "原始数据比过滤数据新，需要重新过滤..."
    fi
    
    if [ "$NEED_FILTER" = true ]; then
        echo "正在过滤数据..."
        echo "Attack types: $ATTACK_TYPES"
        
        # 使用 conda 环境运行过滤脚本
        python "$PROJECT_ROOT/scripts/filter_attack_types.py" \
            --attack_types "$ATTACK_TYPES" \
            --output_suffix "$DATA_SUFFIX"
        
        if [ $? -ne 0 ]; then
            echo "错误: 数据过滤失败!"
            exit 1
        fi
    else
        echo "使用已存在的过滤数据: $FILTERED_TRAIN_DATA"
    fi
    
    TRAIN_DATA="$FILTERED_TRAIN_DATA"
    VAL_DATA="$FILTERED_VAL_DATA"
fi

# ============================================================================
# 实验名称和路径配置
# ============================================================================

# 保存根路径
SAVE_ROOT_DIR="${SAVE_ROOT_DIR:-./outputs/grpo}"

# 根据攻击类型生成实验名称
if [ -z "$DATA_SUFFIX" ]; then
    EXPERIMENT_NAME="attack_or_grpo_qwen3.5-9b-full"
else
    # 提取攻击类型数量
    ATTACK_COUNT=$(echo $ATTACK_TYPES | tr ',' '\n' | wc -l)
    EXPERIMENT_NAME="attack_or_grpo_qwen3.5-9b-ablation${DATA_SUFFIX}"
fi

# 自动生成完整路径
CHECKPOINT_DIR="$SAVE_ROOT_DIR/$EXPERIMENT_NAME/checkpoints"
ROLLOUT_DATA_DIR="$SAVE_ROOT_DIR/$EXPERIMENT_NAME/rollout_data"
VALIDATION_DATA_DIR="$SAVE_ROOT_DIR/$EXPERIMENT_NAME/validation_data"
LOG_DIR="$SAVE_ROOT_DIR/$EXPERIMENT_NAME/logs"

# 创建目录
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$ROLLOUT_DATA_DIR"
mkdir -p "$VALIDATION_DATA_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "消融实验配置:"
echo "  Attack types: $ATTACK_TYPES"
echo "  Experiment name: $EXPERIMENT_NAME"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "=========================================="

# ============================================================================
# 本地 LLM 服务配置验证
# ============================================================================

if [ -z "$LOCAL_LLM_BASE_URL" ]; then
    LOCAL_LLM_BASE_URL=""
fi
export LOCAL_LLM_BASE_URL

export LOCAL_LLM_MODEL_PATH_ENV=${LOCAL_LLM_MODEL_PATH}

if [ -z "$LOCAL_LLM_BASE_URL" ] || [ "$LOCAL_LLM_BASE_URL" = "" ]; then
    echo "Setting default LOCAL_LLM_BASE_URL: http://localhost:8000/v1"
    export LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
fi

export LOCAL_LLM_BASE_URL
export LOCAL_LLM_MODEL_PATH_ENV="${LOCAL_LLM_MODEL_PATH}"

echo "=========================================="
echo "Local LLM Service Configuration:"
echo "  LOCAL_LLM_BASE_URL: $LOCAL_LLM_BASE_URL"
echo "  LOCAL_LLM_MODEL_PATH_ENV: $LOCAL_LLM_MODEL_PATH_ENV"
echo "=========================================="

# 检查 vLLM 服务
if [[ "$LOCAL_LLM_BASE_URL" == *"localhost"* ]] || [[ "$LOCAL_LLM_BASE_URL" == *"127.0.0.1"* ]]; then
    if ! curl -s "${LOCAL_LLM_BASE_URL%/v1}/health" > /dev/null 2>&1; then
        echo "Warning: Cannot reach vLLM service at $LOCAL_LLM_BASE_URL"
        echo "Please ensure vLLM service is running"
    else
        echo "vLLM service is reachable"
    fi
fi

# ============================================================================
# GPU 配置
# ============================================================================

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

# 禁用 FlashInfer 和 V1 引擎
export FLASHINFER_DISABLE_JIT=0
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_MODEL_CLS_OVERRIDES="Qwen3_5ForCausalLM=vllm.model_executor.models.qwen3_5.Qwen3_5ForCausalLM"

# ============================================================================
# 训练配置
# ============================================================================

# Reward function path
REWARD_FN_PATH="$PROJECT_ROOT/grpo/rewards/attack_or.py"

# Config path
CONFIG_PATH="$PROJECT_ROOT/grpo/config"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "=========================================="
echo "Training logs will be saved to: $LOG_FILE"
echo "TensorBoard logs will be saved to: $CHECKPOINT_DIR/tensorboard"
echo "=========================================="

# 启动训练
VLLM_USE_V1=0 FLASHINFER_DISABLE_JIT=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_MODEL_CLS_OVERRIDES="Qwen3_5ForCausalLM=vllm.model_executor.models.qwen3_5.Qwen3_5ForCausalLM" python3 -m verl.trainer.main_ppo \
    --config-name=attack_or_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    data.max_prompt_length=2500 \
    data.max_response_length=4000 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.005 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path=$REWARD_FN_PATH \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.local_llm_base_url="${LOCAL_LLM_BASE_URL}" \
    +reward.custom_reward_function.reward_kwargs.local_llm_model_path="${LOCAL_LLM_MODEL_PATH}" \
    +reward.custom_reward_function.reward_kwargs.actor_model_path="${ACTOR_MODEL_PATH}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_attack_or_ablation' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.default_hdfs_dir="${HDFS_CHECKPOINT_DIR:-null}" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@ 2>&1 | tee "$LOG_FILE"


