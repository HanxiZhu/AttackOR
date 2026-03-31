#!/bin/bash
# GRPO training script for Attack_OR dataset
# Usage: bash run_attack_or.sh [ENGINE] [ACTOR_MODEL_PATH] [LOCAL_LLM_MODEL_PATH]
# Example: bash run_attack_or.sh vllm /path/to/orlm /path/to/llama3-7b




set -x

# Get engine from first argument, default to vllm
ENGINE=${1:-vllm}

# ============================================================================
# 模型路径配置 - 用户需要根据实际情况修改以下路径
# ============================================================================

# Actor 模型路径（orlm，基于 llama3-7b 的运筹学模型）
# 用户需要设置此路径指向训练好的 orlm 模型
ACTOR_MODEL_PATH=${2:-"./outputs/sft"}

# 本地 LLM 模型路径（llama3-7b，用于防御逻辑评估）
# 用户需要设置此路径指向本地的 llama3-7b 模型
LOCAL_LLM_MODEL_PATH=${3:-"meta-llama/Llama-3.1-8B-Instruct"}

# ============================================================================
# 本地 LLM 服务配置（可选）
# ============================================================================

# 本地 LLM 服务地址（vllm OpenAI 兼容 API）
# 如果设置了此环境变量，奖励函数将使用指定的服务地址
# 如果不设置，奖励函数将尝试从 Ray 获取 rollout 服务地址
# 格式示例: http://localhost:8000/v1 或 http://192.168.1.100:8000/v1
# 注意：如果设置为空字符串，将使用默认行为（从 Ray 获取）
if [ -z "$LOCAL_LLM_BASE_URL" ]; then
    # 如果未设置，使用空字符串，让奖励函数自动从 Ray 获取
    LOCAL_LLM_BASE_URL=""
fi
export LOCAL_LLM_BASE_URL

# 本地 LLM 模型路径环境变量（传递给奖励函数）
export LOCAL_LLM_MODEL_PATH_ENV=${LOCAL_LLM_MODEL_PATH}

# ============================================================================
# Checkpoint 和日志路径配置
# ============================================================================

# 保存根路径（所有checkpoint和日志的根目录）
# 请修改为你的实际保存路径，可以是绝对路径或相对路径
SAVE_ROOT_DIR="${SAVE_ROOT_DIR:-./outputs/grpo}"

# 实验名称（用于创建子目录）
EXPERIMENT_NAME="attack_or_grpo_qwen3.5-9b-merged-del-test"

# 自动生成完整路径（根路径/实验名）
CHECKPOINT_DIR="$SAVE_ROOT_DIR/$EXPERIMENT_NAME/checkpoints"
ROLLOUT_DATA_DIR="$SAVE_ROOT_DIR/$EXPERIMENT_NAME/rollout_data"
VALIDATION_DATA_DIR="$SAVE_ROOT_DIR/$EXPERIMENT_NAME/validation_data"
LOG_DIR="$SAVE_ROOT_DIR/$EXPERIMENT_NAME/logs"

# HDFS checkpoint 路径（可选，如果使用 HDFS）
# 如果不需要，留空或设置为 null
HDFS_CHECKPOINT_DIR="${HDFS_CHECKPOINT_DIR:-}"

# 创建目录（如果不存在）
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$ROLLOUT_DATA_DIR"
mkdir -p "$VALIDATION_DATA_DIR"
mkdir -p "$LOG_DIR"

echo "Save paths configured:"
echo "  Root dir: $SAVE_ROOT_DIR"
echo "  Experiment name: $EXPERIMENT_NAME"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Rollout data dir: $ROLLOUT_DATA_DIR"
echo "  Validation data dir: $VALIDATION_DATA_DIR"
echo "  Log dir: $LOG_DIR"

# ============================================================================
# 项目路径配置
# ============================================================================

# Get project root directory (script is in grpo/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Data paths relative to project root
TRAIN_DATA="$PROJECT_ROOT/data/merged_train.parquet"
VAL_DATA="$PROJECT_ROOT/data/merged_val.parquet"

# Reward function path
REWARD_FN_PATH="$PROJECT_ROOT/grpo/rewards/attack_or.py"

# Config path
CONFIG_PATH="$PROJECT_ROOT/grpo/config"



# 如果 vllm 使用 GPU 0，训练使用 GPU 1-7
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7


# ============================================================================
# 本地 LLM 服务配置验证
# ============================================================================

# 确保 LOCAL_LLM_BASE_URL 已设置
if [ -z "$LOCAL_LLM_BASE_URL" ] || [ "$LOCAL_LLM_BASE_URL" = "" ]; then
    echo "Setting default LOCAL_LLM_BASE_URL: http://localhost:8000/v1"
    export LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
fi

# 导出环境变量（确保 Ray worker 可以访问）
export LOCAL_LLM_BASE_URL
export LOCAL_LLM_MODEL_PATH_ENV="${LOCAL_LLM_MODEL_PATH}"

echo "=========================================="
echo "Local LLM Service Configuration:"
echo "  LOCAL_LLM_BASE_URL: $LOCAL_LLM_BASE_URL"
echo "  LOCAL_LLM_MODEL_PATH_ENV: $LOCAL_LLM_MODEL_PATH_ENV"
echo "=========================================="

# 检查 vLLM 服务（可选，如果服务不在同一机器上可以跳过）
if [[ "$LOCAL_LLM_BASE_URL" == *"localhost"* ]] || [[ "$LOCAL_LLM_BASE_URL" == *"127.0.0.1"* ]]; then
    if ! curl -s "${LOCAL_LLM_BASE_URL%/v1}/health" > /dev/null 2>&1; then
        echo "Warning: Cannot reach vLLM service at $LOCAL_LLM_BASE_URL"
        echo "Please ensure vLLM service is running"
    else
        echo "vLLM service is reachable"
    fi
fi


# 1. 禁用 FlashInfer 和 V1 引擎
# 修复：确保环境变量在 Python 进程启动前设置，防止 FlashInfer JIT 编译错误
export FLASHINFER_DISABLE_JIT=0 # 恢复JIT，我们让它尝试编译
export VLLM_USE_V1=0            # 彻底禁用V1引擎（v1引擎强依赖FlashInfer）
export VLLM_ATTENTION_BACKEND=FLASH_ATTN # 强制 vLLM 使用标准 FlashAttention 而不是 FlashInfer
export VLLM_MODEL_CLS_OVERRIDES="Qwen3_5ForCausalLM=vllm.model_executor.models.qwen3_5.Qwen3_5ForCausalLM"

# 验证环境变量已设置
if [ "${VLLM_USE_V1}" != "0" ]; then
    echo "Warning: VLLM_USE_V1 is not 0, forcing to 0"
    export VLLM_USE_V1=0
fi

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "=========================================="
echo "Training logs will be saved to: $LOG_FILE"
echo "TensorBoard logs will be saved to: $CHECKPOINT_DIR/tensorboard"
echo "=========================================="

# 确保环境变量传递给 Python 进程，并使用 tee 保存日志
VLLM_USE_V1=0 FLASHINFER_DISABLE_JIT=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_MODEL_CLS_OVERRIDES="Qwen3_5ForCausalLM=vllm.model_executor.models.qwen3_5.Qwen3_5ForCausalLM" python3 -m verl.trainer.main_ppo \
    --config-name=attack_or_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    data.max_prompt_length=3000 \
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
    trainer.project_name='verl_grpo_attack_or' \
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
