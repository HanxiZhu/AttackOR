#!/bin/bash
# ORLM 3D Hyperparameter Search Script
# 严格对齐论文中的 (LR, Batch Size, Epochs) 网格搜索空间

set -e  # 遇到错误立即退出

export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${RANDOM}"

# ==========================================
# 基础配置
# ==========================================
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-9B}"
DATA_PATH="${DATA_PATH:-./datasets_local/train_data.jsonl}"
BASE_SAVE_PATH="${BASE_SAVE_PATH:-./outputs/sft_search/ORLM-qwen3.5-9b}"

NUM_GPUS=8
BATCH_SIZE_PER_GPU=4  # 保持每张卡的物理 Batch Size 为 2 避免 OOM
PREPROCESSING_NUM_WORKERS=8
MAX_SEQ_LENGTH=14096
WARMUP_RATIO=0.03
DEEPSPEED_CONFIG="sft/configs/stage3_no_offloading_bf16.json"

USE_LORA=false
LORA_RANK=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ==========================================
# 论文标准的超参数搜索网格
# ==========================================
LEARNING_RATES=(1e-5 5e-5 5e-6)
BATCH_SIZES=(64 128 256 512)
EPOCHS_LIST=(2 3)

TOTAL_RUNS=$((${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#EPOCHS_LIST[@]}))

echo "=========================================="
echo "准备启动 ORLM 3D 超参数网格搜索..."
echo "总计组合数: $TOTAL_RUNS"
echo "=========================================="

mkdir -p logs

# 1. 外层循环：遍历 Total Batch Size
for BS in "${BATCH_SIZES[@]}"; do
    
    # 动态计算当前 Batch Size 下的梯度累积步数
    # 例如: BS=64, GPUS=8, PER_GPU=2 -> ACC=4
    # 例如: BS=512, GPUS=8, PER_GPU=2 -> ACC=32
    GRADIENT_ACC_STEPS=$(($BS / $NUM_GPUS / $BATCH_SIZE_PER_GPU))

    # 2. 中层循环：遍历学习率
    for LR in "${LEARNING_RATES[@]}"; do
        
        # 3. 内层循环：遍历 Epoch
        for EPOCHS in "${EPOCHS_LIST[@]}"; do
            
            # 为当前组合生成专属的保存路径和日志
            SAVE_PATH="${BASE_SAVE_PATH}_bs${BS}_lr${LR}_ep${EPOCHS}"
            LOG_FILE="logs/train_bs${BS}_lr${LR}_ep${EPOCHS}.log"

            echo "------------------------------------------"
            echo "[当前任务] Batch Size: $BS (Acc: $GRADIENT_ACC_STEPS) | LR: $LR | Epochs: $EPOCHS"
            echo "[输出目录] $SAVE_PATH"
            echo "------------------------------------------"

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
                --learning_rate $LR \
                --lr_scheduler_type linear \
                --warmup_ratio $WARMUP_RATIO \
                --num_train_epochs $EPOCHS \
                --logging_steps 1 \
                --report_to \"tensorboard\" \
                --gradient_checkpointing True \
                --deepspeed $DEEPSPEED_CONFIG \
                --bf16 True"

            if [ "$USE_LORA" = true ]; then
                TRAIN_CMD="$TRAIN_CMD \
                --use_lora True \
                --lora_rank $LORA_RANK \
                --lora_alpha $LORA_ALPHA \
                --lora_dropout $LORA_DROPOUT \
                --lora_target_modules $LORA_TARGET_MODULES"
            fi

            echo "执行中，实时日志详见: $LOG_FILE"
            eval $TRAIN_CMD 2>&1 | tee "$LOG_FILE"
            
            echo ""
            echo "[任务完成] 已保存至: $SAVE_PATH"
            
            # 强制清理缓存并休眠，防止 72 连跑引发的内存/显存碎片化 OOM
            sleep 15
            
        done
    done
done

echo "🎉 72 组超参数搜索任务全部结束！"