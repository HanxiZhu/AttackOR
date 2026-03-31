#!/bin/bash
# 启动独立的 llama3-7b vllm 服务用于防御逻辑评估

set -x

# ============================================================================
# 配置参数 - 请根据实际情况修改以下路径和参数
# ============================================================================

# llama3-7b 模型路径（请修改为实际路径）
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

# 服务端口
PORT=8000

# vllm 服务参数
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=9600

# ============================================================================
# 检查模型路径
# ============================================================================

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please modify MODEL_PATH in the script to point to your llama3-7b model"
    exit 1
fi

echo "Starting vLLM service..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Service will be available at: http://localhost:$PORT/v1"

# 启动 vllm 服务（OpenAI 兼容 API）
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code