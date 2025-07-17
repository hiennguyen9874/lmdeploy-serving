#!/bin/bash

# Enable debug mode to print each command before execution
set -x
# Exit immediately if a command exits with a non-zero status
set -e

# Activate Python virtual environment
source .venv/bin/activate

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Enable parallel processing for tokenizers to improve performance
export TOKENIZERS_PARALLELISM="true"
export BACKEND=${BACKEND:-"turbomind"}
export PROXY_URL=${PROXY_URL:-"http://0.0.0.0:8000"}
export PORT=${PORT:-23333}
export MODEL_NAME=${MODEL_NAME:-"OpenGVLab/InternVL3-8B-AWQ"}
export TP=${TP:-1}

# OPTIMIZED PARAMETERS FOR SINGLE-IMAGE INFERENCE
# Reduced session length for single Q&A interactions
export SESSION_LEN=${SESSION_LEN:-4096}

# Increased cache allocation for better performance with vision models
export CACHE_MAX_ENTRY_COUNT=${CACHE_MAX_ENTRY_COUNT:-0.1}

# Reduced concurrent requests for memory efficiency
export MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-16}

# Smaller batch size optimized for single-image processing
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-4}

# Reduced prefill tokens for shorter prompts
export MAX_PREFILL_TOKEN_NUM=${MAX_PREFILL_TOKEN_NUM:-2048}

export CHAT_TEMPLATE=${CHAT_TEMPLATE:-"internvl3_chat_template.json"}
export DTYPE=${DTYPE:-"float16"}

# Optimized cache block size for better memory alignment
export CACHE_BLOCK_SEQ_LEN=${CACHE_BLOCK_SEQ_LEN:-64}

# Enable KV cache quantization for memory efficiency
export QUANT_POLICY=${QUANT_POLICY:-8}

# Optimized vision batch size for single-image inference
export VISION_MAX_BATCH_SIZE=${VISION_MAX_BATCH_SIZE:-4}

export LOG_LEVEL=${LOG_LEVEL:-"WARNING"} # Reduced logging for performance

# Additional optimizations for TurboMind engine
export MAX_PREFILL_ITERS=${MAX_PREFILL_ITERS:-1}
export NUM_TOKENS_PER_ITER=${NUM_TOKENS_PER_ITER:-512}

# Function to run the server
run_server() {
    lmdeploy serve api_server ${MODEL_NAME} \
        --proxy-url ${PROXY_URL} \
        --server-port ${PORT} \
        --model-format awq \
        --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
        --max-batch-size ${MAX_BATCH_SIZE} \
        --max-prefill-token-num ${MAX_PREFILL_TOKEN_NUM} \
        --chat-template ${CHAT_TEMPLATE} \
        --dtype ${DTYPE} \
        --cache-block-seq-len ${CACHE_BLOCK_SEQ_LEN} \
        --backend ${BACKEND} \
        --max-concurrent-requests ${MAX_CONCURRENT_REQUESTS} \
        --session-len ${SESSION_LEN} \
        --quant-policy ${QUANT_POLICY} \
        --vision-max-batch-size ${VISION_MAX_BATCH_SIZE} \
        --log-level ${LOG_LEVEL} \
        --tp ${TP} \
        --max-prefill-iters ${MAX_PREFILL_ITERS} \
        --num-tokens-per-iter ${NUM_TOKENS_PER_ITER} \
        --enable-prefix-caching
}

# Infinite retry loop
while true; do
    echo "Starting server..."
    if run_server; then
        echo "Server exited successfully"
        break
    else
        echo "Server crashed, restarting in 5 seconds..."
        sleep 5
    fi
done
