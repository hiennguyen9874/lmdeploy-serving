#!/bin/bash

# Enable debug mode to print each command before execution
set -x
# Exit immediately if a command exits with a non-zero status
set -e

# Activate Python virtual environment
source .venv/bin/activate

# Specify which GPUs to use (GPUs 3 and 5 in this case)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Enable parallel processing for tokenizers to improve performance
export TOKENIZERS_PARALLELISM="true"

export PROXY_URL=${PROXY_URL:-"http://0.0.0.0:8000"}
export PORT=${PORT:-23333}
export MODEL_NAME=${MODEL_NAME:-"OpenGVLab/InternVL3-8B-AWQ"}
export TP=${TP:-1}
export SESSION_LEN=${SESSION_LEN:-8192}
export CACHE_MAX_ENTRY_COUNT=${CACHE_MAX_ENTRY_COUNT:-0.2}
export MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-64}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
export MAX_PREFILL_TOKEN_NUM=${MAX_PREFILL_TOKEN_NUM:-4096}
export CHAT_TEMPLATE=${CHAT_TEMPLATE:-"internvl3_chat_template.json"}
export DTYPE=${DTYPE:-"float16"}
export CACHE_BLOCK_SEQ_LEN=${CACHE_BLOCK_SEQ_LEN:-4096}
export QUANT_POLICY=${QUANT_POLICY:-0}
export VISION_MAX_BATCH_SIZE=${VISION_MAX_BATCH_SIZE:-8}

# Function to run the server
run_server() {
    lmdeploy serve api_server ${MODEL_NAME} \
        --proxy-url ${PROXY_URL} \
        --server-port ${PORT} \
        --model-format awq \
        --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
        --eager-mode \
        --max-batch-size ${MAX_BATCH_SIZE} \
        --max-prefill-token-num ${MAX_PREFILL_TOKEN_NUM} \
        --chat-template ${CHAT_TEMPLATE} \
        --dtype ${DTYPE} \
        --cache-block-seq-len ${CACHE_BLOCK_SEQ_LEN} \
        --backend turbomind \
        --max-concurrent-requests ${MAX_CONCURRENT_REQUESTS} \
        --session-len ${SESSION_LEN} \
        --quant-policy ${QUANT_POLICY} \
        --vision-max-batch-size ${VISION_MAX_BATCH_SIZE}
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
