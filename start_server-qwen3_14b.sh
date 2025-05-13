#!/bin/bash

# Enable debug mode to print each command before execution
set -x
# Exit immediately if a command exits with a non-zero status
set -e

# Activate Python virtual environment
source .venv/bin/activate

# Specify which GPUs to use (GPUs 3 and 5 in this case)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2"}

# Enable parallel processing for tokenizers to improve performance
export TOKENIZERS_PARALLELISM="true"

export PROXY_URL=${PROXY_URL:-"http://0.0.0.0:8000"}
export PORT=${PORT:-23333}
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-14B-AWQ"}
export TP=${TP:-1}
export SESSION_LEN=${SESSION_LEN:-4096}
export CACHE_MAX_ENTRY_COUNT=${CACHE_MAX_ENTRY_COUNT:-0.2}
export MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-10}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-10}
export MAX_PREFILL_TOKEN_NUM=${MAX_PREFILL_TOKEN_NUM:-8192}
export CHAT_TEMPLATE=${CHAT_TEMPLATE:-"qwen3_chat_template.json"}

# Function to run the server
run_server() {
    lmdeploy serve api_server ${MODEL_NAME} \
        --proxy-url ${PROXY_URL} \
        --server-port ${PORT} \
        --backend turbomind \
        --model-format awq \
        --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
        --eager-mode \
        --max-batch-size ${MAX_BATCH_SIZE} \
        --max-prefill-token-num ${MAX_PREFILL_TOKEN_NUM} \
        --chat-template ${CHAT_TEMPLATE} \
        --max-concurrent-requests ${MAX_CONCURRENT_REQUESTS}
        --dtype ${DTYPE}
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