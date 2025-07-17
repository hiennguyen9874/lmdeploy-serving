I am using lmdeploy to run inference with the InternVL3 model. The application only performs inference on a single image and prompt, and the output will be in the format:
{ "accident_detected": false or true }.
Each request involves only a single question and response, with no need to store history or cache past interactions.
Please optimize the configuration and the script below so that lmdeploy maximizes memory efficiency and speed, while maintaining a good balance with accuracy.

Below is the deployment script for lmdeploy.

```bash
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
export BACKEND=${BACKEND:-"turbomind"}
export PROXY_URL=${PROXY_URL:-"http://0.0.0.0:8000"}
export PORT=${PORT:-23333}
export MODEL_NAME=${MODEL_NAME:-"OpenGVLab/InternVL3-8B-AWQ"}
export TP=${TP:-1}
export SESSION_LEN=${SESSION_LEN:-16384}
export CACHE_MAX_ENTRY_COUNT=${CACHE_MAX_ENTRY_COUNT:-0.2}
export MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-64}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
export MAX_PREFILL_TOKEN_NUM=${MAX_PREFILL_TOKEN_NUM:-8192}
export CHAT_TEMPLATE=${CHAT_TEMPLATE:-"internvl3_chat_template.json"}
export DTYPE=${DTYPE:-"float16"}
export CACHE_BLOCK_SEQ_LEN=${CACHE_BLOCK_SEQ_LEN:-8192}
export QUANT_POLICY=${QUANT_POLICY:-0}
export VISION_MAX_BATCH_SIZE=${VISION_MAX_BATCH_SIZE:-8}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"} # INFO, DEBUG, WARNING, ERROR, CRITICAL

# Function to run the server
run_server() {
    lmdeploy serve api_server ${MODEL_NAME} \
        --proxy-url ${PROXY_URL} \
        --server-port ${PORT} \
        --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
        --eager-mode \
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
        --tp ${TP}
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
```

Below are the vLLM Server Arguments.

```
Serve LLMs with restful api using fastapi.

positional arguments:
  model_path            The path of a model. it could be one of the following options: - i) a local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download from ii) and iii). - ii) the model_id of a lmdeploy-quantized model
                        hosted inside a model repo on huggingface.co, such as "internlm/internlm-chat-20b-4bit", "lmdeploy/llama2-chat-70b-4bit",
                        etc. - iii) the model_id of a model hosted inside a model repo on huggingface.co, such as "internlm/internlm-chat-7b",
                        "qwen/qwen-7b-chat ", "baichuan-inc/baichuan2-7b-chat" and so on. Type: str

optional arguments:
  -h, --help            show this help message and exit
  --server-name SERVER_NAME
                        Host ip for serving. Default: 0.0.0.0. Type: str
  --server-port SERVER_PORT
                        Server port. Default: 23333. Type: int
  --allow-origins ALLOW_ORIGINS [ALLOW_ORIGINS ...]
                        A list of allowed origins for cors. Default: ['*']. Type: str
  --allow-credentials   Whether to allow credentials for cors. Default: False
  --allow-methods ALLOW_METHODS [ALLOW_METHODS ...]
                        A list of allowed http methods for cors. Default: ['*']. Type: str
  --allow-headers ALLOW_HEADERS [ALLOW_HEADERS ...]
                        A list of allowed http headers for cors. Default: ['*']. Type: str
  --proxy-url PROXY_URL
                        The proxy url for api server.. Default: None. Type: str
  --max-concurrent-requests MAX_CONCURRENT_REQUESTS
                        This refers to the number of concurrent requests that the server can handle. The server is designed to process the engine’s
                        tasks once the maximum number of concurrent requests is reached, regardless of any additional requests sent by clients
                        concurrently during that time. Default to None.. Type: int
  --backend {pytorch,turbomind}
                        Set the inference backend. Default: turbomind. Type: str
  --log-level {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
                        Set the log level. Default: ERROR. Type: str
  --api-keys [API_KEYS ...]
                        Optional list of space separated API keys. Default: None. Type: str
  --ssl                 Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'. Default: False
  --model-name MODEL_NAME
                        The name of the served model. It can be accessed by the RESTful API `/v1/models`. If it is not specified, `model_path` will
                        be adopted. Default: None. Type: str
  --max-log-len MAX_LOG_LEN
                        Max number of prompt characters or prompt tokens beingprinted in log. Default: Unlimited. Type: int
  --disable-fastapi-docs
                        Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint. Default: False
  --chat-template CHAT_TEMPLATE
                        A JSON file or string that specifies the chat template configuration. Please refer to
                        https://lmdeploy.readthedocs.io/en/latest/advance/chat_template.html for the specification. Default: None. Type: str
  --tool-call-parser TOOL_CALL_PARSER
                        The registered tool parser name dict_keys(['internlm', 'llama3', 'qwen']). Default to None.. Type: str
  --reasoning-parser REASONING_PARSER
                        The registered reasoning parser name from dict_keys(['deepseek-r1', 'qwen-qwq']). Default to None.. Type: str
  --revision REVISION   The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the
                        default version.. Type: str
  --download-dir DOWNLOAD_DIR
                        Directory to download and load the weights, default to the default cache directory of huggingface.. Type: str

PyTorch engine arguments:
  --adapters [ADAPTERS ...]
                        Used to set path(s) of lora adapter(s). One can input key-value pairs in xxx=yyy format for multiple lora adapters. If only
                        have one adapter, one can only input the path of the adapter.. Default: None. Type: str
  --device {cuda,ascend,maca,camb}
                        The device type of running. Default: cuda. Type: str
  --eager-mode          Whether to enable eager mode. If True, cuda graph would be disabled. Default: False
  --dtype {auto,float16,bfloat16}
                        data type for model weights and activations. The "auto" option will use FP16 precision for FP32 and FP16 models, and BF16
                        precision for BF16 models. This option will be ignored if the model is a quantized model. Default: auto. Type: str
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --session-len SESSION_LEN
                        The max session length of a sequence. Default: None. Type: int
  --max-batch-size MAX_BATCH_SIZE
                        Maximum batch size. If not specified, the engine will automatically set it according to the device. Default: None. Type:
                        int
  --cache-max-entry-count CACHE_MAX_ENTRY_COUNT
                        The percentage of free gpu memory occupied by the k/v cache, excluding weights . Default: 0.8. Type: float
  --cache-block-seq-len CACHE_BLOCK_SEQ_LEN
                        The length of the token sequence in a k/v block. For Turbomind Engine, if the GPU compute capability is >= 8.0, it should
                        be a multiple of 32, otherwise it should be a multiple of 64. For Pytorch Engine, if Lora Adapter is specified, this
                        parameter will be ignored. Default: 64. Type: int
  --enable-prefix-caching
                        Enable cache and match prefix. Default: False
  --max-prefill-token-num MAX_PREFILL_TOKEN_NUM
                        the max number of tokens per iteration during prefill. Default: 8192. Type: int
  --quant-policy {0,4,8}
                        Quantize kv or not. 0: no quant; 4: 4bit kv; 8: 8bit kv. Default: 0. Type: int

TurboMind engine arguments:
  --dtype {auto,float16,bfloat16}
                        data type for model weights and activations. The "auto" option will use FP16 precision for FP32 and FP16 models, and BF16
                        precision for BF16 models. This option will be ignored if the model is a quantized model. Default: auto. Type: str
  --tp TP               GPU number used in tensor parallelism. Should be 2^n. Default: 1. Type: int
  --session-len SESSION_LEN
                        The max session length of a sequence. Default: None. Type: int
  --max-batch-size MAX_BATCH_SIZE
                        Maximum batch size. If not specified, the engine will automatically set it according to the device. Default: None. Type:
                        int
  --cache-max-entry-count CACHE_MAX_ENTRY_COUNT
                        The percentage of free gpu memory occupied by the k/v cache, excluding weights . Default: 0.8. Type: float
  --cache-block-seq-len CACHE_BLOCK_SEQ_LEN
                        The length of the token sequence in a k/v block. For Turbomind Engine, if the GPU compute capability is >= 8.0, it should
                        be a multiple of 32, otherwise it should be a multiple of 64. For Pytorch Engine, if Lora Adapter is specified, this
                        parameter will be ignored. Default: 64. Type: int
  --enable-prefix-caching
                        Enable cache and match prefix. Default: False
  --max-prefill-token-num MAX_PREFILL_TOKEN_NUM
                        the max number of tokens per iteration during prefill. Default: 8192. Type: int
  --quant-policy {0,4,8}
                        Quantize kv or not. 0: no quant; 4: 4bit kv; 8: 8bit kv. Default: 0. Type: int
  --model-format {hf,awq,gptq}
                        The format of input model. `hf` means `hf_llama`, `awq` represents the quantized model by AWQ, and `gptq` refers to the
                        quantized model by GPTQ. Default: None. Type: str
  --rope-scaling-factor ROPE_SCALING_FACTOR
                        Rope scaling factor. Default: 0.0. Type: float
  --num-tokens-per-iter NUM_TOKENS_PER_ITER
                        the number of tokens processed in a forward pass. Default: 0. Type: int
  --max-prefill-iters MAX_PREFILL_ITERS
                        the max number of forward passes in prefill stage. Default: 1. Type: int
  --communicator {nccl,native}
                        Communication backend for multi-GPU inference. Default: nccl. Type: str

Vision model arguments:
  --vision-max-batch-size VISION_MAX_BATCH_SIZE
                        the vision model batch size. Default: 1. Type: int
```
