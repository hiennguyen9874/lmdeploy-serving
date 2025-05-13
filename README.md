# LMDeploy Serving

This repository contains scripts for deploying and serving large language models using LMDeploy. It includes support for various models including Qwen3 and InternVL3 in different quantized formats.

## Installation

### Prerequisites

- CUDA-compatible GPU
- Linux environment
- Python 3.9 (required by dependencies)

### Install UV (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
# Install setuptools first
uv pip install setuptools

# Skip flash-attn installation as it will be built from source
uv sync --no-install-package flash-attn

# Install all other dependencies
uv sync
```

## Server Architecture

The server architecture consists of:

1. **Proxy Server**: Acts as a central coordinator for distributing requests to individual model servers
2. **Model Servers**: Serve specific models (Qwen3, InternVL3, etc.) with optimized configurations

## Starting the Servers

### 1. Start the Proxy Server

```bash
./start_proxy.sh
```

Environment variables:
- `SERVER_NAME`: Host IP (default: 0.0.0.0)
- `SERVER_PORT`: Port number (default: 8000)

### 2. Start Model Servers

#### InternVL3 14B (AWQ)

```bash
./start_server-internvl3.sh
```

#### Qwen3 14B (AWQ)

```bash
./start_server-qwen3_14b.sh
```

#### Qwen3 30B (A3B-GPTQ)

```bash
./start_server-qwen3_30b_a3b.sh
```

#### Qwen3 32B (AWQ)

```bash
./start_server-qwen3_32b.sh
```

### Common Environment Variables

Each model server script supports the following environment variables:

- `CUDA_VISIBLE_DEVICES`: Specify which GPU(s) to use
- `PROXY_URL`: URL of the proxy server (default: http://0.0.0.0:8000)
- `PORT`: Port for the model server (default: 23333)
- `MODEL_NAME`: HuggingFace model path
- `TP`: Tensor parallelism (default: 1)
- `SESSION_LEN`: Maximum sequence length (default: 4096)
- `CACHE_MAX_ENTRY_COUNT`: Percentage of GPU memory for KV cache (default: 0.2)
- `MAX_CONCURRENT_REQUESTS`: Maximum concurrent requests (default: 4-10 depending on model)
- `MAX_BATCH_SIZE`: Maximum batch size (default: 4-10 depending on model)
- `MAX_PREFILL_TOKEN_NUM`: Maximum tokens for prefill (default: 8192)
- `CHAT_TEMPLATE`: Chat template JSON file

## Supported Models

| Model | Type | Format | Default Script |
|-------|------|--------|---------------|
| OpenGVLab/InternVL3-14B-AWQ | Multimodal | AWQ | start_server-internvl3.sh |
| Qwen/Qwen3-14B-AWQ | LLM | AWQ | start_server-qwen3_14b.sh |
| Qwen/Qwen3-30B-A3B-GPTQ-Int4 | LLM | GPTQ | start_server-qwen3_30b_a3b.sh |
| Qwen/Qwen3-32B-AWQ | LLM | AWQ | start_server-qwen3_32b.sh |

## Advanced Configuration

For detailed configuration options and parameters, please refer to the [LMDeploy documentation](https://lmdeploy.readthedocs.io/). Available parameters can be found in the SERVER.md file included in this repository.

## Troubleshooting

The server scripts include an automatic restart mechanism that will restart the server after a crash. If you experience issues:

1. Check the CUDA_VISIBLE_DEVICES settings to ensure proper GPU allocation
2. Verify that the model format matches the specified format in the script
3. Adjust MAX_BATCH_SIZE and MAX_CONCURRENT_REQUESTS based on your GPU memory
4. Ensure the proxy server is running before starting model servers