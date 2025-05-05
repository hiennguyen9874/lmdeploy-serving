#!/bin/bash

# Enable debug mode to print each command before execution
set -x
# Exit immediately if a command exits with a non-zero status
set -e

# Activate Python virtual environment
source .venv/bin/activate

export SERVER_NAME=${SERVER_NAME:-0.0.0.0}
export SERVER_PORT=${SERVER_PORT:-8000}

lmdeploy serve proxy --server-name ${SERVER_NAME} --server-port ${SERVER_PORT} --strategy "min_expected_latency"
