#!/usr/bin/env bash
set -Eeuo pipefail

export MODEL_DIR="${MODEL_DIR:-/ssd/models/Qwen3.5-9B-PCL-Qwen38-27B-108samples-1epoch}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b-pcl-qwen38-27b-108samples-1epoch}"
export TP_SIZE="${TP_SIZE:-8}"
export SERVER_COUNT="${SERVER_COUNT:-1}"
export SERVER_CONTEXT_LENGTH="${SERVER_CONTEXT_LENGTH:-131072}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
export HARNESS_SUBCOMMAND="${HARNESS_SUBCOMMAND:-run}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/reasoning/pcl_qwen35_9b_student_108samples_1epoch.yaml}"
exec "$SCRIPT_DIR/run_ephemeral_sglang.sh" "$CONFIG"
