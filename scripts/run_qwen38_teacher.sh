#!/usr/bin/env bash
set -Eeuo pipefail

export MODEL_DIR="${MODEL_DIR:-/ssd/models/Qwen3.8-27B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38-teacher}"
export TP_SIZE="${TP_SIZE:-4}"
export SERVER_COUNT="${SERVER_COUNT:-2}"
export SERVER_CONTEXT_LENGTH="${SERVER_CONTEXT_LENGTH:-90112}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
export HARNESS_SUBCOMMAND="${HARNESS_SUBCOMMAND:-execute}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_ephemeral_sglang.sh" "$@"
