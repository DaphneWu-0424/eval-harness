#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:0.5.19-qwen38-runtime}"
PROJECT_DIR="${PROJECT_DIR:-/ssd/eval-harness}"
MODEL_DIR="${MODEL_DIR:-/ssd/models/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-9B}"
DATASETS_DIR="${DATASETS_DIR:-/ssd/datasets}"
RESULTS_DIR="${RESULTS_DIR:-/ssd/eval-results}"
DEEPSEEK_ENV_FILE="${DEEPSEEK_ENV_FILE:-$PROJECT_DIR/.secrets/deepseek.env}"
CONFIG="${1:-configs/reasoning/pcl_qwen35_9b_base.yaml}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-1800}"
SERVER_CONTEXT_LENGTH="${SERVER_CONTEXT_LENGTH:-131072}"
TP_SIZE="${TP_SIZE:-8}"
SERVER_COUNT="${SERVER_COUNT:-1}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-31000}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
HARNESS_SUBCOMMAND="${HARNESS_SUBCOMMAND:-run}"
REASONING_PARSER="${REASONING_PARSER-qwen3}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
CONTAINER_NAME="eval-harness-${RUN_ID}"
IMAGE_ID=""

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

cleanup() {
  local original_status=$?
  local cleanup_status=0
  trap - EXIT INT TERM
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    local actual_run_id
    actual_run_id="$(docker inspect --format '{{ index .Config.Labels "eval-harness.run_id" }}' "$CONTAINER_NAME" 2>/dev/null || true)"
    if [[ "$actual_run_id" == "$RUN_ID" ]]; then
      log "removing task container $CONTAINER_NAME"
      docker rm -f "$CONTAINER_NAME" >/dev/null || cleanup_status=$?
    else
      log "refusing to remove $CONTAINER_NAME: run_id label mismatch"
      cleanup_status=1
    fi
  fi
  if (( cleanup_status != 0 )); then
    log "cleanup failed; inspect manually: docker inspect $CONTAINER_NAME"
    exit "$cleanup_status"
  fi
  exit "$original_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for path in "$PROJECT_DIR" "$MODEL_DIR" "$DATASETS_DIR"; do
  [[ -e "$path" ]] || { log "required path not found: $path"; exit 2; }
done
if [[ -f "$DEEPSEEK_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEEPSEEK_ENV_FILE"
  set +a
fi
# Judge credentials are evaluator-specific. The configured backend validates its own
# environment variables when llm_fallback is enabled; rule-only and future non-LLM
# evaluators must not require DeepSeek credentials.
mkdir -p "$RESULTS_DIR"
[[ -f "$PROJECT_DIR/$CONFIG" ]] || { log "config not found: $PROJECT_DIR/$CONFIG"; exit 2; }
docker image inspect "$IMAGE" >/dev/null
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
mapfile -t gpu_memory_used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
required_gpus=$((TP_SIZE * SERVER_COUNT))
if (( ${#gpu_memory_used[@]} < required_gpus )); then
  log "$required_gpus GPUs are required, found ${#gpu_memory_used[@]}"
  exit 2
fi
for ((index=0; index<required_gpus; index++)); do
  if (( gpu_memory_used[index] > 1024 )); then
    log "GPU $index is already using ${gpu_memory_used[index]} MiB; refusing to start TP8 container"
    exit 4
  fi
done

log "creating $CONTAINER_NAME from $IMAGE ($IMAGE_ID)"
docker run -d \
  --name "$CONTAINER_NAME" \
  --label eval-harness.managed=true \
  --label "eval-harness.run_id=$RUN_ID" \
  --gpus all \
  --ipc host \
  --network bridge \
  --mount "type=bind,src=$PROJECT_DIR,dst=/workspace/eval-harness,readonly" \
  --mount "type=bind,src=$MODEL_DIR,dst=/workspace/model,readonly" \
  --mount "type=bind,src=$DATASETS_DIR,dst=/workspace/datasets,readonly" \
  --mount "type=bind,src=$RESULTS_DIR,dst=/workspace/results" \
  --entrypoint bash \
  "$IMAGE" -lc 'sleep infinity' >/dev/null

log "starting $SERVER_COUNT SGLang server(s), TP=$TP_SIZE, context=$SERVER_CONTEXT_LENGTH"
REASONING_PARSER_ARG=""
if [[ -n "${REASONING_PARSER}" ]]; then
  REASONING_PARSER_ARG="--reasoning-parser ${REASONING_PARSER}"
fi
for ((server_index=0; server_index<SERVER_COUNT; server_index++)); do
  port=$((SERVER_PORT_BASE + server_index))
  base_gpu_id=$((server_index * TP_SIZE))
  docker exec -d "$CONTAINER_NAME" bash -lc \
    "sglang serve --model-path /workspace/model --served-model-name '$SERVED_MODEL_NAME' --tp-size '$TP_SIZE' --base-gpu-id '$base_gpu_id' --host 127.0.0.1 --port '$port' --dtype bfloat16 --trust-remote-code ${REASONING_PARSER_ARG} --context-length '$SERVER_CONTEXT_LENGTH' --chunked-prefill-size '$CHUNKED_PREFILL_SIZE' --mem-fraction-static '$MEM_FRACTION_STATIC' --max-running-requests '$MAX_RUNNING_REQUESTS' > /workspace/results/sglang-${RUN_ID}-${server_index}.log 2>&1"
done

deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
while true; do
  all_ready=true
  for ((server_index=0; server_index<SERVER_COUNT; server_index++)); do
    port=$((SERVER_PORT_BASE + server_index))
    if ! docker exec "$CONTAINER_NAME" python3 -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/health', timeout=5).read()" \
      >/dev/null 2>&1; then
      all_ready=false
    fi
  done
  if [[ "$all_ready" == true ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    log "SGLang readiness timed out after ${STARTUP_TIMEOUT_SEC}s"
    docker exec "$CONTAINER_NAME" bash -lc "tail -200 /workspace/results/sglang-${RUN_ID}-*.log" || true
    exit 3
  fi
  if ! docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" | grep -qx true; then
    log "container stopped before SGLang became ready"
    exit 3
  fi
  sleep 5
done

log "SGLang is ready; running evaluation"
docker_exec_env=(
  -e PYTHONPATH=/workspace/eval-harness
  -e PYTHONDONTWRITEBYTECODE=1
  -e "EVAL_HARNESS_IMAGE_ID=$IMAGE_ID"
  -e "EVAL_HARNESS_RUN_ID=$RUN_ID"
)
if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
  docker_exec_env+=(-e DEEPSEEK_API_KEY)
fi
if [[ -n "${DEEPSEEK_BASE_URL:-}" ]]; then
  docker_exec_env+=(-e DEEPSEEK_BASE_URL)
fi
set +e
docker exec \
  "${docker_exec_env[@]}" \
  -w /workspace/eval-harness \
  "$CONTAINER_NAME" \
  python3 -m eval_harness "$HARNESS_SUBCOMMAND" --config "$CONFIG"
PIPELINE_STATUS=$?
set -e
log "evaluation exited with status $PIPELINE_STATUS"
exit "$PIPELINE_STATUS"
