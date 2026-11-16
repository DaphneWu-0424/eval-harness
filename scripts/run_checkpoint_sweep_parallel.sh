#!/usr/bin/env bash
set -euo pipefail
# Shared FIFO: one checkpoint x one dataset is one independently scheduled GPU job.
if [[ -n "${GPU_IDS:-}" ]]; then
    devices="$GPU_IDS"
elif [[ -n "${GPUS:-}" ]]; then
    count="$GPUS"
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS must be positive" >&2; exit 2; }
    devices="$(seq -s, 0 $((count - 1)))"
else
    devices=auto
fi
args=(--gpus "$devices" --parallel)
[[ -z "${CHECKPOINT_DIR:-}" ]] || args+=(--checkpoint-dir "$CHECKPOINT_DIR")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${BASE_MODEL:-}" ]] || args+=(--base-model "$BASE_MODEL")
[[ -z "${MERGE_CONTAINER:-}" ]] || args+=(--merge-container "$MERGE_CONTAINER")
[[ -z "${MERGE_ROOT:-}" ]] || args+=(--merge-root "$MERGE_ROOT")
[[ -z "${RESULTS_DIR:-}" ]] || args+=(--results-root "$RESULTS_DIR")
case "${CONFIG:-}" in
  *alfworld*) args+=(--benchmark alfworld) ;;
  *webshop*) args+=(--benchmark webshop) ;;
  *scienceworld*) args+=(--benchmark scienceworld) ;;
  "") ;;
  *) echo "Unsupported CONFIG; use --benchmark alfworld, webshop or scienceworld" >&2; exit 2 ;;
esac
exec python3 "$(dirname "$0")/run_checkpoint_sweep.py" "${args[@]}" "$@"
