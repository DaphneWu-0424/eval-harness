#!/usr/bin/env bash
set -euo pipefail
# Each slot owns a physical GPU and runs its own inference/environment containers.
if [[ -n "${GPU_IDS:-}" ]]; then
    devices="$GPU_IDS"
else
    count="${GPUS:-8}"
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS must be positive" >&2; exit 2; }
    devices="$(seq -s, 0 $((count - 1)))"
fi
args=(--checkpoint-dir "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}" --gpus "$devices" --parallel)
[[ -z "${BASE_MODEL:-}" ]] || args+=(--base-model "$BASE_MODEL")
[[ -z "${MERGE_CONTAINER:-}" ]] || args+=(--merge-container "$MERGE_CONTAINER")
[[ -z "${MERGE_ROOT:-}" ]] || args+=(--merge-root "$MERGE_ROOT")
[[ -z "${RESULTS_DIR:-}" ]] || args+=(--results-root "$RESULTS_DIR")
case "${CONFIG:-}" in
  *alfworld*) args+=(--benchmark alfworld) ;;
  *webshop*) args+=(--benchmark webshop) ;;
  "") ;;
  *) echo "Unsupported CONFIG; use --benchmark alfworld or --benchmark webshop" >&2; exit 2 ;;
esac
exec python3 "$(dirname "$0")/run_checkpoint_sweep.py" "${args[@]}" "$@"
