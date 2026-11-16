#!/usr/bin/env bash
set -euo pipefail
# One checkpoint at a time. Inference and CPU environments are separate containers.
args=(--checkpoint-dir "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}" --gpus "${GPU_ID:-0}")
[[ -z "${BASE_MODEL:-}" ]] || args+=(--base-model "$BASE_MODEL")
[[ -z "${MERGE_CONTAINER:-}" ]] || args+=(--merge-container "$MERGE_CONTAINER")
[[ -z "${MERGE_ROOT:-}" ]] || args+=(--merge-root "$MERGE_ROOT")
[[ -z "${RESULTS_DIR:-}" ]] || args+=(--results-root "$RESULTS_DIR")
# Preserve the legacy CONFIG selector; without it evaluate both environments.
case "${CONFIG:-}" in
  *alfworld*) args+=(--benchmark alfworld) ;;
  *webshop*) args+=(--benchmark webshop) ;;
  "") ;;
  *) echo "Unsupported CONFIG; use --benchmark alfworld or --benchmark webshop" >&2; exit 2 ;;
esac
exec python3 "$(dirname "$0")/run_checkpoint_sweep.py" "${args[@]}" "$@"
