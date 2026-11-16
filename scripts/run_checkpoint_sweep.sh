#!/usr/bin/env bash
set -euo pipefail
# One checkpoint at a time. Inference and CPU environments are separate containers.
args=(--gpus "${GPU_ID:-auto}")
[[ -z "${CHECKPOINT_DIR:-}" ]] || args+=(--checkpoint-dir "$CHECKPOINT_DIR")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${BASE_MODEL:-}" ]] || args+=(--base-model "$BASE_MODEL")
[[ -z "${MERGE_CONTAINER:-}" ]] || args+=(--merge-container "$MERGE_CONTAINER")
[[ -z "${MERGE_ROOT:-}" ]] || args+=(--merge-root "$MERGE_ROOT")
[[ -z "${RESULTS_DIR:-}" ]] || args+=(--results-root "$RESULTS_DIR")
# Preserve the legacy CONFIG selector; without it evaluate all registered environments.
case "${CONFIG:-}" in
  *alfworld*) args+=(--benchmark alfworld) ;;
  *webshop*) args+=(--benchmark webshop) ;;
  *scienceworld*) args+=(--benchmark scienceworld) ;;
  "") ;;
  *) echo "Unsupported CONFIG; use --benchmark alfworld, webshop or scienceworld" >&2; exit 2 ;;
esac
exec python3 "$(dirname "$0")/run_checkpoint_sweep.py" "${args[@]}" "$@"
