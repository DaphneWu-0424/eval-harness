#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# 并行 checkpoint sweep：一张卡跑一个 ckpt。
# 先串行 merge 所有 ckpt，再用 xargs -P 并行跑 serve+eval（8 卡）。
#
# 用法（n30285 上执行）：
#   CHECKPOINT_DIR=/ssd/slime-ckpt/edge-runs/webshop_gigpo_alfworld_init_100step_fullckpt/checkpoints \
#   bash /ssd/eval-harness/scripts/run_checkpoint_sweep_parallel.sh
# ============================================================

CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
BASE_MODEL="${BASE_MODEL:-/ssd/models/Qwen2.5-1.5B-Instruct}"
MERGE_CONTAINER="${MERGE_CONTAINER:-edge-dev}"
EVAL_CONTAINER="${EVAL_CONTAINER:-alfworld-sweep}"
MERGE_ROOT="${MERGE_ROOT:-/ssd/models/sweep}"
OUT_CSV="${OUT_CSV:-/ssd/slime-ckpt/agent-cpu-smoke/sweep_results.csv}"
CFG_ROOT="/ssd/slime-ckpt/agent-cpu-smoke/configs"
GPUS="${GPUS:-8}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# --- 1. 枚举 + 排序 ---
ckpts=(); declare -A step_of=()
for d in "$CHECKPOINT_DIR"/*/; do
    name="$(basename "$d")"
    if [[ "$name" == global_step_* ]]; then step="${name#global_step_}"; elif [[ "$name" == best_checkpoint ]]; then step="$(python3 -c "import json;print(json.load(open('$d/best_checkpoint_info.json'))['global_step'])" 2>/dev/null || echo 0)"; else continue; fi
    ckpts+=("$name"); step_of["$name"]="$step"
done
mapfile -t ckpts < <(for c in "${ckpts[@]}"; do printf '%s %s\n' "${step_of[$c]}" "$c"; done | sort -n | awk '{print $2}')
log "found ${#ckpts[@]} checkpoints: ${ckpts[*]}"

# --- 2. 串行 merge 所有 ckpt ---
for name in "${ckpts[@]}"; do
    log "merge $name"
    if [[ -f "$MERGE_ROOT/$name/model.safetensors" ]]; then
        log "skip merge $name (already merged)"
    else
        docker exec "$MERGE_CONTAINER" python /ssd/slime-ckpt/EDGE/scripts/model_merger.py merge \
            --backend fsdp --local_dir "$CHECKPOINT_DIR/$name/actor" \
            --hf_model_path "$BASE_MODEL" --target_dir "$MERGE_ROOT/$name"
    fi
done

# --- 3. 生成每个 ckpt 的 job 脚本 ---
mkdir -p "$CFG_ROOT" /tmp/sweep-jobs
echo "checkpoint,global_step,success_rate" > "$OUT_CSV"
jobfiles=()
for i in "${!ckpts[@]}"; do
    name="${ckpts[$i]}"; step="${step_of[$name]}"
    gpu=$((i % GPUS)); port=$((31000 + gpu))
    cfg="$CFG_ROOT/sweep-$name.yaml"
    sed -e "s/sweep-current/sweep-$name/g" -e "s#31000#$port#" \
        /ssd/eval-harness/configs/agent/alfworld_sweep.yaml > "$cfg"

    job="/tmp/sweep-jobs/$name.sh"
    cat > "$job" <<EOF
#!/bin/bash
set -euo pipefail
name="$name"; step="$step"; gpu=$gpu; port=$port
log() { printf '[%s] [%s] %s\n' "\$(date -u +%H:%M:%S)" "\$name" "\$*"; }

log "start SGLang (gpu \$gpu port \$port)"
cat > /tmp/start_\$name.sh <<INNER
sglang serve --model-path /workspace/models/sweep/\$name --served-model-name sweep-\$name --tp-size 1 --base-gpu-id \$gpu --host 127.0.0.1 --port \$port --dtype bfloat16 --trust-remote-code --context-length 32768 > /work/sglang-\$name.log 2>&1
INNER
docker cp /tmp/start_\$name.sh "$EVAL_CONTAINER":/work/start_\$name.sh
docker exec -d "$EVAL_CONTAINER" bash /work/start_\$name.sh

ready=0
for _ in \$(seq 1 30); do
    docker exec "$EVAL_CONTAINER" curl -sf http://127.0.0.1:\$port/health >/dev/null 2>&1 && { ready=1; break; }
    sleep 10
done
[ "\$ready" == 1 ] || { log "SGLang timeout"; exit 1; }

log "run eval"
rm -rf /ssd/slime-ckpt/agent-cpu-smoke/results/sweep-\$name
docker exec -e PYTHONPATH=/eval_harness -e ALFWORLD_DATA=/alfworld_data -w /eval_harness \
    "$EVAL_CONTAINER" python -m eval_harness run --config "/work/configs/sweep-\$name.yaml"

sr="\$(python3 -c "import json;print(json.load(open('/ssd/slime-ckpt/agent-cpu-smoke/results/sweep-\$name/summary.json'))['accuracy'])" 2>/dev/null || echo ERROR)"
echo "\$name,\$step,\$sr" >> "$OUT_CSV"
log "success_rate=\$sr"

# 停掉本 ckpt 的 SGLang，释放显存给排队的下一个 ckpt
docker exec "$EVAL_CONTAINER" pkill -f "sweep-\$name" 2>/dev/null || true
sleep 3
EOF
    chmod +x "$job"
    jobfiles+=("$job")
done

# --- 4. 并行执行（xargs -P 控制并发度） ---
log "running ${#jobfiles[@]} jobs with -P $GPUS"
printf '%s\n' "${jobfiles[@]}" | xargs -P "$GPUS" -I{} bash {}

log "done. results -> $OUT_CSV"
sort -t, -k2 -n "$OUT_CSV" -o "$OUT_CSV"
cat "$OUT_CSV"
