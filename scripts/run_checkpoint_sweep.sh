#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Checkpoint sweep: 读取某目录下所有 checkpoint，依次 merge -> 起 SGLang -> 跑评测 -> 记录 success_rate。
#
# 用法（在评测主机 n30285 上执行）：
#   CHECKPOINT_DIR=/ssd/slime-ckpt/edge-runs/webshop_gigpo_alfworld_init_100step/checkpoints \
#   CONFIG=configs/agent/alfworld_sweep.yaml \
#   bash scripts/run_checkpoint_sweep.sh
#
# 环境变量：
#   CHECKPOINT_DIR   (必填) 含 global_step_N/ 或 best_checkpoint/ 子目录的目录（在合并主机上）
#   CONFIG           (必填) 评测配置；backend.model 必须固定为 "sweep-current"
#   BASE_MODEL       合并用基座模型（默认 /ssd/models/Qwen2.5-1.5B-Instruct）
#   MERGE_CONTAINER  合并容器（默认 edge-dev）
#   MERGE_HOST       合并容器所在主机（默认 n30283；用 ssh alias）
#   EVAL_CONTAINER   评测容器（默认 alfworld-eval）
#   GPU_ID           SGLang 用哪张卡（默认 0）
#   MERGE_DIR        合并产物路径（默认 /ssd/models/sweep-current）
#   OUT_CSV          汇总输出 csv（默认 /ssd/slime-ckpt/agent-cpu-smoke/sweep_results.csv）
#
# 前置条件：
#   - 评测容器已把 /ssd/models 挂载为 /workspace/models（只读），且 /ssd/slime-ckpt/agent-cpu-smoke 挂载为 /work
#   - 合并容器（edge-dev）能访问 CHECKPOINT_DIR + 能写 MERGE_DIR
# ============================================================

CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
CONFIG="${CONFIG:?set CONFIG}"
BASE_MODEL="${BASE_MODEL:-/ssd/models/Qwen2.5-1.5B-Instruct}"
MERGE_CONTAINER="${MERGE_CONTAINER:-edge-dev}"
MERGE_HOST="${MERGE_HOST:-10.36.120.89}"   # n30283 内网 IP（主机名互不解析，需用 IP）
EVAL_IP="${EVAL_IP:-10.36.120.105}"        # n30285 内网 IP（供合并主机 rsync 回传）
EVAL_CONTAINER="${EVAL_CONTAINER:-alfworld-sweep}"
GPU_ID="${GPU_ID:-0}"
MERGE_DIR="${MERGE_DIR:-/ssd/models/sweep-current}"
OUT_CSV="${OUT_CSV:-/ssd/slime-ckpt/agent-cpu-smoke/sweep_results.csv}"

RUN_NAME="sweep-current"
OUT_DIR="/ssd/slime-ckpt/agent-cpu-smoke/results/$RUN_NAME"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# --- 1. 枚举 checkpoint，按 step 排序 ---
ckpts=()
declare -A step_of=()

for d in "$CHECKPOINT_DIR"/*/; do
    name="$(basename "$d")"
    if [[ "$name" == global_step_* ]]; then
        step="${name#global_step_}"
    elif [[ "$name" == best_checkpoint ]]; then
        step="$(python3 -c "import json;print(json.load(open('$d/best_checkpoint_info.json'))['global_step'])" 2>/dev/null || echo 0)"
    else
        continue
    fi
    ckpts+=("$name")
    step_of["$name"]="$step"
done

if [[ ${#ckpts[@]} -eq 0 ]]; then
    log "No checkpoints found in $CHECKPOINT_DIR"; exit 1
fi

mapfile -t ckpts < <(for c in "${ckpts[@]}"; do printf '%s %s\n' "${step_of[$c]}" "$c"; done | sort -n | awk '{print $2}')

log "Found ${#ckpts[@]} checkpoints: ${ckpts[*]}"
echo "checkpoint,global_step,success_rate" > "$OUT_CSV"

# --- 2. 逐个 checkpoint ---
for name in "${ckpts[@]}"; do
    step="${step_of[$name]}"
    ckpt_dir="$CHECKPOINT_DIR/$name"
    log "=== [$name] (step $step) merge ==="

    # 2.1 merge FSDP -> HF（在合并主机/容器里）
    ssh "$MERGE_HOST" docker exec "$MERGE_CONTAINER" python \
        /ssd/slime-ckpt/EDGE/scripts/model_merger.py merge \
        --backend fsdp --local_dir "$ckpt_dir/actor" \
        --hf_model_path "$BASE_MODEL" --target_dir "$MERGE_DIR"

    # 2.2 把合并产物同步到评测主机（本机）
    log "syncing merged model from $MERGE_HOST to local"
    ssh "$MERGE_HOST" rsync -a "$MERGE_DIR/" "$EVAL_IP:$MERGE_DIR/"

    # 2.3 起 SGLang（served-model-name=sweep-current）
    log "starting SGLang (GPU $GPU_ID)"
    cat > /tmp/start_sweep.sh <<EOF
sglang serve --model-path /workspace/models/sweep-current --served-model-name sweep-current --tp-size 1 --base-gpu-id $GPU_ID --host 127.0.0.1 --port 31000 --dtype bfloat16 --trust-remote-code --context-length 32768 > /work/sglang.log 2>&1
EOF
    docker cp /tmp/start_sweep.sh "$EVAL_CONTAINER":/work/start_sweep.sh
    docker exec -d "$EVAL_CONTAINER" bash /work/start_sweep.sh

    # 2.4 等 SGLang 就绪
    log "waiting for SGLang"
    ready=0
    for _ in $(seq 1 30); do
        if docker exec "$EVAL_CONTAINER" curl -sf http://127.0.0.1:31000/health >/dev/null 2>&1; then
            ready=1; break
        fi
        sleep 10
    done
    [[ "$ready" == 1 ]] || { log "SGLang not ready"; exit 1; }

    # 2.5 跑评测（先清旧结果避免 fingerprint 冲突）
    log "running eval"
    rm -rf "$OUT_DIR"
    docker exec -e PYTHONPATH=/eval_harness -e ALFWORLD_DATA=/alfworld_data -w /eval_harness \
        "$EVAL_CONTAINER" python -m eval_harness run --config "$CONFIG"

    # 2.6 收集 success rate
    sr="$(python3 -c "import json;print(json.load(open('$OUT_DIR/summary.json'))['accuracy'])" 2>/dev/null || echo ERROR)"
    log "[$name] success_rate=$sr"
    echo "$name,$step,$sr" >> "$OUT_CSV"

    # 2.7 停 SGLang
    log "stopping SGLang"
    docker exec "$EVAL_CONTAINER" pkill -f "sglang serve" 2>/dev/null || true
    sleep 5
done

log "done. results -> $OUT_CSV"
cat "$OUT_CSV"
