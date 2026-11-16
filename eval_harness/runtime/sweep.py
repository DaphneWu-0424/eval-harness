"""Checkpoint scheduling with one independently owned inference container per GPU slot."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .separated import AgentSuite, atomic_json, execute, gpu_ids, parser


def discover(root: Path, selected: set[int] | None = None) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path.name.startswith("global_step_") and path.name.removeprefix("global_step_").isdigit():
            step = int(path.name.removeprefix("global_step_"))
        elif path.name == "best_checkpoint":
            info = path / "best_checkpoint_info.json"
            if not info.is_file():
                raise ValueError(f"missing checkpoint step metadata: {info}")
            step = int(json.loads(info.read_text())["global_step"])
        else:
            continue
        if selected is None or step in selected:
            checkpoints.append((step, path))
    if not checkpoints:
        raise ValueError(f"no selected checkpoints found under {root}")
    return sorted(checkpoints, key=lambda x: (x[0], x[1].name))


def has_weights(path: Path) -> bool:
    return (path / "config.json").is_file() and bool(list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin")))


def prepare_model(path: Path, args) -> Path:
    if has_weights(path):
        return path
    actor = path / "actor"
    if not actor.is_dir():
        raise ValueError(f"checkpoint contains neither exported HF weights nor actor shards: {path}")
    source = {
        "checkpoint": str(path.resolve()), "base_model": str(Path(args.base_model).resolve()),
        "files": [(str(p.relative_to(actor)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in sorted(actor.rglob("*")) if p.is_file()],
    }
    signature = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    # Include source identity in the cache path so two experiments' step100 cannot collide.
    target = Path(args.merge_root).resolve() / f"{path.name}-{signature[:12]}"
    state_file = target / "merge-state.json"
    if target.exists():
        state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        if state.get("signature") == signature and state.get("status") == "completed" and has_weights(target):
            return target
        raise ValueError(f"incomplete or unowned merge target; choose a new merge-root: {target}")
    target.mkdir(parents=True)
    state = {"signature": signature, "source": source, "status": "running"}
    atomic_json(state_file, state)
    with (target / "merge.log").open("w", encoding="utf-8") as log:
        try:
            execute(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", args.merge_container, "python",
                     "/ssd/slime-ckpt/EDGE/scripts/model_merger.py", "merge", "--backend", "fsdp",
                     "--local_dir", str(actor), "--hf_model_path", args.base_model, "--target_dir", str(target)],
                    stdout=log, stderr=-2)
            if not has_weights(target):
                raise RuntimeError("merger exited without usable HF weights")
            state["status"] = "completed"
        except BaseException:
            state["status"] = "failed"
            raise
        finally:
            atomic_json(state_file, state)
    return target


def main(argv=None):
    p = parser()
    p.description = "Evaluate checkpoints with separate SGLang and CPU environment containers"
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--steps", help="Optional comma-separated checkpoint steps")
    p.add_argument("--base-model", default="/ssd/models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--merge-container", default="edge-dev")
    p.add_argument("--merge-root", default="/ssd/models/eval-harness-merged")
    p.add_argument("--parallel", action="store_true")
    args = p.parse_args(argv)
    args.benchmarks = list(dict.fromkeys(args.benchmarks or ["alfworld", "webshop"]))
    if args.endpoint:
        p.error("a checkpoint sweep must own and switch its model server; --endpoint is for single-model suites")
    if args.tp_size != 1:
        p.error("checkpoint sweep uses one TP1 model per GPU slot; use run_agent_suite.py for TP>1")
    ids = gpu_ids(args.gpus)
    if not args.parallel and len(ids) != 1:
        p.error("serial sweep requires one GPU; use --parallel for multiple GPU IDs")
    selected = {int(s) for s in args.steps.split(",")} if args.steps else None
    checkpoints = discover(Path(args.checkpoint_dir).resolve(), selected)
    report_dir = Path(args.results_root).resolve() / f"{args.run_id}-sweep"
    report_dir.mkdir(parents=True, exist_ok=False)
    state = {"status": "preparing", "run_id": args.run_id, "gpus": ids, "checkpoints": []}
    atomic_json(report_dir / "sweep.json", state)
    # Merge serially, outside inference containers; model export is independent of environment execution.
    try:
        jobs = [(step, path, prepare_model(path, args)) for step, path in checkpoints]
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        atomic_json(report_dir / "sweep.json", state)
        raise
    state["status"] = "running"
    lock = threading.RLock()
    stop = threading.Event()
    active: list[AgentSuite] = []

    def persist():
        atomic_json(report_dir / "sweep.json", state)
        temp = report_dir / "matrix.csv.tmp"
        with temp.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["checkpoint", "global_step", "benchmark", "success_rate", "status", "run_id"])
            for item in sorted(state["checkpoints"], key=lambda r: (r["step"], r["checkpoint"])):
                for benchmark in args.benchmarks:
                    result = item.get("benchmarks", {}).get(benchmark, {})
                    writer.writerow([item["checkpoint"], item["step"], benchmark, result.get("metrics", {}).get("accuracy", ""), result.get("status", item["status"]), item["run_id"]])
        temp.replace(report_dir / "matrix.csv")

    def worker(slot):
        # A slot owns one physical GPU for its whole queue, so differing job lengths cannot collide.
        for step, checkpoint, model in jobs[slot::len(ids)]:
            if stop.is_set():
                return
            local = copy.deepcopy(args)
            local.gpus = ids[slot]
            local.model_dir = str(model)
            local.model_name = "current-checkpoint"
            local.run_id = f"{args.run_id}-{checkpoint.name}"
            suite = AgentSuite(local)
            item = {"checkpoint": checkpoint.name, "step": step, "run_id": local.run_id, "status": "running"}
            with lock:
                state["checkpoints"].append(item)
                active.append(suite)
                persist()
            try:
                result = suite.run()
                item.update(status="completed", benchmarks=result["benchmarks"])
            except BaseException as exc:
                item.update(status="failed", error=str(exc), benchmarks=suite.artifacts["benchmarks"])
            finally:
                with lock:
                    active.remove(suite)
                    persist()

    def terminate(signum, frame):
        stop.set()
        with lock:
            for suite in list(active):
                suite.cancel()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        with ThreadPoolExecutor(max_workers=len(ids)) as pool:
            list(pool.map(worker, range(len(ids))))
        state["status"] = "completed" if all(c["status"] == "completed" for c in state["checkpoints"]) else "failed"
    finally:
        if stop.is_set():
            state["status"] = "interrupted"
        with lock:
            persist()
    print(f"Sweep {state['status']}: {report_dir / 'matrix.csv'}", flush=True)
    return 0 if state["status"] == "completed" else 1
