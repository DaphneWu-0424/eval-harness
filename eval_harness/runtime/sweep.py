"""Dynamic GPU queue of independent checkpoint ? dataset evaluations."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import signal
import time
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .separated import AgentSuite, atomic_json, execute, gpu_ids, parser, select_datasets
from .gpu_pool import GpuLease, gpu_status


def discover(root: Path, selected: set[int] | None = None) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path.name.startswith("global_step_") and path.name.removeprefix("global_step_").isdigit():
            step = int(path.name.removeprefix("global_step_"))
        elif path.name == "best_checkpoint" or has_weights(path) or (path / "actor").is_dir():
            step = checkpoint_step(path)
        else:
            continue
        if selected is None or step in selected:
            checkpoints.append((step, path.resolve()))
    if not checkpoints:
        raise ValueError(f"no selected checkpoints found under {root}")
    return sorted(checkpoints, key=lambda x: (x[0] is None, x[0] or 0, x[1].name))


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



class TaskQueue:
    """A shared FIFO of checkpoint × dataset jobs, assigned to currently idle GPUs."""
    def __init__(self, args, tasks, report_dir, state):
        self.args, self.tasks, self.report_dir, self.state = args, tasks, report_dir, state
        self.pending = deque(t for t in tasks if t["status"] == "pending")
        self.active = {}
        self.stopping = False
        self.signal_number = None
        self.simulated = args.backend == "fake"
        inventory = {} if self.simulated else gpu_status()
        self.ids = (list(map(str, range(8))) if self.simulated else sorted(inventory, key=int)) if args.gpus == "auto" else gpu_ids(args.gpus)
        if not self.ids:
            raise ValueError("no visible GPUs found")
        if not self.simulated and any(g not in inventory for g in self.ids):
            raise ValueError("requested GPU IDs are not present on this host")
        self.capacity = min(args.max_parallel or (len(self.ids) if args.parallel else 1), len(self.ids))
        if self.capacity < 1:
            raise ValueError("max-parallel must be positive")
        state.update(gpus=self.ids, max_parallel=self.capacity, simulated_gpu_pool=self.simulated)

    def persist(self):
        self.state["counts"] = {status: sum(t["status"] == status for t in self.tasks)
                                for status in ["pending", "running", "completed", "failed", "cancelled"]}
        atomic_json(self.report_dir / "sweep.json", self.state)
        temporary = self.report_dir / "matrix.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["checkpoint", "checkpoint_path", "global_step", "benchmark", "gpu", "success_rate", "task_score", "aggregation", "status", "run_id", "scenario_success_rate"])
            for task in self.tasks:
                writer.writerow([task["checkpoint"], task["checkpoint_path"], task["step"] if task["step"] is not None else "",
                                 task["benchmark"], task.get("gpu", ""), task.get("metrics", {}).get("accuracy", ""),
                                 task.get("metrics", {}).get("scienceworld_score", ""),
                                 task.get("metrics", {}).get("scienceworld_score_aggregation", "mean_over_episodes"), task["status"], task["run_id"],
                                 task.get('metrics', {}).get('appworld_sgc', '')])
        temporary.replace(self.report_dir / "matrix.csv")

    def request_stop(self, signum=None):
        self.stopping = True
        self.signal_number = signum or self.signal_number
        for task, gpu, lease, suite in list(self.active.values()):
            suite.cancel()

    def collect_finished(self):
        changed = False
        for future, (task, gpu, lease, suite) in list(self.active.items()):
            if not future.done():
                continue
            try:
                report = future.result()
                result = report["benchmarks"][task["benchmark"]]
                task.update(status="completed", metrics=result["metrics"])
            except BaseException as exc:
                task.update(status="cancelled" if self.stopping else "failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                task["ended_at"] = time.time()
                if lease:
                    lease.release()
                self.active.pop(future)
                changed = True
        if changed:
            self.persist()
        return changed

    def assign(self, pool):
        if self.stopping or not self.pending or len(self.active) >= self.capacity:
            return False
        inventory = ({g: {"memory_mib": 0, "has_compute_process": False} for g in self.ids}
                     if self.simulated else gpu_status())
        occupied = {entry[1] for entry in self.active.values()}
        assigned = False
        for gpu in self.ids:
            if not self.pending or len(self.active) >= self.capacity:
                break
            info = inventory.get(gpu)
            if gpu in occupied or not info or info["has_compute_process"] or info["memory_mib"] > self.args.max_gpu_memory_mib:
                continue
            lease = None
            if not self.simulated:
                lease = GpuLease(self.args.lease_dir, gpu, self.args.run_id)
                if not lease.acquire():
                    continue
                # Check again after taking the cross-process lease, before starting the model.
                try:
                    fresh = gpu_status().get(gpu)
                    if not fresh or fresh["has_compute_process"] or fresh["memory_mib"] > self.args.max_gpu_memory_mib:
                        lease.release()
                        continue
                except BaseException:
                    lease.release()
                    raise
            if self.stopping:
                if lease:
                    lease.release()
                break
            task = self.pending.popleft()
            local = copy.deepcopy(self.args)
            local.gpus, local.tp_size = gpu, 1
            local.benchmarks = [task["benchmark"]]
            local.model_dir = task["model_dir"]
            local.model_name = "current-checkpoint"
            local.run_id = task["run_id"]
            suite = AgentSuite(local)
            task.update(status="running", gpu=gpu, started_at=time.time())
            try:
                future = pool.submit(suite.run)
            except BaseException:
                if lease:
                    lease.release()
                task.update(status="failed", error="could not submit task")
                raise
            self.active[future] = (task, gpu, lease, suite)
            occupied.add(gpu)
            print(f"[{self.args.run_id}] GPU {gpu}: {task['checkpoint_path']} × {task['benchmark']}", flush=True)
            self.persist()
            assigned = True
        return assigned

    def run(self):
        self.state["status"] = "running"
        self.persist()
        last_progress = time.monotonic()
        last_message = 0.0
        with ThreadPoolExecutor(max_workers=self.capacity) as pool:
            try:
                while self.pending or self.active:
                    changed = self.collect_finished()
                    if self.stopping:
                        break
                    assigned = self.assign(pool)
                    now = time.monotonic()
                    if changed or assigned:
                        last_progress = now
                    if self.pending and not self.active:
                        self.state["status"] = "waiting_for_gpu"
                        self.persist()
                        if now - last_message >= 60:
                            print(f"[{self.args.run_id}] {len(self.pending)} tasks queued; waiting for an idle GPU", flush=True)
                            last_message = now
                        if self.args.queue_timeout and now - last_progress >= self.args.queue_timeout:
                            raise TimeoutError("no GPU became available before queue-timeout")
                    else:
                        self.state["status"] = "running"
                    if self.pending or self.active:
                        time.sleep(self.args.poll_seconds)
            except BaseException as exc:
                self.state["error"] = f"{type(exc).__name__}: {exc}"
                self.request_stop()
            finally:
                # Stop only this submission's owned containers, then let workers save logs/cleanup.
                if self.stopping:
                    self.request_stop()
                while self.active:
                    self.collect_finished()
                    if self.active:
                        time.sleep(min(self.args.poll_seconds, 0.2))
                while self.pending:
                    self.pending.popleft().update(status="cancelled", error=self.state.get("error", "submission interrupted"))
        failed = any(t["status"] != "completed" for t in self.tasks)
        self.state["status"] = "interrupted" if self.signal_number else "failed" if failed else "completed"
        self.persist()
        return 128 + self.signal_number if self.signal_number else 1 if failed else 0


def checkpoint_step(path):
    match = re.search(r"(?:global_step_|checkpoint[-_]|step[-_]?)(\d+)$", path.name)
    if match:
        return int(match[1])
    for name in ["best_checkpoint_info.json", "verified.json"]:
        info = path / name
        if info.is_file():
            data = json.loads(info.read_text())
            for key in ["global_step", "step", "iteration"]:
                if key in data:
                    return int(data[key])
    return None


def task_id(prefix, checkpoint, dataset):
    identity = hashlib.sha256(f"{prefix}\0{checkpoint.resolve()}\0{dataset}".encode()).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9_.-]", "-", checkpoint.name.lower())[:12].strip(".-") or "checkpoint"
    return f"{prefix[:16]}-{slug}-{dataset[:18]}-{identity}"


def main(argv=None):
    p = parser()
    p.description = "GPU task queue: one checkpoint × one selected dataset per GPU"
    p.set_defaults(gpus="auto")
    p.add_argument("--checkpoint-dir", help="Discover checkpoint subdirectories")
    p.add_argument("--checkpoint", action="append", default=[], help="Explicit checkpoint/export directory; repeatable")
    p.add_argument("--steps", help="Optional comma-separated checkpoint steps")
    p.add_argument("--base-model", default="/ssd/models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--merge-container", default="edge-dev")
    p.add_argument("--merge-root", default="/ssd/models/eval-harness-merged")
    p.add_argument("--parallel", action="store_true", help="Use the whole GPU pool instead of one task at a time")
    p.add_argument("--max-parallel", type=int, help="Optional submission concurrency cap")
    p.add_argument("--poll-seconds", type=float, default=5)
    p.add_argument("--queue-timeout", type=float, default=0, help="Idle wait timeout; 0 waits indefinitely")
    p.add_argument("--lease-dir", default="/tmp/eval-harness-gpu-leases", help="Use the same host directory across queue submissions")
    args = p.parse_args(argv)
    select_datasets(args)
    if args.model_dir:
        args.checkpoint.append(args.model_dir)
    if not args.checkpoint_dir and not args.checkpoint:
        p.error("supply --checkpoint-dir or one or more --checkpoint paths")
    if args.endpoint or args.tp_size != 1:
        p.error("queue jobs own a TP1 model server; external endpoints/TP>1 use run_agent_suite.py")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,60}", args.run_id):
        p.error("run-id must be a short lowercase Docker-safe identifier")
    if args.poll_seconds <= 0 or args.queue_timeout < 0 or (args.max_parallel is not None and args.max_parallel < 1):
        p.error("poll-seconds/max-parallel must be positive and queue-timeout nonnegative")
    selected = {int(s) for s in args.steps.split(",")} if args.steps else None
    checkpoints = discover(Path(args.checkpoint_dir).resolve(), selected) if args.checkpoint_dir else []
    for name in args.checkpoint:
        path = Path(name).resolve()
        if not path.is_dir():
            p.error(f"checkpoint directory not found: {path}")
        step = checkpoint_step(path)
        if selected is None or step in selected:
            checkpoints.append((step, path))
    checkpoints = list({str(path): (step, path) for step, path in checkpoints}.values())
    if not checkpoints:
        p.error("no checkpoints match the selection")
    report_dir = Path(args.results_root).resolve() / f"{args.run_id}-sweep"
    report_dir.mkdir(parents=True, exist_ok=False)
    tasks = [dict(checkpoint=path.name, checkpoint_path=str(path), step=step, benchmark=dataset,
                  run_id=task_id(args.run_id, path, dataset), status="pending")
             for step, path in checkpoints for dataset in args.benchmarks]
    state = {"status": "preparing_models", "run_id": args.run_id, "task_definition": "checkpoint_x_dataset", "tasks": tasks}
    atomic_json(report_dir / "sweep.json", state)
    # Each checkpoint is exported once; its datasets then become independent GPU jobs.
    for step, path in checkpoints:
        try:
            model = prepare_model(path, args)
            for task in tasks:
                if task["checkpoint_path"] == str(path):
                    task["model_dir"] = str(model)
        except Exception as exc:
            for task in tasks:
                if task["checkpoint_path"] == str(path):
                    task.update(status="failed", error=f"model preparation: {exc}")
        atomic_json(report_dir / "sweep.json", state)
    try:
        queue = TaskQueue(args, tasks, report_dir, state)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        atomic_json(report_dir / "sweep.json", state)
        raise
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: queue.request_stop(signum))
    try:
        code = queue.run()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(f"Queue {state['status']}: {report_dir / 'matrix.csv'}", flush=True)
    return code
