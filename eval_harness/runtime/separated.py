from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

SGLANG_IMAGE = "lmsysorg/sglang:0.5.19-qwen38-runtime"
ENV_IMAGES = {"alfworld": "eval-harness/alfworld-env:20261116", "webshop": "eval-harness/webshop-env:20261116",
              "scienceworld": "eval-harness/scienceworld-env:20261116",
              "appworld": "ghcr.io/stonybrooknlp/appworld:latest"}
LABEL = "eval-harness.run_id"


def select_datasets(args: argparse.Namespace) -> None:
    overrides = {}
    for item in args.dataset_configs:
        name, separator, path = item.partition("=")
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,23}", name) or not path:
            raise ValueError("--dataset must be NAME=CONFIG.yaml with a short lowercase name")
        if name in overrides or name in (args.benchmarks or []):
            raise ValueError(f"duplicate dataset name: {name}")
        overrides[name] = path
    args.config_overrides = overrides
    args.benchmarks = list(dict.fromkeys([*(args.benchmarks or []), *overrides])) or list(ENV_IMAGES)


def execute(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, **kwargs)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def gpu_ids(value: str) -> list[str]:
    ids = value.split(",")
    if not ids or any(not re.fullmatch(r"\d+", item) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("gpus must be distinct numeric physical GPU IDs, for example 0 or 0,1")
    return ids


def resolved_config(base: dict, benchmark: str, args: argparse.Namespace, endpoint: str) -> dict:
    config = copy.deepcopy(base)
    config.pop("_resolved", None)
    config["experiment"]["name"] = f"{args.run_id}-{benchmark}"
    config["experiment"]["output_dir"] = f"/results/{benchmark}"
    if args.limit is not None:
        config["dataset"]["limit"] = args.limit
    if args.max_steps is not None:
        config["environment"]["max_steps"] = args.max_steps
    if args.backend == "fake":
        action = "search[shirt]" if config["environment"]["type"] == "webshop" else "look"
        config["backend"] = {"type": "fake", "responses": [f"<think>Inspect the environment.</think><action>{action}</action>"]}
        if config['environment']['type'] == 'appworld':
            config['backend']['responses'] = ['<think>Inspect apps.</think><code>print(apis.api_docs.show_app_descriptions())</code>']
    else:
        config["backend"] = {"type": "openai_compatible", "model": args.model_name, "endpoints": [endpoint]}
    config["generation"]["repetition_penalty"] = args.repetition_penalty
    config["runtime_metadata"] = {
        "layout": "separated", "run_id": args.run_id,
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "model_endpoint": endpoint if args.backend != "fake" else None,
    }
    return config


class AgentSuite:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.results_root).resolve() / args.run_id
        self.network = f"eh-{args.run_id}"
        self.model_container = f"eh-{args.run_id}-inference"
        self.network_created = False
        self.containers: list[str] = []
        self.cancelled = threading.Event()
        self.artifacts: dict[str, Any] = {"run_id": args.run_id, "status": "pending", "benchmarks": {}}

    def docker(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess:
        return execute(["docker", *args], **kwargs)

    def check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise InterruptedError("suite cancelled")

    def cancel(self) -> None:
        """Wake a worker blocked on docker wait without touching another run's containers."""
        self.cancelled.set()
        for name in list(self.containers):
            record = subprocess.run(["docker", "inspect", "--format", "{{json .Config.Labels}}", name], capture_output=True, text=True)
            if record.returncode == 0 and json.loads(record.stdout).get(LABEL) == self.args.run_id:
                subprocess.run(["docker", "stop", "--time", "5", name], capture_output=True, text=True)

    def inspect_image(self, tag: str) -> str:
        return self.docker("image", "inspect", "--format", "{{.Id}}", tag, capture_output=True).stdout.strip()

    def labels(self, role: str) -> list[str]:
        return ["--label", f"{LABEL}={self.args.run_id}", "--label", f"eval-harness.role={role}"]

    def preflight(self) -> dict[str, dict]:
        args = self.args
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,60}", args.run_id):
            raise ValueError("run-id must be a short lowercase Docker-safe identifier")
        for field in ("limit", "max_steps"):
            value = getattr(args, field)
            if value is not None and value < 1:
                raise ValueError(f"{field} must be positive")
        if args.startup_timeout <= 0 or args.context_length < 1 or args.repetition_penalty <= 0:
            raise ValueError("timeout, context length and repetition penalty must be positive")
        if not 0 < args.mem_fraction_static < 1:
            raise ValueError("mem-fraction-static must be between 0 and 1")
        project = Path(args.project_dir).resolve()
        work = Path(args.work_dir).resolve()
        for path in (project / "eval_harness",):
            if not path.is_dir():
                raise ValueError(f"required directory not found: {path}")
        configs, images = {}, {}
        for benchmark in args.benchmarks:
            override = getattr(args, "config_overrides", {}).get(benchmark)
            path = (project / override) if override else project / "configs" / "agent" / f"{benchmark}.yaml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            env_type = config["environment"]["type"]
            if env_type not in ENV_IMAGES:
                raise ValueError(f"unsupported environment in {path}: {env_type}")
            dataset = config["dataset"]["path"]
            if dataset.startswith("/work/"):
                dataset_path = work / dataset.removeprefix("/work/")
            elif dataset.startswith("/eval_harness/"):
                dataset_path = project / dataset.removeprefix("/eval_harness/")
            else:
                raise ValueError(f"dataset is not available through a declared mount: {dataset}")
            if not dataset_path.is_file():
                raise ValueError(f"dataset is missing: {dataset_path}")
            if env_type in ("alfworld", "webshop"):
                for directory in [work, Path(args.env_package_dir), Path(args.datasets_dir) / env_type]:
                    if not directory.is_dir():
                        raise ValueError(f"missing environment resource directory: {directory}")
            if env_type == "webshop":
                if not (work / "webshop/search_engine/indexes_1k").is_dir():
                    raise ValueError("WebShop Lucene indexes_1k is missing")
                data = work / "webshop/data"
                if not (data.exists() or (data.is_symlink() and os.readlink(data) == "/webshop_data/data")):
                    raise ValueError("WebShop package data link is missing or incompatible")
            if env_type == 'appworld' and not (Path(args.datasets_dir) / 'appworld/data/tasks').is_dir():
                raise ValueError('AppWorld data/tasks missing under datasets-dir/appworld/data')
            tag = getattr(args, f"{env_type}_image")
            images[benchmark] = {"tag": tag, "id": self.inspect_image(tag), "environment": env_type}
            configs[benchmark] = config
        if args.backend != "fake" and not args.endpoint:
            if not args.model_dir or not (Path(args.model_dir) / "config.json").is_file():
                raise ValueError("model-dir must contain an exported Hugging Face model")
            self.artifacts["inference_image"] = {"tag": args.sglang_image, "id": self.inspect_image(args.sglang_image)}
            ids = gpu_ids(args.gpus)
            if len(ids) != args.tp_size:
                raise ValueError("number of selected GPUs must equal tp-size")
            output = execute(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True).stdout
            available = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in output.splitlines() if line.strip()}
            for gpu in ids:
                if gpu not in available or available[gpu] > args.max_gpu_memory_mib:
                    raise ValueError(f"GPU {gpu} missing or busy; refusing to start inference")
            self.artifacts["gpus"] = ids
        self.artifacts["environment_images"] = images
        return configs

    @staticmethod
    def mount(source: Path | str, destination: str, readonly: bool = True) -> list[str]:
        source = str(Path(source).resolve())
        if "," in source:
            raise ValueError("Docker mount source must not contain commas")
        return ["--mount", f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")]

    def start_inference(self) -> None:
        self.check_cancelled()
        args = self.args
        ids = gpu_ids(args.gpus)
        devices = f"device={','.join(ids)}"
        if len(ids) > 1:
            devices = json.dumps(devices)
        command = ["run", "-d", "--init", "--name", self.model_container,
                   *self.labels("inference"), "--network", self.network, "--network-alias", "inference",
                   "--gpus", devices, "--ipc", "host",
                   *self.mount(args.model_dir, "/workspace/model"), "--entrypoint", "python3",
                   self.artifacts["inference_image"]["id"], "-m", "sglang.launch_server",
                   "--model-path", "/workspace/model", "--served-model-name", args.model_name,
                   "--tp-size", str(args.tp_size), "--host", "0.0.0.0", "--port", "31000",
                   "--dtype", "bfloat16", "--trust-remote-code", "--context-length", str(args.context_length),
                   "--mem-fraction-static", str(args.mem_fraction_static)]
        if args.reasoning_parser:
            command.extend(["--reasoning-parser", args.reasoning_parser])
        self.containers.append(self.model_container)
        self.docker(*command, capture_output=True)
        self.check_cancelled()
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            self.check_cancelled()
            state = self.docker("inspect", "--format", "{{.State.Running}}", self.model_container, capture_output=True).stdout.strip()
            if state != "true":
                raise RuntimeError("inference container exited during startup")
            check = subprocess.run(["docker", "exec", self.model_container, "python3", "-c",
                                    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:31000/health',timeout=3).read()"],
                                   capture_output=True, text=True)
            if check.returncode == 0:
                return
            time.sleep(3)
        raise TimeoutError("SGLang readiness timed out")

    def run_environment(self, benchmark: str, config: dict) -> None:
        self.check_cancelled()
        args = self.args
        env_type = config["environment"]["type"]
        name = f"eh-{args.run_id}-{benchmark}"
        config_path = self.root / "configs" / f"{benchmark}.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        command = ["run", "-d", "--init", "--name", name, *self.labels(benchmark), "--network", self.network,
                   "--env", "PYTHONPATH=/eval_harness", "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
                   *self.mount(args.project_dir, "/eval_harness"), *self.mount(self.root, "/results", readonly=False),
                   "--workdir", "/eval_harness"]
        if env_type in ("alfworld", "webshop"):
            command += [*self.mount(args.env_package_dir, "/env_package"), *self.mount(args.work_dir, "/work")]
        if env_type == "alfworld":
            command += [*self.mount(Path(args.datasets_dir) / "alfworld", "/alfworld_data"), "--env", "ALFWORLD_DATA=/alfworld_data"]
        elif env_type == "webshop":
            command += self.mount(Path(args.datasets_dir) / "webshop", "/webshop_data")
        elif env_type == 'scienceworld':
            command += ["--env", "JAVA_TOOL_OPTIONS=-Djava.io.tmpdir=/tmp -Duser.home=/tmp"]
        elif env_type == 'appworld':
            native = self.root / f'{benchmark}-native'
            native.mkdir()
            command += [*self.mount(Path(args.datasets_dir) / 'appworld/data', '/run/data'),
                        *self.mount(native, '/run/experiments/outputs', readonly=False),
                        '--env', 'APPWORLD_ROOT=/run']
        python = '/opt/venv/bin/python' if env_type == 'appworld' else 'python3'
        command += ["--entrypoint", python, self.artifacts["environment_images"][benchmark]["id"],
                    "-m", "eval_harness", "run", "--config", f"/results/configs/{benchmark}.yaml"]
        self.containers.append(name)
        self.docker(*command, capture_output=True)
        self.check_cancelled()
        code = int(self.docker("wait", name, capture_output=True).stdout.strip())
        self.check_cancelled()
        if code:
            raise RuntimeError(f"{benchmark} evaluation exited with status {code}")
        metrics = json.loads((self.root / benchmark / "summary.json").read_text(encoding="utf-8"))
        if metrics["counts"]["generation_failed_or_missing"] or metrics["counts"].get("judge_unknown", 0):
            raise RuntimeError(f"{benchmark} has failed or missing episodes; inspect its trajectory and logs")
        self.artifacts["benchmarks"][benchmark] = {"status": "completed", "summary": f"{benchmark}/summary.json", "metrics": metrics}

    def cleanup(self) -> None:
        errors = []
        for name in reversed(self.containers):
            record = subprocess.run(["docker", "inspect", "--format", "{{json .Config.Labels}}", name], capture_output=True, text=True)
            if record.returncode:
                continue
            labels = json.loads(record.stdout)
            if labels.get(LABEL) != self.args.run_id:
                errors.append(f"refusing to remove {name}: ownership mismatch")
                continue
            logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True, errors="replace")
            (self.root / "logs" / f"{name}.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
            inspection = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
            if inspection.returncode == 0:
                (self.root / "logs" / f"{name}.inspect.json").write_text(inspection.stdout, encoding="utf-8")
            else:
                errors.append(f"cannot save inspection for {name}: {inspection.stderr.strip()}")
            removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
            if removed.returncode:
                errors.append(f"cannot remove {name}: {removed.stderr.strip()}")
        if self.network_created:
            removed = subprocess.run(["docker", "network", "rm", self.network], capture_output=True, text=True)
            if removed.returncode:
                errors.append(f"cannot remove network {self.network}: {removed.stderr.strip()}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def run(self) -> dict:
        configs = self.preflight()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "configs").mkdir()
        (self.root / "logs").mkdir()
        self.artifacts.update(started_at=dt.datetime.now(dt.timezone.utc).isoformat(), status="running")
        self.artifacts["protocol"] = {"repetition_penalty": self.args.repetition_penalty, "limit": self.args.limit, "max_steps": self.args.max_steps}
        atomic_json(self.root / "suite.json", self.artifacts)
        error = None
        try:
            self.check_cancelled()
            self.docker("network", "create", *self.labels("network"), self.network, capture_output=True)
            self.network_created = True
            self.check_cancelled()
            endpoint = self.args.endpoint or "http://inference:31000/v1"
            if self.args.backend != "fake" and not self.args.endpoint:
                self.start_inference()
            for benchmark in self.args.benchmarks:
                print(f"[{self.args.run_id}] evaluating {benchmark}", flush=True)
                self.run_environment(benchmark, resolved_config(configs[benchmark], benchmark, self.args, endpoint))
                atomic_json(self.root / "suite.json", self.artifacts)
            self.artifacts["status"] = "completed"
        except BaseException as exc:
            error = exc
            self.artifacts.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                self.cleanup()
                self.artifacts["cleanup"] = "completed"
            except BaseException as exc:
                self.artifacts["cleanup"] = f"failed: {exc}"
                if error is None:
                    error = exc
                    self.artifacts.update(status="failed", error=str(exc))
            self.artifacts["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(self.root / "suite.json", self.artifacts)
        if error is not None:
            raise error
        return self.artifacts


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run CPU environment containers against a separate model server")
    result.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[2]))
    result.add_argument("--work-dir", default="/ssd/slime-ckpt/agent-cpu-smoke")
    result.add_argument("--datasets-dir", default="/ssd/datasets")
    result.add_argument("--env-package-dir", default="/ssd/slime-ckpt/EDGE/agent_system/environments/env_package")
    result.add_argument("--results-root", default="/ssd/eval-results/separated")
    result.add_argument("--run-id", default=dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    result.add_argument("--benchmark", dest="benchmarks", action="append", choices=list(ENV_IMAGES))
    result.add_argument("--dataset", dest="dataset_configs", action="append", default=[], metavar="NAME=CONFIG.yaml", help="A named dataset configuration using a supported environment; repeatable")
    result.add_argument("--model-dir")
    result.add_argument("--model-name", default="current-checkpoint")
    result.add_argument("--endpoint", help="Use an existing endpoint instead of starting SGLang")
    result.add_argument("--backend", choices=["openai_compatible", "fake"], default="openai_compatible")
    result.add_argument("--sglang-image", default=SGLANG_IMAGE)
    for name, tag in ENV_IMAGES.items():
        result.add_argument(f"--{name}-image", default=tag)
    result.add_argument("--gpus", default="0")
    result.add_argument("--tp-size", type=int, default=1)
    result.add_argument("--context-length", type=int, default=32768)
    result.add_argument("--mem-fraction-static", type=float, default=0.7)
    result.add_argument("--startup-timeout", type=float, default=1800)
    result.add_argument("--max-gpu-memory-mib", type=int, default=1024)
    result.add_argument("--reasoning-parser", default="")
    result.add_argument("--repetition-penalty", type=float, default=1.0, help="Explicit protocol; 1.0 disables repetition penalty")
    result.add_argument("--limit", type=int)
    result.add_argument("--max-steps", type=int)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    select_datasets(args)
    suite = AgentSuite(args)
    def terminate(signum, frame):
        suite.cancel()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        report = suite.run()
        print(json.dumps({"run_id": report["run_id"], "status": report["status"], "benchmarks": list(report["benchmarks"])}))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"agent suite failed: {type(exc).__name__}: {exc}", flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
