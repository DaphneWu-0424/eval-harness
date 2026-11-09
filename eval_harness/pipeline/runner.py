from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from eval_harness import __version__
from eval_harness.config import dump_config, execution_fingerprint
from eval_harness.core.models import EvaluationResult, TaskSample, Trajectory
from eval_harness.core.registry import (
    BACKENDS,
    DATASETS,
    ENVIRONMENTS,
    EVALUATORS,
    EXECUTORS,
    HARNESSES,
    PROMPTS,
)
from eval_harness.metrics import compute_metrics
from eval_harness.persistence import ExperimentStore
from eval_harness.reports import render_text_report

# Import modules for registry side effects.
from eval_harness import backends as _backends  # noqa: F401
from eval_harness import datasets as _datasets  # noqa: F401
from eval_harness import environments as _environments  # noqa: F401
from eval_harness import evaluators as _evaluators  # noqa: F401
from eval_harness import executors as _executors  # noqa: F401
from eval_harness import harnesses as _harnesses  # noqa: F401
from eval_harness import prompts as _prompts  # noqa: F401


class EvaluationPipeline:
    def __init__(
        self, config: dict[str, Any], *, raw_input: str | list[str] | None = None
    ):
        self.config = config
        self.store = ExperimentStore(config["experiment"]["output_dir"])
        self.raw_input = raw_input
        self.dataset = DATASETS.create(config["dataset"])
        self.backend = BACKENDS.create(
            config["backend"], generation_config=config.get("generation", {})
        )
        if config["executor"].get("type") == "agent":
            environment = ENVIRONMENTS.create(config["environment"])
            harness = HARNESSES.create(
                config["harness"],
                model_backend=self.backend,
                generation_config=config.get("generation", {}),
            )
            self.executor = EXECUTORS.create(
                config["executor"],
                model_backend=self.backend,
                environment=environment,
                harness=harness,
                generation_config=config.get("generation", {}),
                max_steps=int(config["environment"].get("max_steps", 50)),
                base_seed=int(config.get("experiment", {}).get("seed", 0)),
            )
            self.prompt_builder = None
        else:
            self.prompt_builder = PROMPTS.create(config.get("prompt", {"type": "math_boxed"}))
            self.executor = EXECUTORS.create(
                config["executor"],
                backend=self.backend,
                prompt_builder=self.prompt_builder,
                generation_config=config.get("generation", {}),
                base_seed=int(config.get("experiment", {}).get("seed", 0)),
            )
        evaluator_configs = config.get("evaluators", [])
        if len(evaluator_configs) != 1:
            raise ValueError("first release requires exactly one evaluator")
        evaluator_config = evaluator_configs[0]
        fallback = evaluator_config.get("llm_fallback") or {}
        self.auxiliary_backends = []
        llm_backend = None
        if fallback.get("enabled"):
            llm_backend = BACKENDS.create(
                fallback["backend"], generation_config=fallback.get("generation", {})
            )
            self.auxiliary_backends.append(llm_backend)
        self.evaluation_concurrency = int(fallback.get("concurrency", 8))
        self.evaluator = EVALUATORS.create(evaluator_config, llm_backend=llm_backend)

    async def run(self, *, stage: str = "all") -> dict[str, Any]:
        started = time.time()
        samples = self.dataset.load()
        self._initialize_artifacts(samples, started)
        if self.raw_input:
            inputs = [self.raw_input] if isinstance(self.raw_input, str) else self.raw_input
            for input_path in inputs:
                self.store.import_trajectories(input_path)
        trajectories = self.store.load_trajectories()
        try:
            if stage in {"all", "execute"}:
                self._write_state("running", "pending", "pending", "pending")
                health_backends = [self.backend]
                if stage == "all":
                    health_backends.extend(self.auxiliary_backends)
                await asyncio.gather(*(backend.health_check() for backend in health_backends))
                trajectories = await self._execute(samples, trajectories)
                self._write_state("completed", "pending", "pending", "pending")
            if stage == "execute":
                return {"counts": {"stored_trajectories": len(trajectories)}}
            evaluations = self.store.load_evaluations()
            self._write_state("completed", "running", "pending", "pending")
            if stage != "all":
                await asyncio.gather(
                    *(backend.health_check() for backend in self.auxiliary_backends)
                )
            evaluations = await self._evaluate(samples, trajectories, evaluations)
            self._write_state("completed", "completed", "running", "pending")
            metrics = compute_metrics(
                samples,
                list(trajectories.values()),
                list(evaluations.values()),
                self.config["metrics"],
                int(self.config["generation"]["n"]),
            )
            self._write_state("completed", "completed", "completed", "running")
            self._write_reports(metrics)
            self._write_state("completed", "completed", "completed", "completed")
            self._finalize_metadata(started, time.time())
            return metrics
        except BaseException as exc:
            self.store.write_json_atomic(
                "failure.json",
                {"type": type(exc).__name__, "message": str(exc), "time": time.time()},
            )
            raise
        finally:
            await self.backend.close()
            await asyncio.gather(
                *(backend.close() for backend in self.auxiliary_backends),
                return_exceptions=True,
            )

    async def _execute(
        self,
        samples: list[TaskSample],
        existing: dict[tuple[str, int], Trajectory],
    ) -> dict[tuple[str, int], Trajectory]:
        rollout_count = int(self.config["generation"]["n"])
        concurrency = int(self.config["generation"]["concurrency"])
        semaphore = asyncio.Semaphore(concurrency)

        async def execute_one(sample: TaskSample, rollout_id: int) -> Trajectory:
            async with semaphore:
                return await self.executor.run(sample, rollout_id)

        pending = []
        for sample in samples:
            for rollout_id in range(rollout_count):
                previous = existing.get((sample.sample_id, rollout_id))
                if previous is None or previous.status != "completed":
                    pending.append(asyncio.create_task(execute_one(sample, rollout_id)))
        for future in asyncio.as_completed(pending):
            trajectory = await future
            self.store.append_trajectory(trajectory)
            existing[trajectory.key] = trajectory
        return existing

    async def _evaluate(
        self,
        samples: list[TaskSample],
        trajectories: dict[tuple[str, int], Trajectory],
        existing: dict[tuple[str, int], EvaluationResult],
    ) -> dict[tuple[str, int], EvaluationResult]:
        sample_by_id = {sample.sample_id: sample for sample in samples}
        semaphore = asyncio.Semaphore(self.evaluation_concurrency)

        async def evaluate_one(
            trajectory: Trajectory,
        ) -> EvaluationResult:
            async with semaphore:
                return await self.evaluator.evaluate(
                    sample_by_id[trajectory.sample_id], trajectory
                )

        pending = []
        for key, trajectory in trajectories.items():
            previous = existing.get(key)
            should_refresh = previous is None or (
                trajectory.status == "completed"
                and previous.details.get("reason") == "generation_failed"
            ) or previous.evaluator != self.evaluator.identity
            if not should_refresh:
                continue
            pending.append(asyncio.create_task(evaluate_one(trajectory)))
        for future in asyncio.as_completed(pending):
            evaluation = await future
            self.store.append_evaluation(evaluation)
            existing[evaluation.key] = evaluation
        return existing

    def _initialize_artifacts(self, samples: list[TaskSample], started: float) -> None:
        config_path = self.store.root / "config.yaml"
        current_hash = self.config.get("_resolved", {}).get("config_hash")
        current_execution_fingerprint = execution_fingerprint(self.config)
        current_dataset_sha256 = sha256_file(Path(self.config["dataset"]["path"]))
        metadata_path = self.store.root / "metadata.json"
        previous_metadata: dict[str, Any] = {}
        if metadata_path.exists():
            previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = previous_metadata
            previous_execution_fingerprint = metadata.get("execution_fingerprint")
            if (
                previous_execution_fingerprint
                and previous_execution_fingerprint != current_execution_fingerprint
            ):
                raise ValueError(
                    "output directory contains incompatible execution settings; choose a new output_dir"
                )
            previous_dataset_sha256 = metadata.get("dataset_sha256")
            if previous_dataset_sha256 and previous_dataset_sha256 != current_dataset_sha256:
                raise ValueError("dataset content changed; choose a new output_dir")
        if not config_path.exists():
            dump_config(self.config, config_path)
        elif previous_metadata.get("config_hash") != current_hash:
            history_name = f"config.{str(current_hash)[:12]}.yaml"
            dump_config(self.config, self.store.root / history_name)
        dataset_path = Path(self.config["dataset"]["path"])
        metadata = {
            **previous_metadata,
            "framework_version": __version__,
            "python": platform.python_version(),
            "config_hash": current_hash,
            "execution_fingerprint": current_execution_fingerprint,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": current_dataset_sha256,
            "dataset_samples": len(samples),
            "model": self.config["backend"].get("model"),
            "endpoints": self.config["backend"].get("endpoints"),
            "generation": self.config.get("generation", {}),
            "git_commit": git_commit(),
            "started_at": previous_metadata.get("started_at", started),
            "last_invocation_started_at": started,
        }
        runtime_metadata = self.config.get("runtime_metadata")
        if runtime_metadata:
            metadata["runtime"] = runtime_metadata
        if os.environ.get("EVAL_HARNESS_IMAGE_ID"):
            metadata.setdefault("runtime", {})["image_id"] = os.environ["EVAL_HARNESS_IMAGE_ID"]
        if os.environ.get("EVAL_HARNESS_RUN_ID"):
            metadata.setdefault("runtime", {})["run_id"] = os.environ["EVAL_HARNESS_RUN_ID"]
        self.store.write_json_atomic("metadata.json", metadata)

    def _finalize_metadata(self, started: float, ended: float) -> None:
        path = self.store.root / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update({"started_at": started, "ended_at": ended, "duration_sec": ended - started})
        self.store.write_json_atomic("metadata.json", metadata)

    def _write_state(
        self, execution: str, evaluation: str, metrics: str, report: str
    ) -> None:
        self.store.write_json_atomic(
            "state.json",
            {
                "execution": execution,
                "evaluation": evaluation,
                "metrics": metrics,
                "report": report,
                "updated_at": time.time(),
            },
        )

    def _write_reports(self, metrics: dict[str, Any]) -> None:
        self.store.write_json_atomic("summary.json", metrics)
        name = self.config["experiment"].get("name", self.store.root.name)
        self.store.write_text_atomic("summary.txt", render_text_report(name, metrics))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
