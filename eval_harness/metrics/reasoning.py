from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from eval_harness.core.models import EvaluationResult, TaskSample, Trajectory
from eval_harness.core.registry import METRICS


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pass_at_k(n: int, c: int, k: int) -> float | None:
    if n < k or k < 1:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@dataclass(slots=True)
class MetricContext:
    samples: list[TaskSample]
    trajectories: list[Trajectory]
    evaluations: list[EvaluationResult]
    planned_rollouts: int
    sample_by_id: dict[str, TaskSample] = field(init=False)
    trajectory_by_key: dict[tuple[str, int], Trajectory] = field(init=False)
    evaluation_by_key: dict[tuple[str, int], EvaluationResult] = field(init=False)
    planned_keys: list[tuple[str, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.sample_by_id = {sample.sample_id: sample for sample in self.samples}
        self.trajectory_by_key = {item.key: item for item in self.trajectories}
        self.evaluation_by_key = {item.key: item for item in self.evaluations}
        self.planned_keys = [
            (sample.sample_id, rollout_id)
            for sample in self.samples
            for rollout_id in range(self.planned_rollouts)
        ]


class AccuracyMetric:
    def compute(self, context: MetricContext) -> dict[str, Any]:
        correct = sum(
            1
            for key in context.planned_keys
            if (evaluation := context.evaluation_by_key.get(key)) is not None
            and evaluation.reward == 1.0
        )
        return {"accuracy": correct / len(context.planned_keys) if context.planned_keys else None}


class AvgAtKMetric:
    def compute(self, context: MetricContext) -> dict[str, Any]:
        values = []
        for sample in context.samples:
            correct = sum(
                1
                for rollout_id in range(context.planned_rollouts)
                if (
                    evaluation := context.evaluation_by_key.get((sample.sample_id, rollout_id))
                )
                is not None
                and evaluation.reward == 1.0
            )
            values.append(correct / context.planned_rollouts)
        return {f"avg@{context.planned_rollouts}": _mean(values)}


class PassAtKMetric:
    def __init__(self, config: dict[str, Any]):
        self.ks = [int(k) for k in config.get("k", [1])]

    def compute(self, context: MetricContext) -> dict[str, Any]:
        output = {}
        for k in self.ks:
            values = []
            for sample in context.samples:
                correct = sum(
                    1
                    for rollout_id in range(context.planned_rollouts)
                    if (
                        evaluation := context.evaluation_by_key.get((sample.sample_id, rollout_id))
                    )
                    is not None
                    and evaluation.reward == 1.0
                )
                value = _pass_at_k(context.planned_rollouts, correct, k)
                if value is not None:
                    values.append(value)
            output[f"pass@{k}"] = _mean(values)
        return output


class LengthMetric:
    def compute(self, context: MetricContext) -> dict[str, Any]:
        values = [
            int(item.usage.get("completion_tokens", 0))
            for item in context.trajectories
            if item.status == "completed"
        ]
        return {
            "length": {
                "mean": _mean([float(value) for value in values]),
                "median": statistics.median(values) if values else None,
                "p90": _percentile(values, 0.90),
                "p95": _percentile(values, 0.95),
            }
        }


class TruncationMetric:
    def compute(self, context: MetricContext) -> dict[str, Any]:
        completed = [item for item in context.trajectories if item.status == "completed"]
        truncated = sum(
            1 for item in completed if item.metadata.get("finish_reason") in {"length", "max_tokens"}
        )
        return {"truncation_rate": truncated / len(completed) if completed else None}


class PerSourceMetric:
    def compute(self, context: MetricContext) -> dict[str, Any]:
        values: dict[str, list[float]] = defaultdict(list)
        for sample in context.samples:
            source = sample.source or "unknown"
            for rollout_id in range(context.planned_rollouts):
                evaluation = context.evaluation_by_key.get((sample.sample_id, rollout_id))
                values[source].append(
                    1.0 if evaluation is not None and evaluation.reward == 1.0 else 0.0
                )
        return {"per_source": {name: _mean(scores) for name, scores in sorted(values.items())}}


class CorrectCountDistributionMetric:
    def compute(self, context: MetricContext) -> dict[str, Any]:
        histogram = {str(count): 0 for count in range(context.planned_rollouts + 1)}
        for sample in context.samples:
            correct_count = sum(
                1
                for rollout_id in range(context.planned_rollouts)
                if (
                    evaluation := context.evaluation_by_key.get((sample.sample_id, rollout_id))
                )
                is not None
                and evaluation.reward == 1.0
            )
            histogram[str(correct_count)] += 1
        return {"correct_trajectories_per_problem": histogram}


class CorrectLengthMetric:
    def compute(self, context: MetricContext) -> dict[str, Any]:
        values = []
        for key, trajectory in context.trajectory_by_key.items():
            evaluation = context.evaluation_by_key.get(key)
            if evaluation is not None and evaluation.reward == 1.0:
                values.append(int(trajectory.usage.get("completion_tokens", 0)))
        return {
            "correct_length": {
                "mean": _mean([float(value) for value in values]),
                "median": statistics.median(values) if values else None,
                "p90": _percentile(values, 0.90),
                "p95": _percentile(values, 0.95),
            }
        }


METRICS.register("accuracy", lambda config, **_: AccuracyMetric())
METRICS.register("avg_at_k", lambda config, **_: AvgAtKMetric())
METRICS.register("pass_at_k", lambda config, **_: PassAtKMetric(config))
METRICS.register("length", lambda config, **_: LengthMetric())
METRICS.register("truncation", lambda config, **_: TruncationMetric())
METRICS.register("per_source", lambda config, **_: PerSourceMetric())
METRICS.register(
    "correct_count_distribution", lambda config, **_: CorrectCountDistributionMetric()
)
METRICS.register("correct_length", lambda config, **_: CorrectLengthMetric())


def compute_metrics(
    samples: list[TaskSample],
    trajectories: list[Trajectory],
    evaluations: list[EvaluationResult],
    metric_configs: list[dict[str, Any]],
    planned_rollouts: int,
) -> dict[str, Any]:
    context = MetricContext(samples, trajectories, evaluations, planned_rollouts)
    output: dict[str, Any] = {}
    for config in metric_configs:
        metric = METRICS.create(config)
        values = metric.compute(context)
        overlap = output.keys() & values.keys()
        if overlap:
            raise ValueError(f"duplicate metric output keys: {sorted(overlap)}")
        output.update(values)
    planned_total = len(context.planned_keys)
    completed = [item for item in trajectories if item.status == "completed"]
    stored_failed = [item for item in trajectories if item.status != "completed"]
    resolved = [item for item in evaluations if item.reward is not None]
    correct = [item for item in resolved if item.reward == 1.0]
    llm_judged = [item for item in evaluations if "llm_fallback" in item.details]
    output["counts"] = {
        "samples": len(samples),
        "planned_trajectories": planned_total,
        "stored_trajectories": len(trajectories),
        "generation_completed": len(completed),
        "generation_failed_or_missing": planned_total - len(completed),
        "stored_failed_trajectories": len(stored_failed),
        "judge_resolved": len(resolved),
        "judge_unknown": len(evaluations) - len(resolved),
        "correct": len(correct),
        "llm_judge_used": len(llm_judged),
        "llm_judge_errors": sum(
            1 for item in llm_judged if item.details["llm_fallback"].get("error")
        ),
    }
    output["generation_error_rate"] = (
        (planned_total - len(completed)) / planned_total if planned_total else None
    )
    output["judge_coverage"] = len(resolved) / len(evaluations) if evaluations else None
    return output


__all__ = ["compute_metrics", "_pass_at_k"]
