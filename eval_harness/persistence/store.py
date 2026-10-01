from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Generic, TypeVar

from eval_harness.core.models import EvaluationResult, Trajectory, TrajectoryEvent

T = TypeVar("T", Trajectory, EvaluationResult)


class JsonlCorruptionError(RuntimeError):
    pass


class ExperimentStore:
    def __init__(self, output_dir: str | Path):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.raw_path = self.root / "raw.jsonl"
        self.evaluated_path = self.root / "evaluated.jsonl"

    def load_trajectories(self) -> dict[tuple[str, int], Trajectory]:
        items = self._load_jsonl(self.raw_path, Trajectory.from_dict)
        return {item.key: item for item in items}

    def load_evaluations(self) -> dict[tuple[str, int], EvaluationResult]:
        items = self._load_jsonl(self.evaluated_path, EvaluationResult.from_dict)
        return {item.key: item for item in items}

    @staticmethod
    def _load_jsonl(path: Path, parser: Any) -> list[Any]:
        if not path.exists():
            return []
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        result: list[Any] = []
        valid_bytes = 0
        for index, line in enumerate(lines):
            if not line.strip():
                valid_bytes += len(line)
                continue
            try:
                result.append(parser(json.loads(line.decode("utf-8"))))
                valid_bytes += len(line)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                is_last = index == len(lines) - 1
                if is_last:
                    # A process may die halfway through its final append. Repair only that tail.
                    with path.open("r+b") as handle:
                        handle.truncate(valid_bytes)
                    break
                raise JsonlCorruptionError(f"corrupt JSONL record {path}:{index + 1}: {exc}") from exc
        return result

    @staticmethod
    def _append(path: Path, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append_trajectory(self, trajectory: Trajectory) -> None:
        self._append(self.raw_path, trajectory.to_dict())

    def append_evaluation(self, evaluation: EvaluationResult) -> None:
        self._append(self.evaluated_path, evaluation.to_dict())

    def import_trajectories(self, input_path: str | Path) -> int:
        path = Path(input_path)
        existing = self.load_trajectories()
        imported = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    trajectory = self._adapt_trajectory(record, line_number)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise JsonlCorruptionError(f"invalid import record {path}:{line_number}: {exc}") from exc
                previous = existing.get(trajectory.key)
                if previous is not None and previous.status == "completed":
                    continue
                self.append_trajectory(trajectory)
                existing[trajectory.key] = trajectory
                imported += 1
        return imported

    @staticmethod
    def _adapt_trajectory(record: dict[str, Any], line_number: int) -> Trajectory:
        if "events" in record and "sample_id" in record:
            return Trajectory.from_dict(record)
        sample_id = record.get("sample_id", record.get("id", record.get("problem_id")))
        if sample_id is None:
            raise ValueError("legacy record has no sample_id/id/problem_id")
        rollout_id = int(record.get("rollout_id", record.get("rollout", 0)))
        final_output = record.get(
            "final_output", record.get("answer", record.get("response", record.get("output")))
        )
        reasoning = record.get("reasoning", record.get("reasoning_content"))
        events: list[TrajectoryEvent] = []
        if reasoning is not None:
            events.append(TrajectoryEvent("reasoning", "assistant", reasoning))
        if final_output is not None:
            events.append(TrajectoryEvent("assistant_final", "assistant", final_output))
        status = record.get("status", "completed" if final_output is not None else "failed")
        return Trajectory(
            sample_id=str(sample_id),
            rollout_id=rollout_id,
            events=events,
            final_output=final_output,
            status=status,
            usage=record.get("usage")
            or {
                key: record[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if key in record
            },
            timing=record.get("timing")
            or {
                key: record[key]
                for key in ("latency_sec", "timestamp")
                if key in record
            },
            metadata={
                **(record.get("metadata") or {}),
                **{
                    key: record[key]
                    for key in (
                        "finish_reason",
                        "max_tokens",
                        "server",
                        "teacher",
                        "temperature",
                        "top_k",
                        "top_p",
                    )
                    if key in record
                },
                "imported_legacy_line": line_number,
            },
            error=record.get("error"),
        )

    def write_json_atomic(self, name: str, value: Any) -> None:
        path = self.root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def write_text_atomic(self, name: str, value: str) -> None:
        path = self.root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
