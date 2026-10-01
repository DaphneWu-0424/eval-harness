from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass(slots=True)
class TaskSample:
    sample_id: str
    input: Any
    reference: Any | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryEvent:
    type: str
    role: str | None
    content: Any
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    sample_id: str
    rollout_id: int
    events: list[TrajectoryEvent]
    final_output: Any | None
    status: str
    usage: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.sample_id, self.rollout_id

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Trajectory":
        version = value.get("schema_version", 1)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema_version={version}")
        payload = {k: v for k, v in value.items() if k != "schema_version"}
        payload["events"] = [TrajectoryEvent(**event) for event in payload.get("events", [])]
        return cls(**payload)


@dataclass(slots=True)
class ModelResponse:
    reasoning: str | None
    content: str | None
    tool_calls: list[Any] | None
    finish_reason: str | None
    usage: dict[str, Any]
    raw_response: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    sample_id: str
    rollout_id: int
    status: str
    reward: float | None
    scores: dict[str, float]
    labels: dict[str, Any]
    details: dict[str, Any]
    evaluator: str

    @property
    def key(self) -> tuple[str, int]:
        return self.sample_id, self.rollout_id

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationResult":
        version = value.get("schema_version", 1)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema_version={version}")
        return cls(**{k: v for k, v in value.items() if k != "schema_version"})


@dataclass(slots=True)
class ExtractResult:
    value: str | None
    method: str
    confidence: float
    error: str | None = None


@dataclass(slots=True)
class MetricResult:
    name: str
    value: Any
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    observation: Any
    reward: float | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

