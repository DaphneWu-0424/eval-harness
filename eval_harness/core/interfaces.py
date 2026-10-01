from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import EvaluationResult, ModelResponse, StepResult, TaskSample, Trajectory


class Dataset(ABC):
    @abstractmethod
    def load(self) -> list[TaskSample]: ...


class PromptBuilder(ABC):
    @abstractmethod
    def build(self, sample: TaskSample) -> list[dict[str, Any]]: ...


class ModelBackend(ABC):
    @abstractmethod
    async def generate(
        self, messages: list[dict[str, Any]], generation_config: dict[str, Any]
    ) -> ModelResponse: ...

    async def health_check(self) -> None:
        return None

    async def close(self) -> None:
        return None


class Executor(ABC):
    @abstractmethod
    async def run(self, sample: TaskSample, rollout_id: int) -> Trajectory: ...


class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self, sample: TaskSample, trajectory: Trajectory) -> EvaluationResult: ...


class Environment(ABC):
    @abstractmethod
    async def reset(self, sample: TaskSample) -> Any: ...

    @abstractmethod
    async def step(self, action: Any) -> StepResult: ...

    async def close(self) -> None:
        return None


class AgentHarness(ABC):
    async def initialize(self, sample: TaskSample) -> None:
        return None

    @abstractmethod
    async def next_action(self, trajectory: Trajectory, observation: Any) -> Any: ...

    async def finalize(self, trajectory: Trajectory) -> None:
        return None

