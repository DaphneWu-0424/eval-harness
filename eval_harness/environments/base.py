from __future__ import annotations

from typing import Any

from eval_harness.core.interfaces import Environment
from eval_harness.core.models import StepResult, TaskSample


class NullEnvironment(Environment):
    async def reset(self, sample: TaskSample) -> Any:
        return None

    async def step(self, action: Any) -> StepResult:
        return StepResult(observation=None, done=True)

