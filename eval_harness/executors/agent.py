from __future__ import annotations

from eval_harness.core.interfaces import AgentHarness, Environment, Executor, ModelBackend
from eval_harness.core.models import TaskSample, Trajectory


class AgentExecutor(Executor):
    """Reserved first-class Agent executor interface; loop implementation is phase two."""

    def __init__(
        self,
        model_backend: ModelBackend,
        environment: Environment,
        harness: AgentHarness,
        max_steps: int,
    ):
        self.model_backend = model_backend
        self.environment = environment
        self.harness = harness
        self.max_steps = max_steps

    async def run(self, sample: TaskSample, rollout_id: int) -> Trajectory:
        raise NotImplementedError("AgentExecutor execution loop is reserved for phase two")
