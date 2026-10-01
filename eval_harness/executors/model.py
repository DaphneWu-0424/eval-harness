from __future__ import annotations

import hashlib
import time
from typing import Any

from eval_harness.core.interfaces import Executor, ModelBackend, PromptBuilder
from eval_harness.core.models import TaskSample, Trajectory, TrajectoryEvent
from eval_harness.core.registry import EXECUTORS


class ModelExecutor(Executor):
    def __init__(
        self,
        backend: ModelBackend,
        prompt_builder: PromptBuilder,
        generation_config: dict[str, Any],
        base_seed: int = 0,
    ):
        self.backend = backend
        self.prompt_builder = prompt_builder
        self.generation_config = generation_config
        self.base_seed = base_seed

    def _request_config(self, sample: TaskSample, rollout_id: int) -> dict[str, Any]:
        config = dict(self.generation_config)
        seed_material = f"{self.base_seed}:{sample.sample_id}:{rollout_id}".encode("utf-8")
        config["seed"] = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
        return config

    async def run(self, sample: TaskSample, rollout_id: int) -> Trajectory:
        started = time.time()
        messages = self.prompt_builder.build(sample)
        events = [
            TrajectoryEvent(type="message", role=m.get("role"), content=m.get("content"))
            for m in messages
        ]
        try:
            response = await self.backend.generate(
                messages, self._request_config(sample, rollout_id)
            )
            if response.reasoning:
                events.append(
                    TrajectoryEvent(type="reasoning", role="assistant", content=response.reasoning)
                )
            if response.content is not None:
                events.append(
                    TrajectoryEvent(type="assistant_final", role="assistant", content=response.content)
                )
            if response.tool_calls:
                events.append(
                    TrajectoryEvent(type="tool_calls", role="assistant", content=response.tool_calls)
                )
            return Trajectory(
                sample_id=sample.sample_id,
                rollout_id=rollout_id,
                events=events,
                final_output=response.content,
                status="completed",
                usage=response.usage,
                timing={"started_at": started, "ended_at": time.time()},
                metadata={
                    **response.metadata,
                    "finish_reason": response.finish_reason,
                    "raw_response": response.raw_response,
                },
            )
        except Exception as exc:
            return Trajectory(
                sample_id=sample.sample_id,
                rollout_id=rollout_id,
                events=events,
                final_output=None,
                status="failed",
                timing={"started_at": started, "ended_at": time.time()},
                metadata={
                    "error_type": type(exc).__name__,
                    **getattr(exc, "metadata", {}),
                },
                error=str(exc),
            )


EXECUTORS.register(
    "model",
    lambda config, **kwargs: ModelExecutor(
        kwargs["backend"],
        kwargs["prompt_builder"],
        kwargs["generation_config"],
        kwargs.get("base_seed", 0),
    ),
)
