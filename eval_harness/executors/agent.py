from __future__ import annotations

import time
from typing import Any

from eval_harness.core.interfaces import AgentHarness, Environment, Executor, ModelBackend
from eval_harness.core.models import TaskSample, Trajectory, TrajectoryEvent
from eval_harness.core.registry import EXECUTORS


class AgentExecutor(Executor):
    """Multi-turn agent loop.

    ``reset -> (generate -> act -> step -> observe) x max_steps -> trajectory``.
    Records every model output, action, and environment observation as events.
    Distinguishes: model/infra failure (status="failed") from a completed
    episode that simply was not won (status="completed", won=False).
    """

    def __init__(
        self,
        model_backend: ModelBackend,
        environment: Environment,
        harness: AgentHarness,
        generation_config: dict[str, Any],
        max_steps: int = 50,
        base_seed: int = 0,
    ):
        self.model_backend = model_backend
        self.environment = environment
        self.harness = harness
        self.generation_config = dict(generation_config)
        self.max_steps = max_steps
        self.base_seed = base_seed

    async def run(self, sample: TaskSample, rollout_id: int) -> Trajectory:
        started = time.time()
        events: list[TrajectoryEvent] = []
        trajectory = Trajectory(
            sample_id=sample.sample_id,
            rollout_id=rollout_id,
            events=events,
            final_output=None,
            status="running",
        )
        won = False
        episode_length = 0
        invalid_count = 0
        error: str | None = None
        try:
            obs_ctx = await self.environment.reset(sample)
            events.append(
                TrajectoryEvent(type="observation", role="environment", content=obs_ctx["observation"])
            )
            await self.harness.initialize(sample)
            for step in range(self.max_steps):
                result = await self.harness.next_action(trajectory, obs_ctx)
                if result.get("reasoning"):
                    events.append(
                        TrajectoryEvent(type="reasoning", role="assistant", content=result["reasoning"])
                    )
                if result.get("content") is not None:
                    events.append(
                        TrajectoryEvent(type="assistant_final", role="assistant", content=result["content"])
                    )
                events.append(
                    TrajectoryEvent(
                        type="action",
                        role="assistant",
                        content=result["action"],
                        metadata={"is_action_valid": result["is_action_valid"]},
                    )
                )
                trajectory.usage = result.get("usage") or trajectory.usage
                if not result["is_action_valid"]:
                    invalid_count += 1
                step_result = await self.environment.step(result["action"])
                events.append(
                    TrajectoryEvent(type="observation", role="environment", content=step_result.observation)
                )
                obs_ctx = {
                    "observation": step_result.observation,
                    "admissible_commands": step_result.metadata.get("admissible_commands", []),
                    "won": step_result.metadata.get("won", False),
                }
                won = bool(step_result.metadata.get("won", False))
                episode_length = step + 1
                if step_result.done:
                    break
            trajectory.status = "completed"
            trajectory.final_output = {
                "won": won,
                "episode_length": episode_length,
                "invalid_action_count": invalid_count,
            }
        except Exception as exc:
            trajectory.status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            trajectory.final_output = {
                "won": False,
                "episode_length": episode_length,
                "invalid_action_count": invalid_count,
            }
        finally:
            try:
                await self.environment.close()
            except Exception:
                pass

        trajectory.error = error
        trajectory.timing = {"started_at": started, "ended_at": time.time()}
        trajectory.metadata = {
            "won": won,
            "episode_length": episode_length,
            "invalid_action_count": invalid_count,
            "error_type": None if trajectory.status == "completed" else (error or "").split(":")[0],
        }
        return trajectory


EXECUTORS.register(
    "agent",
    lambda config, **kwargs: AgentExecutor(
        kwargs["model_backend"],
        kwargs["environment"],
        kwargs["harness"],
        kwargs.get("generation_config", {}),
        kwargs.get("max_steps", 50),
        kwargs.get("base_seed", 0),
    ),
)
