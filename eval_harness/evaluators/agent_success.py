from __future__ import annotations

import hashlib
import json
from typing import Any

from eval_harness.core.interfaces import Evaluator
from eval_harness.core.models import EvaluationResult, TaskSample, Trajectory
from eval_harness.core.registry import EVALUATORS

IMPLEMENTATION_VERSION = 1


class AgentSuccessEvaluator(Evaluator):
    """Maps an agent trajectory to a success verdict.

    - completed + won      -> reward 1.0, success True
    - completed + not won  -> reward 0.0, success False (task genuinely failed)
    - failed (infra/model) -> reward None (NOT counted as task failure)
    """

    def __init__(self, config: dict[str, Any], **kwargs):
        self.name = config.get("name", "agent_success_v1")
        encoded = json.dumps(
            {"config": config, "implementation_version": IMPLEMENTATION_VERSION},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.identity = f"{self.name}:{self.fingerprint[:12]}"

    async def evaluate(self, sample: TaskSample, trajectory: Trajectory) -> EvaluationResult:
        won = bool(trajectory.metadata.get("won", False))
        episode_length = int(trajectory.metadata.get("episode_length", 0))
        invalid_count = int(trajectory.metadata.get("invalid_action_count", 0))
        if trajectory.status == "failed":
            reward = None
            won = False
        else:
            reward = 1.0 if won else 0.0
        return EvaluationResult(
            sample_id=trajectory.sample_id,
            rollout_id=trajectory.rollout_id,
            status=trajectory.status if trajectory.status == "failed" else "completed",
            reward=reward,
            scores={"success": reward} if reward is not None else {},
            labels={
                "success": won,
                "episode_length": episode_length,
                "invalid_action_count": invalid_count,
            },
            details={
                "error": trajectory.error,
                "error_type": trajectory.metadata.get("error_type"),
            },
            evaluator=self.identity,
        )


EVALUATORS.register("agent_success", lambda config, **kwargs: AgentSuccessEvaluator(config, **kwargs))
