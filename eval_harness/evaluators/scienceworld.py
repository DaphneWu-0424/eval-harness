import hashlib
import json
import math

from eval_harness.core.interfaces import Evaluator
from eval_harness.core.models import EvaluationResult
from eval_harness.core.registry import EVALUATORS


class ScienceWorldEvaluator(Evaluator):
    def __init__(self, config, **kwargs):
        fingerprint = hashlib.sha256(json.dumps({"config": config, "version": 1}, sort_keys=True).encode()).hexdigest()[:12]
        self.identity = f"scienceworld:{fingerprint}"

    async def evaluate(self, sample, trajectory):
        raw = trajectory.metadata.get("raw_score")
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = None
        failed = trajectory.status != "completed" or raw is None or not math.isfinite(float(raw)) or float(raw) > 100
        scores = {}
        success = False
        if not failed:
            raw = float(raw)
            success = raw == 100.0
            scores = {"task_score": max(0.0, raw), "raw_score": raw,
                      "retained_score": float(trajectory.metadata.get("retained_score", max(0.0, raw))),
                      "episode_return": float(trajectory.metadata.get("episode_return", 0.0)),
                      "success": float(success)}
        return EvaluationResult(
            sample_id=sample.sample_id, rollout_id=trajectory.rollout_id,
            status="failed" if failed else "completed", reward=None if failed else float(success), scores=scores,
            labels={"success": success, "task_type": sample.metadata.get("task_type", sample.source),
                    "termination": trajectory.metadata.get("termination")},
            details={"error": trajectory.error, "reason": "generation_failed" if trajectory.status != "completed" else "missing_or_invalid_score" if failed else None,
                     "raw_score": raw}, evaluator=self.identity,
        )


EVALUATORS.register("scienceworld", lambda config, **kwargs: ScienceWorldEvaluator(config, **kwargs))
