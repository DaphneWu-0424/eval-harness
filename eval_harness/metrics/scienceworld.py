from collections import defaultdict

from eval_harness.core.registry import METRICS


class ScienceWorldMetrics:
    def compute(self, context):
        groups = defaultdict(list)
        planned = defaultdict(int)
        for sample in context.samples:
            task_type = sample.metadata["task_type"]
            for rollout_id in range(context.planned_rollouts):
                planned[task_type] += 1
                evaluation = context.evaluation_by_key.get((sample.sample_id, rollout_id))
                if evaluation is not None and evaluation.status == "completed" and "task_score" in evaluation.scores:
                    groups[task_type].append(evaluation)
        per_type = {}
        for task_type, count in sorted(planned.items()):
            rows = groups[task_type]
            complete = len(rows) == count
            per_type[task_type] = {"planned": count, "resolved": len(rows),
                "score": sum(r.scores["task_score"] for r in rows) / count if complete else None,
                "success_rate": sum(r.scores["success"] for r in rows) / count if complete else None}
        complete = all(v["score"] is not None for v in per_type.values()) and bool(per_type)
        score = sum(v["score"] for v in per_type.values()) / len(per_type) if complete else None
        success = sum(v["success_rate"] for v in per_type.values()) / len(per_type) if complete else None
        expected_size = context.samples[0].metadata.get("manifest_suite_size", len(context.samples)) if context.samples else 0
        expected_types = context.samples[0].metadata.get("manifest_task_types", len(per_type)) if context.samples else 0
        return {"accuracy": success, "scienceworld_score": score, "scienceworld_success_rate": success,
                "scienceworld_score_aggregation": "macro_over_task_types", "scienceworld_per_type": per_type,
                "scienceworld_panel": {"episodes": len(context.samples), "task_types": len(per_type),
                    "expected_episodes": expected_size, "expected_task_types": expected_types,
                    "full_panel": len(context.samples) == expected_size and len(per_type) == expected_types,
                    "all_resolved": complete}}


METRICS.register("scienceworld", lambda config, **_: ScienceWorldMetrics())
