from collections import defaultdict
from eval_harness.core.registry import METRICS


class AppWorldMetrics:
    def compute(self, context):
        groups = defaultdict(list)
        resolved = []
        for sample in context.samples:
            for rollout in range(context.planned_rollouts):
                evaluation = context.evaluation_by_key.get((sample.sample_id, rollout))
                value = evaluation.reward if evaluation is not None and evaluation.status == 'completed' else None
                groups[(sample.metadata['scenario'], rollout)].append((sample, value))
                resolved.append(value)
        complete = bool(resolved) and all(v is not None for v in resolved)
        tgc = sum(resolved)/len(resolved) if complete else None
        scenarios = []
        for rows in groups.values():
            expected = rows[0][0].metadata['scenario_tasks']
            if len(rows) != expected or any(value is None for _, value in rows):
                scenarios.append(None)
            else:
                scenarios.append(float(all(value == 1 for _, value in rows)))
        sgc = sum(scenarios)/len(scenarios) if scenarios and all(v is not None for v in scenarios) else None
        expected_size = context.samples[0].metadata['manifest_suite_size'] if context.samples else 0
        return {'accuracy': tgc, 'appworld_tgc': tgc, 'appworld_sgc': sgc,
                'appworld_full_panel': len(context.samples) == expected_size,
                'appworld_expected_tasks': expected_size}


METRICS.register('appworld', lambda config, **_: AppWorldMetrics())
