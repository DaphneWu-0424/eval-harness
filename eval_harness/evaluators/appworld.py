from eval_harness.core.registry import EVALUATORS
from .agent_success import AgentSuccessEvaluator


class AppWorldEvaluator(AgentSuccessEvaluator):
    async def evaluate(self, sample, trajectory):
        result = await super().evaluate(sample, trajectory)
        if trajectory.status == 'completed' and not trajectory.metadata.get('evaluated'):
            result.status = 'failed'
            result.reward = None
            result.scores = {}
            result.labels['success'] = False
            result.details['reason'] = 'missing_native_evaluation'
        result.details['native_tests'] = {key: trajectory.metadata.get(key)
                                         for key in ('tests_passed', 'tests_total', 'termination')}
        return result


EVALUATORS.register('appworld', lambda config, **kwargs: AppWorldEvaluator(config, **kwargs))
