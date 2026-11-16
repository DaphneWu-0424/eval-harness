import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eval_harness.backends.fake import FakeBackend
from eval_harness.core.models import EvaluationResult, Trajectory, TrajectoryEvent
from eval_harness.datasets.appworld import AppWorldManifest
from eval_harness.environments.appworld_worker import main
from eval_harness.evaluators.appworld import AppWorldEvaluator
from eval_harness.harnesses.appworld import AppWorldHarness, parse_code
from eval_harness.harnesses.appworld_prompts import APPWORLD_TEMPLATE_NO_HIS
from eval_harness.metrics import compute_metrics


class AppWorldTests(unittest.IsolatedAsyncioTestCase):
    def samples(self):
        return AppWorldManifest({'path': 'configs/manifests/appworld-test-normal.json'}).load()

    def test_fixed_split_sizes(self):
        self.assertEqual(len(self.samples()), 168)
        self.assertEqual(len(AppWorldManifest({'path': 'configs/manifests/appworld-test-challenge.json'}).load()), 417)

    def test_parser_preserves_python_case_and_training_fallback(self):
        code = '\nToken = "AbC"\nprint(Token)\n'
        self.assertEqual(parse_code('<think>x</think><code>' + code + '</code>'), (code, True))
        self.assertFalse(parse_code('<code>print(1)</code>')[1])
        self.assertEqual(parse_code('x' * 120), ('x' * 100, False))

    def test_prompt_and_code_result_history(self):
        harness = AppWorldHarness({}, FakeBackend({}), {})
        supervisor = dict(first_name='A', last_name='B', email='a@b', phone_number='123')
        obs = dict(supervisor=supervisor, task_description='task', observation='initial')
        trajectory = Trajectory('id', 0, [], None, 'running')
        expected = APPWORLD_TEMPLATE_NO_HIS.format(task_description='task', **{
            'supervisor_' + k: v for k, v in supervisor.items()})
        self.assertEqual(harness.build_prompt(trajectory, obs), expected)
        trajectory.events = [TrajectoryEvent(type='observation', role='environment', content='initial'),
                             TrajectoryEvent(type='action', role='assistant', content='print(123)'),
                             TrajectoryEvent(type='observation', role='environment', content='123')]
        prompt = harness.build_prompt(trajectory, obs)
        self.assertIn('Code 1: \nprint(123)\n\nResult 1: \n123', prompt)
        self.assertIn('Now you are at step 2', prompt)

    async def test_completion_declaration_does_not_override_native_tests(self):
        sample = self.samples()[0]
        trajectory = Trajectory(sample.sample_id, 0, [], None, 'completed', metadata={
            'won': False, 'evaluated': True, 'task_completed': True, 'tests_passed': 1, 'tests_total': 3})
        evaluator = AppWorldEvaluator({})
        result = await evaluator.evaluate(sample, trajectory)
        self.assertEqual(result.reward, 0)
        trajectory.metadata['evaluated'] = False
        self.assertIsNone((await evaluator.evaluate(sample, trajectory)).reward)

    def test_scenario_metric_requires_all_variants(self):
        samples = self.samples()[:3]
        self.assertEqual(len({s.metadata['scenario'] for s in samples}), 1)
        rows = [EvaluationResult(s.sample_id, 0, 'completed', value, {}, {}, {}, 'test')
                for s, value in zip(samples, [1, 1, 0])]
        result = compute_metrics(samples, [], rows, [{'type': 'appworld'}], 1)
        self.assertAlmostEqual(result['appworld_tgc'], 2/3)
        self.assertEqual(result['appworld_sgc'], 0)
        partial = compute_metrics(samples[:2], [], rows[:2], [{'type': 'appworld'}], 1)
        self.assertIsNone(partial['appworld_sgc'])
        rows.pop()
        self.assertIsNone(compute_metrics(samples, [], rows, [{'type': 'appworld'}], 1)['appworld_tgc'])

    def test_worker_evaluates_at_horizon_and_keeps_protocol_clean(self):
        class World:
            def __init__(self, **kwargs):
                self.task = SimpleNamespace(instruction='task', supervisor={})
            def execute(self, code):
                print('SDK diagnostic')
                return 'result'
            def task_completed(self):
                return False
            def evaluate(self, **kwargs):
                return SimpleNamespace(num_tests=3, total_count=3, pass_count=2, success=False)
            def close(self):
                pass
        requests = [{'operation': 'reset', 'task_id': 'id', 'max_steps': 1, 'timeout_seconds': 10, 'seed': 0},
                    {'operation': 'step', 'code': 'print(1)'}, {'operation': 'close'}]
        wire = io.StringIO()
        with patch.dict('sys.modules', {'appworld': SimpleNamespace(AppWorld=World)}), \
             patch('sys.stdin', io.StringIO(''.join(json.dumps(r)+'\n' for r in requests))), \
             patch('sys.stdout', wire), patch('sys.stderr', io.StringIO()):
            main()
        rows = [json.loads(line) for line in wire.getvalue().splitlines()]
        result = rows[1]['result']
        self.assertTrue(result['done'])
        self.assertFalse(result['won'])
        self.assertTrue(result['trajectory_metadata']['evaluated'])
        self.assertEqual(result['trajectory_metadata']['termination'], 'step_limit')


if __name__ == '__main__':
    unittest.main()
