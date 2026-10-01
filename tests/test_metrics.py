import unittest

from eval_harness.core.models import EvaluationResult, TaskSample, Trajectory
from eval_harness.metrics.reasoning import _pass_at_k, compute_metrics


class MetricsTests(unittest.TestCase):
    def test_pass_at_k(self):
        self.assertAlmostEqual(_pass_at_k(8, 2, 1), 0.25)
        self.assertAlmostEqual(_pass_at_k(8, 2, 8), 1.0)

    def test_invalid_k(self):
        self.assertIsNone(_pass_at_k(2, 1, 4))

    def test_teacher_distribution_and_correct_length(self):
        samples = [TaskSample("a", "x"), TaskSample("b", "y")]
        trajectories = [
            Trajectory(sample, rollout, [], "x", "completed", {"completion_tokens": 10 + rollout})
            for sample in ("a", "b")
            for rollout in range(2)
        ]
        evaluations = [
            EvaluationResult("a", 0, "completed", 1.0, {}, {}, {}, "e"),
            EvaluationResult("a", 1, "completed", 1.0, {}, {}, {}, "e"),
            EvaluationResult("b", 0, "completed", 0.0, {}, {}, {}, "e"),
            EvaluationResult("b", 1, "completed", None, {}, {}, {}, "e"),
        ]
        result = compute_metrics(
            samples,
            trajectories,
            evaluations,
            [{"type": "correct_count_distribution"}, {"type": "correct_length"}],
            2,
        )
        self.assertEqual(result["correct_trajectories_per_problem"], {"0": 1, "1": 0, "2": 1})
        self.assertEqual(result["correct_length"]["mean"], 10.5)


if __name__ == "__main__":
    unittest.main()
