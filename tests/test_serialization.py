import unittest

from eval_harness.core.models import EvaluationResult, Trajectory, TrajectoryEvent


class SerializationTests(unittest.TestCase):
    def test_trajectory_round_trip(self):
        original = Trajectory(
            sample_id="sample",
            rollout_id=2,
            events=[TrajectoryEvent("tool_result", "tool", {"ok": True})],
            final_output={"answer": 1},
            status="completed",
            usage={"total_tokens": 5},
        )
        restored = Trajectory.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_evaluation_round_trip(self):
        original = EvaluationResult(
            "sample", 2, "completed", 1.0, {"correct": 1.0}, {"verdict": "CORRECT"}, {}, "math"
        )
        self.assertEqual(EvaluationResult.from_dict(original.to_dict()), original)

    def test_unknown_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            Trajectory.from_dict({"schema_version": 999})


if __name__ == "__main__":
    unittest.main()
