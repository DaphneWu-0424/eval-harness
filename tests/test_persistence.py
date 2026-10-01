import tempfile
import unittest
from pathlib import Path

from eval_harness.core.models import Trajectory, TrajectoryEvent
from eval_harness.persistence import ExperimentStore


class PersistenceTests(unittest.TestCase):
    def test_round_trip_and_latest_record_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(directory)
            failed = Trajectory("x", 0, [], None, "failed", error="boom")
            completed = Trajectory(
                "x", 0, [TrajectoryEvent("assistant_final", "assistant", "ok")], "ok", "completed"
            )
            store.append_trajectory(failed)
            store.append_trajectory(completed)
            loaded = store.load_trajectories()
            self.assertEqual(loaded[("x", 0)].status, "completed")

    def test_truncated_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(directory)
            store.append_trajectory(Trajectory("x", 0, [], "ok", "completed"))
            with Path(store.raw_path).open("a", encoding="utf-8") as handle:
                handle.write('{"broken":')
            self.assertEqual(len(store.load_trajectories()), 1)
            store.append_trajectory(Trajectory("y", 0, [], "ok", "completed"))
            self.assertEqual(len(store.load_trajectories()), 2)

    def test_legacy_import(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(directory)
            count = store.import_trajectories("tests/fixtures/legacy_raw.jsonl")
            self.assertEqual(count, 2)
            loaded = store.load_trajectories()
            self.assertEqual(loaded[("0", 0)].final_output, r"\boxed{1}")
            self.assertEqual(loaded[("1", 0)].events[0].type, "reasoning")
            self.assertEqual(store.import_trajectories("tests/fixtures/legacy_raw.jsonl"), 0)


if __name__ == "__main__":
    unittest.main()
