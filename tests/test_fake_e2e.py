import asyncio
import copy
import tempfile
import unittest
from pathlib import Path

from eval_harness.config import load_config
from eval_harness.pipeline import EvaluationPipeline


class FakeE2ETests(unittest.TestCase):
    def test_pipeline_and_resume(self):
        config = load_config("configs/reasoning/fake_e2e.yaml")
        with tempfile.TemporaryDirectory() as directory:
            config["experiment"]["output_dir"] = directory
            # Hash is recalculated in production before overrides; remove it for this test fixture.
            config["_resolved"]["config_hash"] = None
            first = asyncio.run(EvaluationPipeline(config).run())
            self.assertEqual(first["counts"]["generation_completed"], 3)
            self.assertEqual(first["accuracy"], 1.0)
            raw = Path(directory, "raw.jsonl")
            first_lines = raw.read_text(encoding="utf-8").splitlines()
            second = asyncio.run(EvaluationPipeline(config).run())
            second_lines = raw.read_text(encoding="utf-8").splitlines()
            self.assertEqual(first_lines, second_lines)
            self.assertEqual(second["counts"]["stored_trajectories"], 3)
            self.assertTrue(Path(directory, "summary.json").exists())
            self.assertTrue(Path(directory, "summary.txt").exists())

            changed = copy.deepcopy(config)
            changed["evaluators"][0]["name"] = "math_v2"
            changed["_resolved"]["config_hash"] = "judge-only-change"
            asyncio.run(EvaluationPipeline(changed).run(stage="judge"))
            evaluated_lines = Path(directory, "evaluated.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(evaluated_lines), 6)
            self.assertEqual(len(raw.read_text(encoding="utf-8").splitlines()), 3)
            self.assertTrue(Path(directory, "config.judge-only-c.yaml").exists())


if __name__ == "__main__":
    unittest.main()
