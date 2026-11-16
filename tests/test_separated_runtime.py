import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval_harness.runtime.separated import AgentSuite, LABEL, gpu_ids, parser, resolved_config


class SeparatedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = parser().parse_args([
            "--run-id", "unit-separated", "--model-dir", str(root / "model"),
            "--work-dir", str(root / "work"), "--datasets-dir", str(root / "datasets"),
            "--env-package-dir", str(root / "package"), "--results-root", str(root / "results"),
            "--limit", "2", "--max-steps", "3",
        ])
        self.args.benchmarks = ["alfworld", "webshop"]
        for name in ["model", "package", "work", "datasets/alfworld", "datasets/webshop", "work/webshop/search_engine/indexes_1k", "work/webshop/data"]:
            (root / name).mkdir(parents=True, exist_ok=True)
        (root / "model/config.json").write_text("{}")
        for name in ["alfworld_games_128.jsonl", "webshop_sessions_500.jsonl"]:
            (root / "work" / name).write_text("{}\n")
        self.suite = AgentSuite(self.args)
        self.calls = []
        self.fail_environment = False
        self.foreign_container = False
        self.busy_gpu = False

    def process(self, command, **kwargs):
        self.calls.append(command)
        out, status = "", 0
        if command[0] == "nvidia-smi":
            out = f"0, {2000 if self.busy_gpu else 0}\n"
        elif command[1:3] == ["image", "inspect"]:
            out = "sha256:unit-image\n"
        elif command[1] == "inspect":
            if "{{.State.Running}}" in command:
                out = "true\n"
            elif "{{json .Config.Labels}}" in command:
                out = json.dumps({LABEL: "foreign" if self.foreign_container else self.args.run_id})
            else:
                out = "[]"
        elif command[1] == "wait":
            name = command[-1].split("-")[-1]
            target = self.suite.root / name
            target.mkdir(exist_ok=True)
            (target / "summary.json").write_text(json.dumps({"accuracy": 0.0, "counts": {"generation_failed_or_missing": 0}}))
            out = "7\n" if self.fail_environment else "0\n"
        elif command[1] == "logs":
            out = "container output\n"
        return subprocess.CompletedProcess(command, status, out, "")

    def test_shared_model_and_gpu_free_environments(self):
        with patch("eval_harness.runtime.separated.subprocess.run", side_effect=self.process):
            report = self.suite.run()
        runs = [c for c in self.calls if c[:2] == ["docker", "run"]]
        self.assertEqual(len(runs), 3)
        self.assertIn("--gpus", runs[0])
        self.assertIn("0.0.0.0", runs[0])
        for c in runs[1:]:
            self.assertNotIn("--gpus", c)
            self.assertIn("NVIDIA_VISIBLE_DEVICES=void", c)
            self.assertFalse(any("dst=/workspace/model" in token for token in c))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["cleanup"], "completed")
        self.assertEqual(len([c for c in self.calls if c[:3] == ["docker", "rm", "-f"]]), 3)
        self.assertTrue((self.suite.root / "logs/eh-unit-separated-inference.log").exists())

    def test_environment_failure_is_reported_and_resources_cleaned(self):
        self.fail_environment = True
        with patch("eval_harness.runtime.separated.subprocess.run", side_effect=self.process):
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                self.suite.run()
        report = json.loads((self.suite.root / "suite.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cleanup"], "completed")
        self.assertEqual(len([c for c in self.calls if c[:3] == ["docker", "rm", "-f"]]), 2)

    def test_busy_gpu_fails_before_resource_creation(self):
        self.busy_gpu = True
        with patch("eval_harness.runtime.separated.subprocess.run", side_effect=self.process):
            with self.assertRaisesRegex(ValueError, "busy"):
                self.suite.run()
        self.assertFalse(any(c[1] == "run" or c[1:3] == ["network", "create"] for c in self.calls if c[0] == "docker"))

    def test_foreign_resources_are_not_removed(self):
        self.foreign_container = True
        with patch("eval_harness.runtime.separated.subprocess.run", side_effect=self.process):
            with self.assertRaisesRegex(RuntimeError, "ownership mismatch"):
                self.suite.run()
        self.assertFalse(any(c[:3] == ["docker", "rm", "-f"] for c in self.calls))

    def test_existing_result_directory_is_never_overwritten(self):
        self.suite.root.mkdir(parents=True)
        marker = self.suite.root / "existing.txt"
        marker.write_text("previous experiment")
        with patch("eval_harness.runtime.separated.subprocess.run", side_effect=self.process):
            with self.assertRaises(FileExistsError):
                self.suite.run()
        self.assertEqual(marker.read_text(), "previous experiment")

    def test_configuration_preserves_protocol_and_changes_connection(self):
        base = {"experiment": {"name": "a", "output_dir": "/old"}, "dataset": {}, "environment": {"max_steps": 50}, "generation": {"temperature": 0, "max_tokens": 512}, "backend": {"endpoint_env": "OLD"}}
        original = copy.deepcopy(base)
        result = resolved_config(base, "alfworld", self.args, "http://inference:31000/v1")
        self.assertEqual(base, original)
        self.assertEqual(result["backend"]["endpoints"], ["http://inference:31000/v1"])
        self.assertNotIn("endpoint_env", result["backend"])
        self.assertEqual(result["generation"]["max_tokens"], 512)
        self.assertEqual(result["generation"]["repetition_penalty"], 1.0)

    def test_fake_mode_never_requests_gpu_or_starts_model(self):
        self.args.backend = "fake"
        self.args.model_dir = None
        with patch("eval_harness.runtime.separated.subprocess.run", side_effect=self.process):
            self.suite.run()
        runs = [c for c in self.calls if c[:2] == ["docker", "run"]]
        self.assertEqual(len(runs), 2)
        self.assertFalse(any("--gpus" in c for c in runs))

    def test_gpu_id_validation(self):
        self.assertEqual(gpu_ids("0,3"), ["0", "3"])
        for invalid in ["", "0,0", "all", "-1", "0, 1"]:
            with self.assertRaises(ValueError):
                gpu_ids(invalid)

    def test_external_endpoint_does_not_start_or_inspect_a_gpu(self):
        self.args.endpoint = "http://model-service:31000/v1"
        self.args.model_dir = None
        with patch("eval_harness.runtime.separated.subprocess.run", side_effect=self.process):
            self.suite.run()
        self.assertFalse(any(c[0] == "nvidia-smi" for c in self.calls))
        self.assertEqual(len([c for c in self.calls if c[:2] == ["docker", "run"]]), 2)

    def test_cancellation_respects_resource_ownership(self):
        self.suite.containers.append("foreign-container")
        self.foreign_container = True
        with patch("eval_harness.runtime.separated.subprocess.run", side_effect=self.process):
            self.suite.cancel()
        self.assertTrue(self.suite.cancelled.is_set())
        self.assertFalse(any(c[:2] == ["docker", "stop"] for c in self.calls))


if __name__ == "__main__":
    unittest.main()
