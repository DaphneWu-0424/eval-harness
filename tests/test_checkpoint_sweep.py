import json
import tempfile
import unittest
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval_harness.runtime.sweep import discover, prepare_model


class CheckpointSweepTests(unittest.TestCase):
    def test_discovery_is_numeric_and_filterable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["global_step_100", "global_step_2", "global_step_20", "unrelated"]:
                (root / name).mkdir()
            best = root / "best_checkpoint"
            best.mkdir()
            (best / "best_checkpoint_info.json").write_text(json.dumps({"global_step": 10}))
            self.assertEqual([s for s, p in discover(root)], [2, 10, 20, 100])
            self.assertEqual([s for s, p in discover(root, {20})], [20])

    def test_existing_hf_export_is_read_without_merging(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(prepare_model(model, None), model)

    def test_unowned_merge_target_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "global_step_20"
            (checkpoint / "actor").mkdir(parents=True)
            (checkpoint / "actor/shard.pt").write_bytes(b"fixture")
            args = SimpleNamespace(base_model=str(root / "base"), merge_root=str(root / "merged"))
            # Build exactly the source signature used by the cache, then insert an unowned directory.
            import hashlib
            shard = checkpoint / "actor/shard.pt"
            source = {"checkpoint": str(checkpoint.resolve()), "base_model": str(Path(args.base_model).resolve()), "files": [("shard.pt", shard.stat().st_size, shard.stat().st_mtime_ns)]}
            signature = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
            target = Path(args.merge_root) / f"global_step_20-{signature[:12]}"
            target.mkdir(parents=True)
            marker = target / "keep.txt"
            marker.write_text("previous work")
            with self.assertRaisesRegex(ValueError, "unowned"):
                prepare_model(checkpoint, args)
            self.assertEqual(marker.read_text(), "previous work")

    def test_queue_reuses_first_free_gpu_instead_of_static_assignment(self):
        from eval_harness.runtime import sweep
        active, failures, assignments = set(), [], {}
        lock = threading.Lock()

        class FakeSuite:
            def __init__(self, args):
                self.args = args
                self.artifacts = {"benchmarks": {}}

            def run(self):
                with lock:
                    if self.args.gpus in active:
                        failures.append(self.args.gpus)
                    active.add(self.args.gpus)
                    name = Path(self.args.model_dir).name
                    assignments[name] = self.args.gpus
                time.sleep(0.08 if name == "global_step_1" else 0.005)
                with lock:
                    active.remove(self.args.gpus)
                return {"benchmarks": {"alfworld": {"status": "completed", "metrics": {"accuracy": 0.5}}}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in [1, 2, 3, 4, 5]:
                path = root / "checkpoints" / f"global_step_{step}"
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}")
                (path / "model.safetensors").write_bytes(b"fixture")
            with patch.object(sweep, "AgentSuite", FakeSuite), patch.object(sweep.signal, "signal"):
                code = sweep.main(["--checkpoint-dir", str(root / "checkpoints"), "--results-root", str(root / "results"), "--run-id", "parallel-test", "--parallel", "--gpus", "0,1", "--backend", "fake", "--poll-seconds", "0.001", "--benchmark", "alfworld"])
            self.assertEqual(code, 0)
            self.assertEqual(failures, [])
            self.assertEqual(assignments["global_step_1"], "0")
            self.assertEqual(assignments["global_step_2"], "1")
            self.assertEqual(assignments["global_step_3"], "1")
            self.assertEqual(len(assignments), 5)
            self.assertTrue((root / "results/parallel-test-sweep/matrix.csv").is_file())


if __name__ == "__main__":
    unittest.main()
