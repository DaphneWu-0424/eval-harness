import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from eval_harness.runtime import sweep
from eval_harness.runtime.gpu_pool import GpuLease, gpu_status
from eval_harness.runtime.separated import parser, select_datasets


class GpuQueueTests(unittest.TestCase):
    def test_gpu_inventory_detects_even_small_foreign_processes(self):
        with patch("eval_harness.runtime.gpu_pool.subprocess.check_output", side_effect=[
            "0, GPU-a, 100\n1, GPU-b, 0\n", "GPU-a, 1234\n"
        ]):
            status = gpu_status()
        self.assertTrue(status["0"]["has_compute_process"])
        self.assertFalse(status["1"]["has_compute_process"])

    def test_two_submissions_cannot_lease_the_same_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            first = GpuLease(directory, "0", "first")
            second = GpuLease(directory, "0", "second")
            try:
                self.assertTrue(first.acquire())
                self.assertFalse(second.acquire())
                first.release()
                self.assertTrue(second.acquire())
            finally:
                first.release()
                second.release()

    def test_named_dataset_selection(self):
        args = parser().parse_args(["--benchmark", "webshop", "--dataset", "seen=configs/agent/alfworld.yaml"])
        select_datasets(args)
        self.assertEqual(args.benchmarks, ["webshop", "seen"])
        self.assertEqual(args.config_overrides["seen"], "configs/agent/alfworld.yaml")
        args = parser().parse_args(["--benchmark", "alfworld", "--dataset", "alfworld=other.yaml"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_datasets(args)

    def make_args(self, root, *options):
        p = sweep.parser()
        args = p.parse_args(["--run-id", "queue-test", "--results-root", str(root), "--gpus", "0,1"])
        args.parallel, args.max_parallel = True, None
        args.poll_seconds, args.queue_timeout = 0.001, 0.03
        args.lease_dir = str(root / "leases")
        args.benchmarks = ["alfworld", "webshop"]
        args.config_overrides = {}
        return args

    def task(self, benchmark, number=0):
        return dict(checkpoint="same-checkpoint", checkpoint_path="/source/same-checkpoint", step=100,
                    benchmark=benchmark, model_dir="/model", run_id=f"task-{benchmark}-{number}", status="pending")

    def test_checkpoint_times_dataset_are_separate_tasks_and_skip_busy_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            tasks = [self.task("alfworld"), self.task("webshop")]
            assignments = []
            inventory = {"0": {"memory_mib": 50, "has_compute_process": True}, "1": {"memory_mib": 0, "has_compute_process": False}}

            class FakeSuite:
                def __init__(self, args):
                    self.args = args
                    self.artifacts = {}
                def run(self):
                    assignments.append((self.args.gpus, list(self.args.benchmarks)))
                    return {"benchmarks": {self.args.benchmarks[0]: {"metrics": {"accuracy": 0.5}}}}
                def cancel(self):
                    pass

            state = {"tasks": tasks}
            with patch.object(sweep, "gpu_status", return_value=inventory), patch.object(sweep, "AgentSuite", FakeSuite):
                code = sweep.TaskQueue(args, tasks, root, state).run()
            self.assertEqual(code, 0)
            self.assertEqual(assignments, [("1", ["alfworld"]), ("1", ["webshop"])])
            self.assertTrue(all(t["status"] == "completed" for t in tasks))

    def test_idle_gpu_is_rechecked_after_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            args.gpus = "0"
            task = self.task("alfworld")
            idle = {"0": {"memory_mib": 0, "has_compute_process": False}}
            busy = {"0": {"memory_mib": 20, "has_compute_process": True}}
            with patch.object(sweep, "gpu_status", side_effect=[idle, idle, busy]):
                queue = sweep.TaskQueue(args, [task], root, {"tasks": [task]})
                with patch.object(sweep, "ThreadPoolExecutor") as pool:
                    self.assertFalse(queue.assign(pool))
                    pool.submit.assert_not_called()
            self.assertEqual(task["status"], "pending")
            lease = GpuLease(args.lease_dir, "0", "after-recheck")
            try:
                self.assertTrue(lease.acquire())
            finally:
                lease.release()

    def test_queue_timeout_does_not_stop_foreign_workloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            task = self.task("alfworld")
            inventory = {"0": {"memory_mib": 500, "has_compute_process": True}, "1": {"memory_mib": 500, "has_compute_process": True}}
            state = {"tasks": [task]}
            with patch.object(sweep, "gpu_status", return_value=inventory), patch.object(sweep, "AgentSuite") as suite:
                code = sweep.TaskQueue(args, [task], root, state).run()
                suite.assert_not_called()
            self.assertEqual(code, 1)
            self.assertEqual(task["status"], "cancelled")
            self.assertIn("TimeoutError", state["error"])

    def test_task_identity_includes_full_checkpoint_path_and_dataset(self):
        a = sweep.task_id("run", Path("/first/step100"), "alfworld")
        b = sweep.task_id("run", Path("/second/step100"), "alfworld")
        c = sweep.task_id("run", Path("/first/step100"), "webshop")
        self.assertEqual(len({a, b, c}), 3)
        self.assertLessEqual(len(a), 61)
        self.assertNotEqual(sweep.task_id("20261116-120000-aaaa", Path("/model"), "alfworld"),
                            sweep.task_id("20261116-120000-bbbb", Path("/model"), "alfworld"))

    def test_discovery_accepts_named_hf_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "my-export"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(sweep.discover(root), [(None, model.resolve())])

    def test_one_failed_task_does_not_block_the_next_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            args.backend = "fake"
            args.max_parallel = 1
            tasks = [self.task("alfworld"), self.task("webshop")]

            class FakeSuite:
                def __init__(self, args):
                    self.args = args
                def run(self):
                    if self.args.benchmarks == ["alfworld"]:
                        raise RuntimeError("fixture failure")
                    return {"benchmarks": {"webshop": {"metrics": {"accuracy": 0.5}}}}
                def cancel(self):
                    pass

            with patch.object(sweep, "AgentSuite", FakeSuite):
                code = sweep.TaskQueue(args, tasks, root, {"tasks": tasks}).run()
            self.assertEqual(code, 1)
            self.assertEqual([t["status"] for t in tasks], ["failed", "completed"])

    def test_interrupt_cancels_pending_and_releases_active_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            args.max_parallel = 1
            args.gpus = "0"
            tasks = [self.task("alfworld"), self.task("webshop")]
            started, cancelled = threading.Event(), threading.Event()

            class FakeSuite:
                def __init__(self, args):
                    pass
                def run(self):
                    started.set()
                    cancelled.wait(2)
                    raise InterruptedError("cancelled fixture")
                def cancel(self):
                    cancelled.set()

            inventory = {"0": {"memory_mib": 0, "has_compute_process": False}}
            with patch.object(sweep, "gpu_status", return_value=inventory), patch.object(sweep, "AgentSuite", FakeSuite):
                queue = sweep.TaskQueue(args, tasks, root, {"tasks": tasks})
                result = []
                thread = threading.Thread(target=lambda: result.append(queue.run()))
                thread.start()
                self.assertTrue(started.wait(1))
                # wait until the submitted future is registered in the host queue
                import time
                deadline = time.monotonic() + 1
                while not queue.active and time.monotonic() < deadline:
                    time.sleep(0.001)
                queue.request_stop(2)
                thread.join(3)
                self.assertFalse(thread.is_alive())
            self.assertEqual(result, [130])
            self.assertEqual([t["status"] for t in tasks], ["cancelled", "cancelled"])
            lease = GpuLease(args.lease_dir, "0", "next-submission")
            try:
                self.assertTrue(lease.acquire())
            finally:
                lease.release()
