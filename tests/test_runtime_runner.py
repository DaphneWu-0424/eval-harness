import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "runtime runner is a remote Linux component")
class RuntimeRunnerTests(unittest.TestCase):
    def runtime_environment(self, root: Path, pipeline_status: int):
        repository = Path(__file__).resolve().parents[1]
        bin_dir = root / "bin"
        project = root / "project"
        model = root / "model"
        datasets = root / "datasets"
        results = root / "results"
        bin_dir.mkdir()
        model.mkdir()
        datasets.mkdir()
        results.mkdir()
        (project / "configs").mkdir(parents=True)
        (project / "configs" / "smoke.yaml").write_text("fake: true\n", encoding="utf-8")
        for command in ("docker", "nvidia-smi"):
            destination = bin_dir / command
            shutil.copy(repository / "tests" / "fixtures" / "fake_runtime" / command, destination)
            destination.chmod(0o755)
        state = root / "container.state"
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PROJECT_DIR": str(project),
            "MODEL_DIR": str(model),
            "DATASETS_DIR": str(datasets),
            "RESULTS_DIR": str(results),
            "RUN_ID": "unit-runtime",
            "MOCK_CONTAINER_STATE": str(state),
            "MOCK_PIPELINE_STATUS": str(pipeline_status),
            "STARTUP_TIMEOUT_SEC": "5",
            "DEEPSEEK_API_KEY": "fake-secret",
            "DEEPSEEK_BASE_URL": "https://deepseek.invalid/v1",
        }
        return repository / "scripts" / "run_ephemeral_sglang.sh", env, state

    def run_runner(self, pipeline_status: int):
        with tempfile.TemporaryDirectory() as directory:
            runner, env, state = self.runtime_environment(Path(directory), pipeline_status)
            completed = subprocess.run(
                ["bash", str(runner), "configs/smoke.yaml"],
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            return completed, state.exists()

    def test_success_cleans_container(self):
        completed, state_exists = self.run_runner(0)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(state_exists)
        self.assertIn("removing task container", completed.stdout)

    def test_pipeline_failure_is_preserved_and_container_is_cleaned(self):
        completed, state_exists = self.run_runner(7)
        self.assertEqual(completed.returncode, 7, completed.stderr)
        self.assertFalse(state_exists)
        self.assertIn("evaluation exited with status 7", completed.stdout)

    def test_sigint_cleans_container_and_returns_130(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, env, state = self.runtime_environment(Path(directory), 0)
            env["MOCK_PIPELINE_SLEEP"] = "30"
            process = subprocess.Popen(
                ["bash", str(runner), "configs/smoke.yaml"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            deadline = time.time() + 5
            while not state.exists() and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(state.exists(), "mock container was not created")
            os.killpg(process.pid, signal.SIGINT)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertFalse(state.exists())
            self.assertIn("removing task container", stdout)


if __name__ == "__main__":
    unittest.main()
