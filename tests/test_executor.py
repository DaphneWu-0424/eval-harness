import unittest

from eval_harness.backends.openai_compatible import BackendRequestError
from eval_harness.core.interfaces import ModelBackend
from eval_harness.core.models import TaskSample
from eval_harness.executors.model import ModelExecutor
from eval_harness.prompts.math import MathBoxedPrompt


class FailingBackend(ModelBackend):
    async def generate(self, messages, generation_config):
        raise BackendRequestError("failed", metadata={"endpoint": "x", "attempts": 5})


class CapturingBackend(ModelBackend):
    def __init__(self):
        self.configs = []

    async def generate(self, messages, generation_config):
        from eval_harness.core.models import ModelResponse

        self.configs.append(generation_config)
        return ModelResponse(content="ok")


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_becomes_retryable_trajectory(self):
        executor = ModelExecutor(FailingBackend(), MathBoxedPrompt({}), {})
        trajectory = await executor.run(TaskSample("s", "question"), 2)
        self.assertEqual(trajectory.status, "failed")
        self.assertEqual(trajectory.rollout_id, 2)
        self.assertEqual(trajectory.metadata["attempts"], 5)
        self.assertIn("failed", trajectory.error)

    async def test_rollouts_receive_stable_distinct_seeds(self):
        backend = CapturingBackend()
        executor = ModelExecutor(backend, MathBoxedPrompt({}), {"temperature": 1.0}, 17)
        sample = TaskSample("sample-1", "question")
        await executor.run(sample, 0)
        await executor.run(sample, 1)
        first, second = backend.configs
        self.assertNotEqual(first["seed"], second["seed"])
        self.assertEqual(first["temperature"], 1.0)

        replay = CapturingBackend()
        await ModelExecutor(replay, MathBoxedPrompt({}), {}, 17).run(sample, 0)
        self.assertEqual(first["seed"], replay.configs[0]["seed"])


if __name__ == "__main__":
    unittest.main()
