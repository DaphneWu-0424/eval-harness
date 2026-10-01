import unittest

from eval_harness.backends.fake import FakeBackend
from eval_harness.core.models import TaskSample, Trajectory
from eval_harness.evaluators.llm_judge import parse_judge_output
from eval_harness.evaluators.math import MathEvaluator


def evaluator(response: str) -> MathEvaluator:
    return MathEvaluator(
        {
            "type": "math",
            "llm_fallback": {
                "enabled": True,
                "trigger_verdicts": ["UNKNOWN", "INCORRECT"],
                "backend": {"model": "deepseek-test"},
                "generation": {"temperature": 0.0},
            },
        },
        llm_backend=FakeBackend({"responses": [response]}),
    )


class LLMJudgeTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_json_in_fence(self):
        verdict, reason = parse_judge_output(
            '```json\n{"verdict":"CORRECT","reason":"equivalent"}\n```'
        )
        self.assertEqual(verdict, "CORRECT")
        self.assertEqual(reason, "equivalent")

    async def test_llm_can_override_rule_incorrect(self):
        judge = evaluator('{"verdict":"CORRECT","reason":"reference alias"}')
        result = await judge.evaluate(
            TaskSample("s", "problem", "3"),
            Trajectory("s", 0, [], "\\boxed{2}", "completed"),
        )
        self.assertEqual(result.labels["rule_verdict"], "INCORRECT")
        self.assertEqual(result.labels["llm_verdict"], "CORRECT")
        self.assertEqual(result.reward, 1.0)

    async def test_unparseable_llm_keeps_rule_incorrect(self):
        judge = evaluator("I cannot provide structured output")
        result = await judge.evaluate(
            TaskSample("s", "problem", "3"),
            Trajectory("s", 0, [], "\\boxed{2}", "completed"),
        )
        self.assertEqual(result.labels["verdict"], "INCORRECT")
        self.assertIsNotNone(result.details["llm_fallback"]["error"])


if __name__ == "__main__":
    unittest.main()
