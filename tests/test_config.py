import copy
import unittest

from eval_harness.config import execution_fingerprint, load_config


class ConfigFingerprintTests(unittest.TestCase):
    def test_metrics_do_not_change_execution_fingerprint(self):
        config = {
            "task": {"type": "reasoning"},
            "dataset": {"type": "jsonl", "path": "x"},
            "executor": {"type": "model"},
            "prompt": {"type": "math_boxed"},
            "backend": {"type": "fake"},
            "generation": {"n": 1},
            "metrics": [{"type": "accuracy"}],
        }
        changed = copy.deepcopy(config)
        changed["metrics"] = [{"type": "length"}]
        self.assertEqual(execution_fingerprint(config), execution_fingerprint(changed))

    def test_generation_changes_execution_fingerprint(self):
        config = {"generation": {"n": 1}}
        changed = {"generation": {"n": 2}}
        self.assertNotEqual(execution_fingerprint(config), execution_fingerprint(changed))

    def test_formal_and_smoke_configs_load(self):
        for path in (
            "configs/reasoning/pcl_qwen35_9b_base.yaml",
            "configs/reasoning/pcl_qwen35_9b_smoke.yaml",
            "configs/reasoning/pcl_qwen35_9b_student_108samples_1epoch.yaml",
            "configs/reasoning/qwen38_27b_teacher_train_shard0.yaml",
            "configs/reasoning/qwen38_27b_teacher_train_shard1.yaml",
            "configs/reasoning/qwen38_27b_teacher_train_judge.yaml",
        ):
            config = load_config(path)
            self.assertIn("llm_fallback", config["evaluators"][0])

        judge_config = load_config(
            "configs/reasoning/qwen38_27b_teacher_train_judge.yaml"
        )
        self.assertEqual(
            judge_config["evaluators"][0]["llm_fallback"]["trigger_verdicts"],
            ["UNKNOWN", "INCORRECT"],
        )


if __name__ == "__main__":
    unittest.main()
