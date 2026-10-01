from __future__ import annotations

from typing import Any

from eval_harness.core.interfaces import PromptBuilder
from eval_harness.core.models import TaskSample
from eval_harness.core.registry import PROMPTS


DEFAULT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


class MathBoxedPrompt(PromptBuilder):
    def __init__(self, config: dict[str, Any]):
        self.system = config.get("system")
        self.suffix = config.get("suffix", DEFAULT_SUFFIX).strip()

    def build(self, sample: TaskSample) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": f"{sample.input}\n\n{self.suffix}"})
        return messages


PROMPTS.register("math_boxed", lambda config, **_: MathBoxedPrompt(config))

