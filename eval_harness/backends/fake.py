from __future__ import annotations

from typing import Any

from eval_harness.core.interfaces import ModelBackend
from eval_harness.core.models import ModelResponse
from eval_harness.core.registry import BACKENDS


class FakeBackend(ModelBackend):
    def __init__(self, config: dict[str, Any]):
        self.responses = list(config.get("responses", ["\\boxed{1}"]))
        self.index = 0

    async def generate(
        self, messages: list[dict[str, Any]], generation_config: dict[str, Any]
    ) -> ModelResponse:
        content = self.responses[self.index % len(self.responses)]
        self.index += 1
        return ModelResponse(
            reasoning="fake reasoning",
            content=content,
            tool_calls=None,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw_response={"fake": True},
            metadata={"endpoint": "fake://local", "attempt": 1},
        )


BACKENDS.register("fake", lambda config, **_: FakeBackend(config))

