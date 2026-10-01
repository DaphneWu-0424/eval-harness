from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from eval_harness.core.interfaces import ModelBackend
from eval_harness.core.models import TaskSample


@dataclass(slots=True)
class LLMJudgeResult:
    verdict: str
    reason: str
    usage: dict[str, Any]
    metadata: dict[str, Any]
    raw_output: str | None = None
    error: str | None = None


class LLMJudge:
    def __init__(
        self,
        backend: ModelBackend,
        generation_config: dict[str, Any],
        model_name: str,
    ):
        self.backend = backend
        self.generation_config = generation_config
        self.model_name = model_name

    async def judge(
        self,
        sample: TaskSample,
        candidate: object,
        rule_verdict: str,
    ) -> LLMJudgeResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict mathematical answer judge. Determine whether the candidate "
                    "answer is mathematically equivalent to the reference answer for the given "
                    "problem. Ignore formatting and minor wording differences. Do not solve a "
                    "different problem. Return exactly one JSON object with keys verdict and reason. "
                    "verdict must be CORRECT, INCORRECT, or UNKNOWN. Use UNKNOWN when the information "
                    "is insufficient to judge reliably."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{sample.input}\n\n"
                    f"Reference answer:\n{sample.reference}\n\n"
                    f"Candidate answer:\n{candidate}\n\n"
                    f"Rule judge verdict (context only): {rule_verdict}"
                ),
            },
        ]
        try:
            response = await self.backend.generate(messages, self.generation_config)
            output = response.content or response.reasoning or ""
            verdict, reason = parse_judge_output(output)
            return LLMJudgeResult(
                verdict=verdict,
                reason=reason,
                usage=response.usage,
                metadata={"model": self.model_name, **response.metadata},
                raw_output=output,
            )
        except Exception as exc:
            return LLMJudgeResult(
                verdict="UNKNOWN",
                reason="LLM judge request or response parsing failed",
                usage={},
                metadata={"model": self.model_name},
                error=f"{type(exc).__name__}: {exc}",
            )


def parse_judge_output(output: str) -> tuple[str, str]:
    text = output.strip()
    candidates = [text]
    candidates.extend(re.findall(r"\{[^{}]*\}", text, flags=re.S))
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        verdict = str(value.get("verdict", "")).upper()
        if verdict in {"CORRECT", "INCORRECT", "UNKNOWN"}:
            return verdict, str(value.get("reason", "")).strip()
    labels = re.findall(r"\b(CORRECT|INCORRECT|UNKNOWN)\b", text.upper())
    if labels:
        return labels[-1], text[:2000]
    raise ValueError(f"unparseable LLM judge output: {text[:500]}")

