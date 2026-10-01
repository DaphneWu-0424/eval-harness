from __future__ import annotations

import math
import hashlib
import json
import re
from fractions import Fraction
from typing import Any

from eval_harness.core.interfaces import Evaluator
from eval_harness.core.models import EvaluationResult, TaskSample, Trajectory
from eval_harness.core.registry import EVALUATORS, EXTRACTORS
# Import built-in extractors for registry side effects.
from eval_harness import extractors as _extractors  # noqa: F401
from eval_harness.evaluators.llm_judge import LLMJudge, LLMJudgeResult

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except ImportError:  # pragma: no cover - tested in environments without optional dependency
    sympy = None

IMPLEMENTATION_VERSION = 4


def normalize_answer(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"^\\(?:\(|\[)(.*)\\(?:\)|\])$", r"\1", text, flags=re.S)
    text = text.replace("$", "").replace("\\,", "").replace(" ", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text)
    text = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = text.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    return text.rstrip(".，。")


def _numeric(value: str) -> float | None:
    try:
        if "/" in value and re.fullmatch(r"[+-]?\d+/[+-]?\d+", value):
            return float(Fraction(value))
        return float(value)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _simple_radical_signature(value: str) -> tuple[int, Fraction] | None:
    """Return the sign and exact square for sqrt(rational)/rational forms."""
    match = re.fullmatch(
        r"sqrt\(\(?([+-]?\d+(?:/[+-]?\d+)?)\)?\)(?:/([+-]?\d+(?:/[+-]?\d+)?))?",
        value,
    )
    if not match:
        return None
    try:
        radicand = Fraction(match.group(1))
        divisor = Fraction(match.group(2) or "1")
        if radicand < 0 or divisor == 0:
            return None
        sign = 1 if divisor > 0 else -1
        return sign, radicand / (divisor * divisor)
    except (ValueError, ZeroDivisionError):
        return None


def _assignment_map(value: str) -> dict[str, str] | None:
    parts = value.split(",")
    if not parts or any("=" not in part for part in parts):
        return None
    result: dict[str, str] = {}
    for part in parts:
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)=(.+)", part)
        if not match or match.group(1) in result:
            return None
        result[match.group(1)] = match.group(2)
    return result


def compare_math_answers(prediction: object, reference: object) -> bool | None:
    if prediction is None or reference is None:
        return None
    left = normalize_answer(prediction)
    right = normalize_answer(reference)
    if not left or not right:
        return None
    if left.casefold() == right.casefold():
        return True
    left_assignments = _assignment_map(left)
    right_assignments = _assignment_map(right)
    if left_assignments is not None and right_assignments is not None:
        if left_assignments.keys() != right_assignments.keys():
            return None
        decisions = [
            compare_math_answers(left_assignments[key], right_assignments[key])
            for key in left_assignments
        ]
        if any(decision is False for decision in decisions):
            return False
        if all(decision is True for decision in decisions):
            return True
        return None
    left_number, right_number = _numeric(left), _numeric(right)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-12)
    left_radical = _simple_radical_signature(left)
    right_radical = _simple_radical_signature(right)
    if left_radical is not None and right_radical is not None:
        return left_radical == right_radical
    if sympy is None:
        return None
    transformations = standard_transformations + (convert_xor, implicit_multiplication_application)
    try:
        left_expr = parse_expr(left, transformations=transformations, evaluate=True)
        right_expr = parse_expr(right, transformations=transformations, evaluate=True)
        if left_expr.free_symbols != right_expr.free_symbols:
            return None
        difference = sympy.simplify(left_expr - right_expr)
        return bool(difference == 0)
    except Exception:
        return None


class MathEvaluator(Evaluator):
    def __init__(self, config: dict[str, Any], llm_backend: Any | None = None):
        self.name = config.get("name", "math_v1")
        encoded = json.dumps(
            {"config": config, "implementation_version": IMPLEMENTATION_VERSION},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.identity = f"{self.name}:{self.fingerprint[:12]}"
        self.extractor = EXTRACTORS.create(
            config.get("extractor", {"type": "composite_math"})
        )
        fallback = config.get("llm_fallback") or {}
        self.fallback_on = set(
            fallback.get("trigger_verdicts", fallback.get("on", ["UNKNOWN"]))
        )
        self.llm_judge = None
        if fallback.get("enabled"):
            if llm_backend is None:
                raise ValueError("llm_fallback.enabled requires a separate llm_backend")
            backend_config = fallback["backend"]
            self.llm_judge = LLMJudge(
                llm_backend,
                fallback.get("generation", {}),
                backend_config["model"],
            )

    async def evaluate(self, sample: TaskSample, trajectory: Trajectory) -> EvaluationResult:
        if trajectory.status != "completed":
            return self._result(trajectory, "UNKNOWN", None, {"reason": "generation_failed"})
        extracted = self.extractor.extract(trajectory.final_output)
        if extracted.value is None:
            rule_result = self._result(
                trajectory,
                "UNKNOWN",
                None,
                {"reason": "extraction_failed", "extraction": vars_like(extracted)},
            )
        else:
            decision = compare_math_answers(extracted.value, sample.reference)
            details = {
                "prediction": extracted.value,
                "reference": sample.reference,
                "extraction": vars_like(extracted),
            }
            if decision is True:
                rule_result = self._result(trajectory, "CORRECT", 1.0, details)
            elif decision is False:
                rule_result = self._result(trajectory, "INCORRECT", 0.0, details)
            else:
                details["reason"] = "judge_unresolved"
                rule_result = self._result(trajectory, "UNKNOWN", None, details)
        rule_verdict = rule_result.labels["verdict"]
        if (
            self.llm_judge is None
            or rule_verdict not in self.fallback_on
            or trajectory.final_output is None
        ):
            return rule_result
        candidate = extracted.value if extracted.value is not None else trajectory.final_output
        llm_result = await self.llm_judge.judge(sample, candidate, rule_verdict)
        return self._combine_llm_result(rule_result, llm_result)

    def _combine_llm_result(
        self, rule_result: EvaluationResult, llm_result: LLMJudgeResult
    ) -> EvaluationResult:
        rule_verdict = rule_result.labels["verdict"]
        final_verdict = llm_result.verdict if llm_result.verdict != "UNKNOWN" else rule_verdict
        reward = {"CORRECT": 1.0, "INCORRECT": 0.0}.get(final_verdict)
        details = {
            **rule_result.details,
            "rule_verdict": rule_verdict,
            "llm_fallback": {
                "verdict": llm_result.verdict,
                "reason": llm_result.reason,
                "model": llm_result.metadata.get("model"),
                "usage": llm_result.usage,
                "metadata": llm_result.metadata,
                "raw_output": llm_result.raw_output,
                "error": llm_result.error,
            },
        }
        return EvaluationResult(
            sample_id=rule_result.sample_id,
            rollout_id=rule_result.rollout_id,
            status="completed",
            reward=reward,
            scores={} if reward is None else {"correct": reward},
            labels={
                "verdict": final_verdict,
                "rule_verdict": rule_verdict,
                "llm_verdict": llm_result.verdict,
            },
            details=details,
            evaluator=rule_result.evaluator,
        )

    def _result(
        self, trajectory: Trajectory, label: str, reward: float | None, details: dict[str, Any]
    ) -> EvaluationResult:
        return EvaluationResult(
            sample_id=trajectory.sample_id,
            rollout_id=trajectory.rollout_id,
            status="completed",
            reward=reward,
            scores={} if reward is None else {"correct": reward},
            labels={"verdict": label},
            details=details,
            evaluator=self.identity,
        )


def vars_like(value: Any) -> dict[str, Any]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


EVALUATORS.register(
    "math", lambda config, **kwargs: MathEvaluator(config, llm_backend=kwargs.get("llm_backend"))
)
