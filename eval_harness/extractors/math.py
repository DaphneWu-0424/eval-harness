from __future__ import annotations

import re

from eval_harness.core.models import ExtractResult
from eval_harness.core.registry import EXTRACTORS


def _balanced_boxed_values(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            values.append(text[start : index - 1].strip())
    return values


class CompositeMathExtractor:
    patterns = (
        ("final_answer", re.compile(r"(?is)final\s+answer\s*:\s*(.+?)(?:\n|$)"), 0.9),
        ("answer", re.compile(r"(?is)(?:\*\*)?answer(?:\*\*)?\s*:\s*(.+?)(?:\n|$)"), 0.85),
    )

    def extract(self, output: object) -> ExtractResult:
        if output is None:
            return ExtractResult(None, "none", 0.0, "missing final output")
        text = str(output).strip()
        if not text:
            return ExtractResult(None, "empty", 0.0, "empty final output")
        boxed = _balanced_boxed_values(text)
        if boxed and boxed[-1]:
            return ExtractResult(boxed[-1], "last_boxed", 1.0)
        for method, pattern, confidence in self.patterns:
            matches = list(pattern.finditer(text))
            if matches:
                value = matches[-1].group(1).strip().strip("$`* ")
                if value:
                    return ExtractResult(value, method, confidence)
        lines = [line.strip().strip("$`* ") for line in text.splitlines() if line.strip()]
        if lines:
            candidate = lines[-1]
            if len(candidate) <= 200:
                return ExtractResult(candidate, "last_short_line", 0.5)
        return ExtractResult(None, "unresolved", 0.0, "no answer pattern matched")


EXTRACTORS.register("composite_math", lambda config, **_: CompositeMathExtractor())

