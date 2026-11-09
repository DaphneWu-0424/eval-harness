from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, category: str):
        self.category = category
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._factories:
            raise ValueError(f"duplicate {self.category} registration: {name}")
        self._factories[name] = factory

    def create(self, config: dict[str, Any], **kwargs: Any) -> Any:
        kind = config.get("type")
        if not kind:
            raise ValueError(f"{self.category} config requires 'type'")
        try:
            factory = self._factories[kind]
        except KeyError as exc:
            choices = ", ".join(sorted(self._factories))
            raise ValueError(f"unknown {self.category} type {kind!r}; available: {choices}") from exc
        return factory(config, **kwargs)


DATASETS = Registry("dataset")
PROMPTS = Registry("prompt")
BACKENDS = Registry("backend")
EXECUTORS = Registry("executor")
EXTRACTORS = Registry("extractor")
EVALUATORS = Registry("evaluator")
METRICS = Registry("metric")
ENVIRONMENTS = Registry("environment")
HARNESSES = Registry("harness")
