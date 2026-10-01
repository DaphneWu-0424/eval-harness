from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    config = copy.deepcopy(data)
    validate_config(config)
    config.setdefault("_resolved", {})["source_path"] = str(config_path)
    config["_resolved"]["config_hash"] = config_hash(config, exclude_resolved=True)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("experiment", "dataset", "executor", "backend", "evaluators", "metrics")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing config sections: {', '.join(missing)}")
    generation = config.setdefault("generation", {})
    generation.setdefault("n", 1)
    generation.setdefault("concurrency", 1)
    generation.setdefault("timeout_sec", 300)
    generation.setdefault("retries", 3)
    if generation["n"] < 1 or generation["concurrency"] < 1:
        raise ValueError("generation.n and generation.concurrency must be positive")


def config_hash(config: dict[str, Any], *, exclude_resolved: bool = False) -> str:
    value = {k: v for k, v in config.items() if not (exclude_resolved and k == "_resolved")}
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def execution_fingerprint(config: dict[str, Any]) -> str:
    """Hash only inputs that can change raw model trajectories."""
    sections = ("task", "dataset", "executor", "prompt", "backend", "generation")
    value = {section: config.get(section) for section in sections}
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dump_config(config: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
