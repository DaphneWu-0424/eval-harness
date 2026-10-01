from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
from collections import Counter
from pathlib import Path

from eval_harness.config import load_config
from eval_harness.persistence import ExperimentStore
from eval_harness.pipeline import EvaluationPipeline
from eval_harness.core.registry import DATASETS
from eval_harness import datasets as _datasets  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "execute", "judge", "report"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        if name == "judge":
            sub.add_argument(
                "--input",
                action="append",
                help="current or legacy raw trajectory JSONL; repeat to merge shards",
            )
    dataset_parser = subparsers.add_parser("dataset-info")
    dataset_parser.add_argument("--config", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--output-dir", required=True)
    inspect_parser.add_argument("--sample-id", required=True)
    inspect_parser.add_argument("--rollout-id", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        store = ExperimentStore(args.output_dir)
        key = (args.sample_id, args.rollout_id)
        trajectory = store.load_trajectories().get(key)
        evaluation = store.load_evaluations().get(key)
        payload = {
            "trajectory": trajectory.to_dict() if trajectory else None,
            "evaluation": evaluation.to_dict() if evaluation else None,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    config = load_config(args.config)
    if args.command == "dataset-info":
        dataset = DATASETS.create(config["dataset"])
        samples = dataset.load()
        path = Path(config["dataset"]["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(
            json.dumps(
                {
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "samples": len(samples),
                    "sources": dict(sorted(Counter(s.source or "unknown" for s in samples).items())),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    stage = "all" if args.command == "run" else args.command
    raw_input = getattr(args, "input", None)
    metrics = asyncio.run(EvaluationPipeline(config, raw_input=raw_input).run(stage=stage))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0
