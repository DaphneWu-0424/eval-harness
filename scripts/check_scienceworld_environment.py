#!/usr/bin/env python3
"""CPU-only SDK/adapter checks; scripted gold actions are not model evaluations."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_harness.datasets.scienceworld import ScienceWorldManifest
from eval_harness.environments.scienceworld import ScienceWorldAdapter
from eval_harness.environments.scienceworld_worker import ScienceWorldWorker


async def check(args):
    samples = ScienceWorldManifest({"path": args.manifest}).load()
    representatives = {}
    for sample in samples:
        representatives.setdefault(sample.source, sample)
    selected = list(representatives.values())[:args.task_types]
    report = {"model_inference": False, "gpu": False, "checks": [], "gold_horizon": 1000}
    reference = None
    if args.reference_worker:
        path = Path(args.reference_worker)
        provenance = json.loads((Path(__file__).resolve().parents[1] / "configs/manifests/scienceworld-provenance.json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["worker_full_source_sha256"]
        spec = importlib.util.spec_from_file_location("colleague_worker", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    for index, sample in enumerate(selected):
        env = ScienceWorldAdapter({"max_steps": 50})
        baseline = reference.ScienceWorldWorker(horizon=50) if reference else None
        try:
            current = await env.reset(sample)
            if baseline:
                observation, info = baseline.reset(sample.input)
                assert current["observation"] == observation
                assert current["action_templates"] == info["action_templates"]
                for action in ["look", "inventory", "look"]:
                    expected_obs, expected_reward, expected_done, expected_info = baseline.step(action)
                    actual = await env.step(action)
                    assert actual.observation == expected_obs
                    assert actual.reward == expected_reward and actual.done == expected_done
                    assert actual.metadata["trajectory_metadata"]["raw_score"] == expected_info["scienceworld_score"]
        finally:
            await env.close()
            if baseline:
                baseline.close()
        if index >= args.gold_task_types:
            report["checks"].append({"episode_id": sample.sample_id, "task_type": sample.source,
                                     "reference_reset_step_match": baseline is not None})
            continue
        workers = []
        class GoldWorker(ScienceWorldWorker):
            def reset(self, episode):
                return super().reset(episode, generate_gold=True)
        def factory(horizon):
            worker = GoldWorker(horizon);workers.append(worker);return worker
        env = ScienceWorldAdapter({"max_steps": 1000}, worker_factory=factory)
        try:
            await env.reset(sample)
            actions = workers[0].gold_actions
            steps = 0
            for action in actions:
                result = await env.step(action)
                steps += 1
                if result.done:
                    break
            assert result.metadata["won"], f"gold path did not complete: {sample.sample_id}"
            report["checks"].append({"episode_id": sample.sample_id, "task_type": sample.source,
                                     "reference_reset_step_match": baseline is not None,
                                     "gold_steps": steps, "gold_score": result.metadata["trajectory_metadata"]["raw_score"]})
        finally:
            await env.close()
    report["training_imports"] = [name for name in ["ray", "verl", "torch"] if name in sys.modules]
    assert not report["training_imports"], report["training_imports"]
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="configs/manifests/scienceworld-claim.json")
    p.add_argument("--task-types", type=int, default=2)
    p.add_argument("--gold-task-types", type=int, default=2)
    p.add_argument("--reference-worker")
    p.add_argument("--output", required=True)
    asyncio.run(check(p.parse_args()))
