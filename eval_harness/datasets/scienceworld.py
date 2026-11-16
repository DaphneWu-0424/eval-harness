from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval_harness.core.interfaces import Dataset
from eval_harness.core.models import TaskSample
from eval_harness.core.registry import DATASETS


class ScienceWorldManifest(Dataset):
    """Read a frozen official-test episode list without resampling variations."""
    def __init__(self, config):
        self.config = config

    def load(self):
        path = Path(self.config["path"])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("task") != "scienceworld" or manifest.get("split") != "official_test":
            raise ValueError("expected a version-1 ScienceWorld official_test manifest")
        episodes = manifest["episodes"]
        if not episodes or len(episodes) != manifest["suite_size"]:
            raise ValueError("ScienceWorld suite_size does not match episodes")
        counts = Counter(e.get("task_type", e["task_name"]) for e in episodes)
        if dict(counts) != manifest["task_type_counts"]:
            raise ValueError("ScienceWorld task_type_counts mismatch")
        seen = set()
        samples = []
        for entry in episodes:
            episode = dict(entry)
            key = str(episode["episode_id"])
            if key in seen:
                raise ValueError(f"duplicate ScienceWorld episode: {key}")
            seen.add(key)
            variation = episode["variation_id"]
            seed = episode["env_seed"]
            if not isinstance(variation, int) or variation < 0 or not isinstance(seed, int) or seed != variation:
                raise ValueError("ScienceWorld env_seed must equal the nonnegative variation_id")
            episode.setdefault("task_type", episode["task_name"])
            samples.append(TaskSample(
                sample_id=key, input=episode, source=episode["task_type"],
                metadata={"task_type": episode["task_type"], "manifest_suite_size": manifest["suite_size"],
                          "manifest_task_types": len(counts), "split": manifest["split"]},
            ))
        limit = self.config.get("limit")
        if limit is not None:
            if int(limit) < 1:
                raise ValueError("dataset.limit must be positive")
            samples = samples[:int(limit)]
        return samples


DATASETS.register("scienceworld_manifest", lambda config, **_: ScienceWorldManifest(config))
