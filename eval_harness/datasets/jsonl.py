from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eval_harness.core.interfaces import Dataset
from eval_harness.core.models import TaskSample
from eval_harness.core.registry import DATASETS


class JsonlDataset(Dataset):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.path = Path(config["path"])

    def load(self) -> list[TaskSample]:
        input_key = self.config.get("input_key", "input")
        reference_key = self.config.get("reference_key", "reference")
        source_key = self.config.get("source_key", "source")
        id_key = self.config.get("id_key")
        limit = self.config.get("limit")
        num_shards = int(self.config.get("num_shards", 1))
        shard_index = int(self.config.get("shard_index", 0))
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("dataset shard_index must satisfy 0 <= shard_index < num_shards")
        samples: list[TaskSample] = []
        seen: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if input_key not in record:
                    raise ValueError(f"{self.path}:{line_number}: missing {input_key!r}")
                global_index = line_number - 1
                if global_index % num_shards != shard_index:
                    continue
                sample_id = str(record.get(id_key, global_index) if id_key else global_index)
                if sample_id in seen:
                    raise ValueError(f"duplicate sample_id {sample_id!r}")
                seen.add(sample_id)
                excluded = {input_key, reference_key, source_key}
                if id_key:
                    excluded.add(id_key)
                samples.append(
                    TaskSample(
                        sample_id=sample_id,
                        input=record[input_key],
                        reference=record.get(reference_key),
                        source=record.get(source_key),
                        metadata={k: v for k, v in record.items() if k not in excluded},
                    )
                )
                if limit is not None and len(samples) >= int(limit):
                    break
        if not samples:
            raise ValueError(f"dataset is empty: {self.path}")
        return samples

    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


DATASETS.register("jsonl", lambda config, **_: JsonlDataset(config))
