import tempfile
import unittest
from pathlib import Path

from eval_harness.datasets.jsonl import JsonlDataset


class DatasetShardingTests(unittest.TestCase):
    def test_two_shards_are_disjoint_and_preserve_global_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "data.jsonl")
            path.write_text(
                "".join(
                    f'{{"problem":"p{index}","ground_truth":"{index}"}}\n'
                    for index in range(7)
                ),
                encoding="utf-8",
            )
            base = {"path": str(path), "input_key": "problem", "num_shards": 2}
            shard0 = JsonlDataset({**base, "shard_index": 0}).load()
            shard1 = JsonlDataset({**base, "shard_index": 1}).load()
            self.assertEqual([sample.sample_id for sample in shard0], ["0", "2", "4", "6"])
            self.assertEqual([sample.sample_id for sample in shard1], ["1", "3", "5"])
            self.assertEqual(
                {sample.sample_id for sample in shard0} & {sample.sample_id for sample in shard1},
                set(),
            )


if __name__ == "__main__":
    unittest.main()
