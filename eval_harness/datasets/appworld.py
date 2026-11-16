import json
import re
from collections import Counter
from pathlib import Path

from eval_harness.core.interfaces import Dataset
from eval_harness.core.models import TaskSample
from eval_harness.core.registry import DATASETS


class AppWorldManifest(Dataset):
    def __init__(self, config):
        self.config = config

    def load(self):
        data = json.loads(Path(self.config['path']).read_text(encoding='utf-8'))
        ids = data['task_ids']
        if (data.get('schema_version') != 1 or data.get('task') != 'appworld'
                or data.get('split') not in ('train', 'dev', 'test_normal', 'test_challenge')
                or not ids or len(ids) != data['suite_size'] or len(set(ids)) != len(ids)
                or any(not re.fullmatch(r'[a-z0-9]+_\d+', i) for i in ids)):
            raise ValueError('invalid AppWorld manifest')
        counts = Counter(i.rsplit('_', 1)[0] for i in ids)
        samples = [TaskSample(i, i, source=data['split'], metadata={
            'scenario': i.rsplit('_', 1)[0], 'scenario_tasks': counts[i.rsplit('_', 1)[0]],
            'manifest_suite_size': len(ids), 'split': data['split']}) for i in ids]
        limit = self.config.get('limit')
        if limit is not None and int(limit) < 1:
            raise ValueError('dataset.limit must be positive')
        return samples if limit is None else samples[:int(limit)]


DATASETS.register('appworld_manifest', lambda config, **_: AppWorldManifest(config))
