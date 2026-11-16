#!/usr/bin/env python3
"""Run inside the official image, with /run/data mounted; no model inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gold-train', action='store_true', help='Validate one training gold solution; never test-set solutions')
    args = parser.parse_args()
    from appworld import AppWorld, load_task_ids
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / 'configs/manifests/appworld-provenance.json').read_text())
    version = importlib.metadata.version('appworld')
    if version != provenance['appworld_version']:
        raise RuntimeError(f'AppWorld SDK changed: {version}; revalidate before changing protocol')
    report = {'sdk_version': version, 'model_inference': False, 'splits': {}}
    for split in ['test_normal', 'test_challenge']:
        path = root / f"configs/manifests/appworld-{split.replace('_', '-')}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance['manifests'][path.name]
        manifest = json.loads(path.read_text())
        assert load_task_ids(split) == manifest['task_ids'], f'{split} data differs from frozen manifest'
        report['splits'][split] = len(manifest['task_ids'])
    task_id = load_task_ids('train' if args.gold_train else 'test_normal')[0]
    world = AppWorld(task_id=task_id, experiment_name='sdk-check',
                     ground_truth_mode='full' if args.gold_train else 'minimal')
    try:
        initial = world.evaluate()
        assert not initial.success, 'Unmodified task should fail'
        world.execute('print(apis.api_docs.show_app_descriptions())')
        report.update(task_id=task_id, initial_success=initial.success)
        if args.gold_train:
            output = world.execute(world.task.ground_truth.compiled_solution_code_body)
            tracker = world.evaluate()
            assert tracker.success, output
            report.update(gold_split='train', gold_success=tracker.success,
                          tests_passed=tracker.pass_count, tests_total=tracker.num_tests)
    finally:
        world.close()
    report['training_imports'] = [name for name in ['ray', 'verl', 'torch'] if name in sys.modules]
    assert not report['training_imports']
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
