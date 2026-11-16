"""Private JSON-lines subprocess: AppWorld globals and code execution stay here."""
import contextlib
import json
import sys
import uuid


def main():
    wire = sys.stdout
    world = None
    steps = 0
    horizon = 50
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                operation = request['operation']
                if operation == 'reset':
                    from appworld import AppWorld
                    if world is not None:
                        world.close()
                    horizon = int(request['max_steps'])
                    world = AppWorld(task_id=request['task_id'],
                                     experiment_name='eval-' + uuid.uuid4().hex,
                                     max_interactions=horizon,
                                     timeout_seconds=int(request['timeout_seconds']),
                                     random_seed=int(request['seed']),
                                     load_ground_truth=True, ground_truth_mode='minimal')
                    steps = 0
                    result = {'observation': world.task.instruction,
                              'supervisor': dict(world.task.supervisor)}
                elif operation == 'step':
                    if world is None:
                        raise RuntimeError('AppWorld step before reset')
                    observation = world.execute(request['code'])
                    steps += 1
                    completed = world.task_completed()
                    done = completed or steps >= horizon
                    facts = {'task_completed': completed, 'step_count': steps,
                             'termination': 'task_completed' if completed else 'step_limit' if done else None}
                    won = False
                    if done:
                        tracker = world.evaluate(suppress_errors=True)
                        # A missing/broken evaluation is an infrastructure error, not zero success.
                        if tracker.num_tests < 1 or tracker.total_count != tracker.num_tests:
                            raise RuntimeError('AppWorld evaluation tests missing or incomplete')
                        won = bool(tracker.success)
                        facts.update(evaluated=True, tests_passed=tracker.pass_count,
                                     tests_total=tracker.num_tests, task_success=won)
                    result = {'observation': observation, 'done': done, 'won': won,
                              'reward': 10.0 if won else 0.0, 'trajectory_metadata': facts}
                elif operation == 'close':
                    if world is not None:
                        world.close()
                    world = None
                    result = {}
                else:
                    raise ValueError('unknown AppWorld operation')
            response = {'result': result}
        except Exception as exc:
            response = {'error': f'{type(exc).__name__}: {exc}'}
        wire.write(json.dumps(response) + '\n')
        wire.flush()
        if request['operation'] == 'close':
            break
    if world is not None:
        with contextlib.redirect_stdout(sys.stderr):
            world.close()


if __name__ == '__main__':
    main()
