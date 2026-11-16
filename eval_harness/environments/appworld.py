import asyncio
import contextvars
import json
import subprocess
import sys

from eval_harness.core.interfaces import Environment
from eval_harness.core.models import StepResult
from eval_harness.core.registry import ENVIRONMENTS


class AppWorldAdapter(Environment):
    """One isolated SDK subprocess per episode, including its main-thread timers."""
    def __init__(self, config):
        self.horizon = int(config.get('max_steps', 50))
        self.timeout = int(config.get('timeout_seconds', 60))
        self.seed = int(config.get('seed', 0))
        if min(self.horizon, self.timeout) < 1:
            raise ValueError('AppWorld horizon and timeout must be positive')
        self.current = contextvars.ContextVar('appworld_process', default=None)

    async def call(self, **request):
        process = self.current.get()
        if process is None:
            raise RuntimeError('AppWorld worker is not running')
        def exchange():
            process.stdin.write(json.dumps(request) + '\n')
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f'AppWorld worker exited: {process.poll()}; inspect container logs')
            response = json.loads(line)
            if 'error' in response:
                raise RuntimeError(response['error'])
            return response['result']
        # Also covers reset/evaluation hangs outside the SDK's code execution timer.
        try:
            return await asyncio.wait_for(asyncio.to_thread(exchange), timeout=self.timeout + 180)
        except BaseException:
            process.appworld_broken = True
            raise

    async def reset(self, sample):
        await self.close()
        process = subprocess.Popen([sys.executable, '-m', 'eval_harness.environments.appworld_worker'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.current.set(process)
        result = await self.call(operation='reset', task_id=sample.input, max_steps=self.horizon,
                                 timeout_seconds=self.timeout, seed=self.seed)
        return {**result, 'task_description': result['observation'],
                'trajectory_metadata': {'task_id': sample.input, 'evaluated': False}}

    async def step(self, action):
        result = await self.call(operation='step', code=str(action))
        return StepResult(observation=result.pop('observation'), reward=result.pop('reward'),
                          done=result.pop('done'), metadata=result)

    async def close(self):
        process = self.current.get()
        if process is None:
            return
        error = None
        try:
            # After a timeout, kill first: the previous readline may still own stdout.
            if process.poll() is None and not getattr(process, 'appworld_broken', False):
                try:
                    await asyncio.wait_for(self.call(operation='close'), timeout=15)
                    await asyncio.to_thread(process.wait, 10)
                except Exception as exc:
                    error = exc
            if process.poll() is None:
                process.terminate()
            try:
                await asyncio.to_thread(process.wait, 10)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
        finally:
            process.stdin.close()
            process.stdout.close()
            self.current.set(None)
        if error is not None:
            raise RuntimeError(f'AppWorld worker did not close cleanly: {error}') from error


ENVIRONMENTS.register('appworld', lambda config, **_: AppWorldAdapter(config))
