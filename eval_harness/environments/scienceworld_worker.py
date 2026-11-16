# Derived from the EDGE-based ScienceWorld implementation (Apache-2.0).
# Frozen source: n30285:/ssd/sw-fix-20261113/runs/A-S-r1-8g/code/agent_system/environments/env_package/scienceworld/envs.py
# Full source SHA256: 94cd6bfb51adb2e6f1e1c7cb062ad969fb8b636dd42e697b63b44d1afc8fb378
# Only the single-episode worker is retained; no Ray/Trainer imports.
"""ScienceWorld backend; no simplifications or cumulative-score summation."""
import random
import os
import subprocess
import threading
import numpy as np
from scienceworld import ScienceWorldEnv
import scienceworld.scienceworld as scienceworld_sdk


def canonical_object_lists(text):
    """Canonicalize unordered tab-indented siblings, preserving all subtrees."""
    text = str(text)
    if not any(line.startswith('\t') for line in text.splitlines()):
        return text
    newline = '\r\n' if '\r\n' in text else ('\r' if '\r' in text and '\n' not in text else '\n')
    terminated = text.endswith(('\n','\r'))
    def depth(line):
        return len(line)-len(line.lstrip('\t'))
    def ordered(lines):
        level=min(depth(line) for line in lines)
        if depth(lines[0])!=level:
            return lines
        chunks=[]
        for line in lines:
            if depth(line)==level:
                chunks.append([line])
            else:
                chunks[-1].append(line)
        canonical=[chunk[:1]+(ordered(chunk[1:]) if len(chunk)>1 else []) for chunk in chunks]
        return [line for chunk in sorted(canonical,key=lambda c:''.join(c)) for line in chunk]
    output=[]
    pending=[]
    for line in text.splitlines():
        if line.startswith('\t'):
            pending.append(line)
        else:
            if pending:
                output.extend(ordered(pending))
                pending=[]
            output.append(line)
    if pending:
        output.extend(ordered(pending))
    return newline.join(output)+(newline if terminated else '')


class EpisodeScienceWorldEnv(ScienceWorldEnv):
    """The upstream destructor closes again; make episode teardown idempotent."""
    def close(self):
        if getattr(self, '_episode_closed', False):
            return
        self._episode_closed = True
        if hasattr(self, '_gateway'):
            try:
                super().close()
            finally:
                self._gateway.shutdown_callback_server()
                process = self._gateway.java_process
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)


_JVM_CREATION_LOCK = threading.RLock()


def create_episode_environment(seed):
    """Seed before native interface construction and pin identity-hash order."""
    with _JVM_CREATION_LOCK:
        gateway_factory = scienceworld_sdk.JavaGateway
        previous_options = os.environ.get('JAVA_TOOL_OPTIONS')
        def seeded_gateway(*args, **kwargs):
            gateway = gateway_factory(*args, **kwargs)
            native_rng = getattr(getattr(gateway.jvm.scala.util, 'Random$'), 'MODULE$')
            native_rng.setSeed(int(seed))
            return gateway
        scienceworld_sdk.JavaGateway = seeded_gateway
        os.environ['JAVA_TOOL_OPTIONS'] = ((previous_options or '')+
            ' -XX:+UnlockExperimentalVMOptions -XX:hashCode=2').strip()
        try:
            return EpisodeScienceWorldEnv(envStepLimit=2**31 - 1)
        finally:
            scienceworld_sdk.JavaGateway = gateway_factory
            if previous_options is None:
                os.environ.pop('JAVA_TOOL_OPTIONS', None)
            else:
                os.environ['JAVA_TOOL_OPTIONS'] = previous_options


class ScienceWorldWorker:
    def __init__(self, horizon=50):
        self.horizon = horizon
        # ScienceWorld counts simulator moves, including reset/internal actions.
        # The recipe horizon is agent actions; enforce it in step(), below.
        self.env = None
        self.gold_actions = None
        self.closed = False

    def task_catalog(self):
        if self.env is None:
            self.env = create_episode_environment(0)
            self.closed = False
        tasks = self.env.get_task_names()
        if len(tasks) != 30:
            raise ValueError(f'Expected exactly 30 task types, got {len(tasks)}')
        result = {}
        for task in tasks:
            self.env.load(task, 0, '')
            result[task] = {'train': self.env.get_variations_train(),
                            'dev': self.env.get_variations_dev(),
                            'test': self.env.get_variations_test()}
        return result

    def reset(self, episode, generate_gold=False):
        if int(episode['env_seed']) != int(episode['variation_id']):
            raise ValueError('ScienceWorld natively seeds from variation_id; a distinct env_seed is unsupported')
        # A JVM retains simulator globals/task caches across loads. A new
        # episode must not depend on a worker's different previous rollout.
        if self.env is not None:
            self.env.close()
        self.env = create_episode_environment(int(episode['env_seed']))
        self.closed = False
        self.episode = dict(episode)
        random.seed(int(episode['env_seed']))
        np.random.seed(int(episode['env_seed']))
        native_rng = getattr(getattr(self.env._gateway.jvm.scala.util, 'Random$'), 'MODULE$')
        native_rng.setSeed(int(episode['env_seed']))
        self.env.load(episode['task_name'], int(episode['variation_id']), '',
                      generateGoldPath=generate_gold)
        self.gold_actions = list(self.env.get_gold_action_sequence()) if generate_gold else None
        if generate_gold and (not self.gold_actions or any('ERROR:' in a for a in self.gold_actions)):
            raise ValueError(f'Gold sequence unavailable for {episode}')
        native_rng.setSeed(int(episode['env_seed']))
        obs, info = self.env.reset()
        self.steps = 0
        self.previous_clipped_score = 0.0
        self.done = False
        self.last = (canonical_object_lists(obs), self._info(info))
        return self.last

    def _info(self, info):
        info = dict(info)
        info['look'] = canonical_object_lists(info['look'])
        info['inv'] = canonical_object_lists(info['inv'])
        score = float(info['score'])
        info.update(self.episode)
        info.update(won=score == 100.0, binary_completion=score == 100.0,
                    scienceworld_score=score, task_score=score,
                    admissible_commands=list(info['valid']),
                    action_templates=sorted(self.env.get_possible_actions()),
                    inventory=info['inv'])
        return info

    def step(self, action):
        if self.done:
            return self.last[0], 0.0, True, dict(self.last[1])
        obs, _, native_done, info = self.env.step(action)
        self.steps += 1
        info = self._info(info)
        if info['scienceworld_score'] < 0:
            reward = 0.0
        else:
            clipped_score = info['scienceworld_score'] / 100.0
            reward = clipped_score - self.previous_clipped_score
            self.previous_clipped_score = clipped_score
        self.done = bool(native_done or info['won'] or self.steps >= self.horizon)
        self.last = (canonical_object_lists(obs), info)
        return self.last[0], reward, self.done, info

    def close(self):
        if not self.closed:
            if self.env is not None:
                self.env.close()
            self.closed = True
