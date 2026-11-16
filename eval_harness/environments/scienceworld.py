from __future__ import annotations

import asyncio
import contextvars
import random
import threading
from dataclasses import dataclass
from typing import Any

from eval_harness.core.interfaces import Environment
from eval_harness.core.models import StepResult
from eval_harness.core.registry import ENVIRONMENTS


@dataclass
class Episode:
    worker: Any
    episode: dict
    python_rng: Any = None
    numpy_rng: Any = None
    retained_score: float = 0.0


_CURRENT = contextvars.ContextVar("scienceworld_episode", default=None)
_NATIVE_LOCK = threading.RLock()


def make_worker(horizon):
    # The SDK/JVM dependencies are required only in the ScienceWorld container.
    from .scienceworld_worker import ScienceWorldWorker
    return ScienceWorldWorker(horizon=horizon)


class ScienceWorldAdapter(Environment):
    def __init__(self, config, worker_factory=None):
        self.horizon = int(config.get("max_steps", 50))
        if self.horizon < 1:
            raise ValueError("ScienceWorld max_steps must be positive")
        self.factory = worker_factory or make_worker

    @staticmethod
    def invoke(session, operation, *args):
        """Keep each concurrent episode's Python/NumPy RNG independent."""
        try:
            import numpy as np
        except ImportError:
            np = None  # Allows CPU interface tests with an injected fake worker.
        with _NATIVE_LOCK:
            external_python = random.getstate()
            external_numpy = np.random.get_state() if np is not None else None
            if session.python_rng is not None:
                random.setstate(session.python_rng)
            if np is not None and session.numpy_rng is not None:
                np.random.set_state(session.numpy_rng)
            try:
                return operation(*args)
            finally:
                session.python_rng = random.getstate()
                session.numpy_rng = np.random.get_state() if np is not None else None
                random.setstate(external_python)
                if np is not None:
                    np.random.set_state(external_numpy)

    def context(self, session, observation, info, done=False):
        raw = float(info["scienceworld_score"])
        if raw >= 0:
            session.retained_score = raw
        termination = None
        if done:
            termination = "native_failure" if raw < 0 else "success" if raw == 100 else "step_limit" if session.worker.steps >= self.horizon else "environment_done"
        facts = {**session.episode, "raw_score": raw, "task_score": max(0.0, raw),
                 "retained_score": session.retained_score, "termination": termination}
        return {"observation": str(observation), "task_description": info["taskDesc"],
                "look": info["look"], "inventory": info["inventory"],
                "action_templates": list(info["action_templates"]),
                "admissible_commands": list(info["admissible_commands"]),
                "won": raw == 100.0, "trajectory_metadata": facts}

    async def reset(self, sample):
        if _CURRENT.get() is not None:
            await self.close()
        episode = dict(sample.input)
        if episode["env_seed"] != episode["variation_id"]:
            raise ValueError("ScienceWorld seed/variation mismatch")
        session = Episode(self.factory(self.horizon), episode)
        _CURRENT.set(session)
        observation, info = await asyncio.to_thread(self.invoke, session, session.worker.reset, episode)
        return self.context(session, observation, info)

    async def step(self, action):
        session = _CURRENT.get()
        if session is None:
            raise RuntimeError("ScienceWorld step before reset")
        observation, reward, done, info = await asyncio.to_thread(self.invoke, session, session.worker.step, str(action))
        context = self.context(session, observation, info, done)
        return StepResult(observation=context.pop("observation"), reward=float(reward), done=bool(done), metadata=context)

    async def close(self):
        session = _CURRENT.get()
        if session is not None:
            try:
                await asyncio.to_thread(self.invoke, session, session.worker.close)
            finally:
                _CURRENT.set(None)


ENVIRONMENTS.register("scienceworld", lambda config, **_: ScienceWorldAdapter(config))
