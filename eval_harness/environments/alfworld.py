from __future__ import annotations

import contextvars
import os
import sys
from typing import Any

from eval_harness.core.interfaces import Environment
from eval_harness.core.models import StepResult, TaskSample
from eval_harness.core.registry import ENVIRONMENTS

# Task-local env: each asyncio task (one per sample) holds its own textworld.gym
# instance, so concurrent samples don't clobber each other's environment.
_current_env: contextvars.ContextVar = contextvars.ContextVar("alfworld_current_env", default=None)


class ALFWorldAdapter(Environment):
    """Thin adapter over the EDGE-vendored ``AlfredTWEnv`` (textworld.gym).

    Runs the environment directly on CPU, bypassing Ray/veRL/torch. The shared
    ``AlfredTWEnv`` base is built once in ``__init__``; each ``reset`` creates a
    per-sample textworld.gym env stored in a task-local ContextVar, so the
    pipeline can run multiple samples concurrently.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.pkg_path = config["alfworld_package_path"]
        self.config_path = config.get(
            "config_path", os.path.join(self.pkg_path, "configs", "config_tw.yaml")
        )
        self.eval_dataset = config.get("eval_dataset", "eval_in_distribution")
        data_root_env = config.get("data_root_env", "ALFWORLD_DATA")
        if data_root_env in os.environ:
            os.environ.setdefault("ALFWORLD_DATA", os.environ[data_root_env])
        if self.pkg_path not in sys.path:
            sys.path.insert(0, self.pkg_path)
        import yaml
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # noqa: WPS433

        with open(self.config_path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        cfg["env"]["type"] = "AlfredTWEnv"
        cfg["general"]["training_method"] = "dagger"
        self._base_env = AlfredTWEnv(cfg, train_eval=self.eval_dataset)

    async def reset(self, sample: TaskSample) -> dict[str, Any]:
        game_file = sample.input if isinstance(sample.input, str) else sample.input["game_file"]
        env = self._base_env.init_env(batch_size=1, game_files=[game_file], fixed_eval=True)
        _current_env.set(env)
        obs, infos = env.reset()
        return {
            "observation": self._first(obs),
            "admissible_commands": list(infos["admissible_commands"][0]),
            "won": bool(infos["won"][0]),
            "info": {key: self._first(value) for key, value in infos.items()},
        }

    async def step(self, action: Any) -> StepResult:
        env = _current_env.get()
        if env is None:
            raise RuntimeError("ALFWorldAdapter.step called before reset")
        step_obs, scores, dones, step_infos = env.step([str(action)])
        return StepResult(
            observation=self._first(step_obs),
            reward=self._first(scores),
            done=bool(self._first(dones)),
            metadata={
                "won": bool(step_infos["won"][0]),
                "admissible_commands": list(step_infos["admissible_commands"][0]),
                "info": {key: self._first(value) for key, value in step_infos.items()},
            },
        )

    async def close(self) -> None:
        env = _current_env.get()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            _current_env.set(None)

    @staticmethod
    def _first(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return value[0]
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.item() if value.size == 1 else value[0]
        except ImportError:
            pass
        return value


ENVIRONMENTS.register("alfworld", lambda config, **_: ALFWorldAdapter(config))
