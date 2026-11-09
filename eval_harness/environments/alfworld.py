from __future__ import annotations

import os
import sys
from typing import Any

from eval_harness.core.interfaces import Environment
from eval_harness.core.models import StepResult, TaskSample
from eval_harness.core.registry import ENVIRONMENTS


class ALFWorldAdapter(Environment):
    """Thin adapter over the EDGE-vendored ``AlfredTWEnv`` (textworld.gym).

    Runs the environment directly on CPU, bypassing Ray/veRL/torch. Holds a
    single textworld.gym instance, so it is only safe under ``concurrency=1``
    (per-sample env isolation is a follow-up).

    Config keys::

        alfworld_package_path: parent dir of the vendored ``alfworld`` package
        config_path:           path to config_tw.yaml (default under pkg path)
        eval_dataset:          'eval_in_distribution' | 'eval_out_of_distribution'
        data_root_env:         env var holding the ALFWorld data root (default ALFWORLD_DATA)
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
        self._env = None
        self._base_env = None
        self._yaml = None
        self._AlfredTWEnv = None

    def _import_alfworld(self) -> None:
        if self.pkg_path not in sys.path:
            sys.path.insert(0, self.pkg_path)
        import yaml
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # noqa: WPS433

        self._yaml = yaml
        self._AlfredTWEnv = AlfredTWEnv

    def _base(self):
        if self._base_env is None:
            self._import_alfworld()
            with open(self.config_path, encoding="utf-8") as handle:
                cfg = self._yaml.safe_load(handle)
            cfg["env"]["type"] = "AlfredTWEnv"
            cfg["general"]["training_method"] = "dagger"
            self._base_env = self._AlfredTWEnv(cfg, train_eval=self.eval_dataset)
        return self._base_env

    async def reset(self, sample: TaskSample) -> dict[str, Any]:
        game_file = sample.input if isinstance(sample.input, str) else sample.input["game_file"]
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        base = self._base()
        self._env = base.init_env(batch_size=1, game_files=[game_file], fixed_eval=True)
        obs, infos = self._env.reset()
        return {
            "observation": self._first(obs),
            "admissible_commands": list(infos["admissible_commands"][0]),
            "won": bool(infos["won"][0]),
            "info": {key: self._first(value) for key, value in infos.items()},
        }

    async def step(self, action: Any) -> StepResult:
        if self._env is None:
            raise RuntimeError("ALFWorldAdapter.step called before reset")
        step_obs, scores, dones, step_infos = self._env.step([str(action)])
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
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

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
