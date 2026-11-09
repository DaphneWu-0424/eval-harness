from __future__ import annotations

import re
import sys
from typing import Any

from eval_harness.core.interfaces import Environment
from eval_harness.core.models import StepResult, TaskSample
from eval_harness.core.registry import ENVIRONMENTS


def _extract_instruction(obs: str) -> str:
    match = re.search(r"Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]", obs)
    return match.group(1).strip() if match else ""


class WebShopAdapter(Environment):
    """Thin adapter over the EDGE-vendored ``web_agent_site.WebAgentTextEnv``.

    Holds a single env instance (products + search index loaded once); each
    ``reset`` switches the browser session. Safe under ``concurrency=1``.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.pkg_path = config["webshop_package_path"]
        self.num_products = config.get("num_products", 1000)
        if self.pkg_path not in sys.path:
            sys.path.insert(0, self.pkg_path)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433

        self._env = WebAgentTextEnv(observation_mode="text", num_products=self.num_products)

    def _available_actions(self) -> list[str]:
        info = self._env.get_available_actions()
        actions = []
        if info.get("has_search_bar"):
            actions.append("search[...]")
        for clickable in info.get("clickables", []):
            actions.append(f"click[{clickable}]")
        return actions

    async def reset(self, sample: TaskSample) -> dict[str, Any]:
        session = sample.input if isinstance(sample.input, (int, str)) else sample.input["session"]
        obs, info = self._env.reset(session=int(session))
        return {
            "observation": obs,
            "available_actions": self._available_actions(),
            "task_description": _extract_instruction(obs),
            "info": info,
        }

    async def step(self, action: Any) -> StepResult:
        state, reward, done, info = self._env.step(str(action))
        return StepResult(
            observation=state,
            reward=reward,
            done=bool(done),
            metadata={
                "won": bool(done and reward == 1.0),
                "available_actions": self._available_actions(),
                "info": info,
            },
        )

    async def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


ENVIRONMENTS.register("webshop", lambda config, **_: WebShopAdapter(config))
