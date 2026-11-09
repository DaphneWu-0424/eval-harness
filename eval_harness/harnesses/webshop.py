from __future__ import annotations

import re
from typing import Any

from eval_harness.core.interfaces import AgentHarness, ModelBackend
from eval_harness.core.models import TaskSample, Trajectory
from eval_harness.core.registry import HARNESSES
from eval_harness.harnesses.alfworld import parse_action

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

TEMPLATE_NO_HIS = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

TEMPLATE = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.

## Retrieved Relevant Experience
No external general and task-specific experiences are provided. Use your learned strategy and current observation.

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


def is_valid_action(action: str) -> bool:
    return bool(re.fullmatch(r"search\[.+\]", action) or re.fullmatch(r"click\[.+\]", action))


class WebShopHarness(AgentHarness):
    """Builds the EDGE-aligned WebShop prompt, calls the model, and parses the action."""

    def __init__(
        self,
        config: dict[str, Any],
        model_backend: ModelBackend,
        generation_config: dict[str, Any],
    ):
        self.config = config
        self.model_backend = model_backend
        self.generation_config = dict(generation_config)
        self.history_length = int(config.get("history_length", 2))

    async def initialize(self, sample: TaskSample) -> None:
        return None

    async def next_action(self, trajectory: Trajectory, observation: Any) -> dict[str, Any]:
        obs_text = observation["observation"]
        admissible = observation.get("available_actions", [])
        task_description = observation.get("task_description", "")
        step_count = sum(1 for e in trajectory.events if e.type == "action")
        history = self._build_history(trajectory)
        if step_count == 0:
            prompt = TEMPLATE_NO_HIS.format(
                task_description=task_description,
                current_observation=obs_text,
                available_actions="\n".join(admissible),
            )
        else:
            start_index = step_count - len(history) + 1
            prompt = TEMPLATE.format(
                task_description=task_description,
                step_count=step_count,
                history_length=len(history),
                action_history=self._format_history(history, start_index),
                current_step=step_count + 1,
                current_observation=obs_text,
                available_actions="\n".join(admissible),
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = await self.model_backend.generate(messages, self.generation_config)
        content = response.content or ""
        action, format_valid = parse_action(content)
        return {
            "action": action,
            "is_action_valid": format_valid and is_valid_action(action),
            "reasoning": response.reasoning,
            "content": content,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }

    async def finalize(self, trajectory: Trajectory) -> None:
        return None

    def _build_history(self, trajectory: Trajectory) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        last_obs = None
        for event in trajectory.events:
            if event.type == "observation":
                last_obs = event.content
            elif event.type == "action" and last_obs is not None:
                pairs.append((str(last_obs), event.content))
                last_obs = None
        return pairs[-self.history_length :]

    @staticmethod
    def _format_history(history: list[tuple[str, str]], start_index: int) -> str:
        parts = []
        for offset, (obs, action) in enumerate(history):
            step_num = start_index + offset
            parts.append(f"[Observation {step_num}: '{obs}', Action {step_num}: '{action}']")
        return "\n".join(parts)


HARNESSES.register(
    "webshop",
    lambda config, **kwargs: WebShopHarness(
        config, kwargs["model_backend"], kwargs.get("generation_config", {})
    ),
)
