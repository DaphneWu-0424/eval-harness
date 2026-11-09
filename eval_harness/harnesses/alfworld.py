from __future__ import annotations

import re
from typing import Any

from eval_harness.core.interfaces import AgentHarness, ModelBackend
from eval_harness.core.models import TaskSample, Trajectory
from eval_harness.core.registry import HARNESSES

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

TEMPLATE_NO_HIS = """You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Retrieved Relevant Experience

No external general and task-specific experiences are provided. Use your learned strategy and current observation.

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


def format_admissible(admissible: list[str]) -> str:
    return "'" + "'\n '".join(admissible) + "'"


def extract_task(observation: str) -> str:
    match = re.search(r"Your task is to:\s*(.+)", observation)
    return match.group(1).strip() if match else ""


def parse_action(text: str | None) -> tuple[str, bool]:
    """Extract ``<action>...</action>`` and validate format like EDGE's projection."""
    if not text:
        return "", False
    original = text
    lowered = text.lower()
    start = lowered.find("<action>")
    end = lowered.find("</action>")
    think_start = original.find("<think>")
    think_end = original.find("</think>")
    format_valid = (
        0 <= start < end
        and think_start != -1
        and think_end != -1
        and re.search(r"[一-鿿]", original) is None
    )
    if start == -1 or end == -1:
        action = text[-30:].strip()
    else:
        action = text[start + len("<action>") : end].strip().lower()
    return action, format_valid


class ALFWorldHarness(AgentHarness):
    """Builds the EDGE-aligned ALFWorld prompt, calls the model, and parses the action."""

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
        admissible = observation.get("admissible_commands", [])
        step_count = sum(1 for e in trajectory.events if e.type == "action")
        history = self._build_history(trajectory)
        if step_count == 0:
            prompt = TEMPLATE_NO_HIS.format(
                current_observation=obs_text,
                admissible_actions=format_admissible(admissible),
            )
        else:
            start_index = step_count - len(history) + 1
            prompt = TEMPLATE.format(
                task_description=self._task_description(trajectory),
                step_count=step_count,
                history_length=len(history),
                action_history=self._format_history(history, start_index),
                current_step=step_count + 1,
                current_observation=obs_text,
                admissible_actions=format_admissible(admissible),
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = await self.model_backend.generate(messages, self.generation_config)
        content = response.content or ""
        action, format_valid = parse_action(content)
        admissible_set = {a.strip().lower() for a in admissible if a.strip().lower() != "help"}
        is_admissible = action.strip().lower() in admissible_set
        return {
            "action": action,
            "is_action_valid": format_valid and is_admissible,
            "reasoning": response.reasoning,
            "content": content,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }

    async def finalize(self, trajectory: Trajectory) -> None:
        return None

    def _task_description(self, trajectory: Trajectory) -> str:
        for event in trajectory.events:
            if event.type == "observation":
                return extract_task(str(event.content))
        return ""

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
    "alfworld",
    lambda config, **kwargs: ALFWorldHarness(
        config, kwargs["model_backend"], kwargs.get("generation_config", {})
    ),
)
