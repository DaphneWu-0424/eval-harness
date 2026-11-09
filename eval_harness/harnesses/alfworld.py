from __future__ import annotations

import re
from typing import Any

from eval_harness.core.interfaces import AgentHarness, ModelBackend
from eval_harness.core.models import TaskSample, Trajectory
from eval_harness.core.registry import HARNESSES


PROMPT_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment.

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


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
        action = text[start + len("<action>") : end].strip()
    return action, format_valid


class ALFWorldHarness(AgentHarness):
    """Builds the ALFWorld prompt, calls the model, and parses the action."""

    def __init__(
        self,
        config: dict[str, Any],
        model_backend: ModelBackend,
        generation_config: dict[str, Any],
    ):
        self.config = config
        self.model_backend = model_backend
        self.generation_config = dict(generation_config)
        self.history_length = int(config.get("history_length", 5))

    async def initialize(self, sample: TaskSample) -> None:
        return None

    async def next_action(self, trajectory: Trajectory, observation: Any) -> dict[str, Any]:
        obs_text = observation["observation"]
        admissible = observation.get("admissible_commands", [])
        history = self._build_history(trajectory)
        prompt = PROMPT_TEMPLATE.format(
            step_count=len(history),
            history_length=min(self.history_length, len(history)),
            action_history=self._format_history(history),
            current_step=len(history) + 1,
            current_observation=obs_text,
            admissible_actions=", ".join(admissible),
        )
        messages = [{"role": "user", "content": prompt}]
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

    def _build_history(self, trajectory: Trajectory) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        last_action = None
        for event in trajectory.events:
            if event.type == "action":
                last_action = event.content
            elif event.type == "observation" and last_action is not None:
                pairs.append((last_action, str(event.content)[:200]))
                last_action = None
        return pairs[-self.history_length :]

    @staticmethod
    def _format_history(history: list[tuple[str, str]]) -> str:
        if not history:
            return "None"
        parts = []
        for index, (action, obs) in enumerate(history, start=1):
            parts.append(f"Step {index}: Action: {action} | Observation: {obs}")
        return "\n".join(parts)


HARNESSES.register(
    "alfworld",
    lambda config, **kwargs: ALFWorldHarness(
        config, kwargs["model_backend"], kwargs.get("generation_config", {})
    ),
)
