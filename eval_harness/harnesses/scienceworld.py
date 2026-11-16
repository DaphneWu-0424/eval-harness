from __future__ import annotations

import re

from eval_harness.core.interfaces import AgentHarness
from eval_harness.core.models import TrajectoryEvent
from eval_harness.core.registry import HARNESSES


def parse_action(text):
    """Match the colleague's format-only ALFWorld projection used by ScienceWorld."""
    original = text or ""
    lowered = original.lower()
    start, end = lowered.find("<action>"), lowered.find("</action>")
    if start == -1 or end == -1:
        return lowered[-30:], False
    action = lowered[start + len("<action>"):end].strip()
    valid = "<think>" in original and "</think>" in original and re.search(r"[\u4e00-\u9fff]", original) is None
    return action, valid


class ScienceWorldHarness(AgentHarness):
    def __init__(self, config, model_backend, generation_config):
        self.backend = model_backend
        self.generation = dict(generation_config)
        self.history_length = int(config.get("history_length", 4))
        self.max_prompt_tokens = int(config.get("max_prompt_tokens", 4096))
        if self.history_length < 0 or self.max_prompt_tokens < 1:
            raise ValueError("ScienceWorld history length must be nonnegative and prompt budget positive")

    def build_prompt(self, trajectory, observation):
        history, previous = [], None
        for event in trajectory.events:
            if event.type == "observation":
                previous = event.content
            elif event.type == "action" and previous is not None:
                history.append((previous, event.content))
                previous = None
        recent = history[-self.history_length:] if self.history_length else []
        start = len(history) - len(recent) + 1
        history_text = "\n".join(f"[Observation {start+i}: '{obs}', Action {start+i}: '{action}']" for i, (obs, action) in enumerate(recent))
        templates = "\n ".join(repr(x) for x in observation["action_templates"])
        return (f"Task: {observation['task_description']}\n"
                f"Recent interaction history:\n{history_text}\n"
                f"Current observation: {observation['observation']}\n"
                f"Current room: {observation['look']}\n"
                f"Inventory: {observation['inventory']}\n"
                f"Admissible action templates:\n {templates}\n"
                'Respond in <think>reasoning</think><action>one action</action> format. '
                'Fill template object slots using objects visible in the current room or inventory.')

    async def next_action(self, trajectory, observation):
        messages = [{"role": "user", "content": self.build_prompt(trajectory, observation)}]
        trajectory.events.append(TrajectoryEvent(type="model_request", role="user", content=messages))
        response = await self.backend.generate(messages, self.generation)
        prompt_tokens = response.usage.get("prompt_tokens")
        if prompt_tokens is None:
            raise ValueError("ScienceWorld prompt limit cannot be verified: missing prompt_tokens usage")
        if prompt_tokens > self.max_prompt_tokens:
            raise ValueError(f"ScienceWorld prompt exceeds frozen budget: {prompt_tokens} > {self.max_prompt_tokens}; no action executed")
        action, valid = parse_action(response.content)
        return {"action": action, "is_action_valid": valid, "content": response.content,
                "reasoning": response.reasoning, "usage": response.usage, "finish_reason": response.finish_reason}


HARNESSES.register("scienceworld", lambda config, **kwargs: ScienceWorldHarness(
    config, kwargs["model_backend"], kwargs.get("generation_config", {})))
