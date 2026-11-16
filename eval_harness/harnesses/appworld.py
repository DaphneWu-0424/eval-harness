from eval_harness.core.interfaces import AgentHarness
from eval_harness.core.models import TrajectoryEvent
from eval_harness.core.registry import HARNESSES
from .appworld_prompts import APPWORLD_TEMPLATE, APPWORLD_TEMPLATE_NO_HIS


def parse_code(text):
    start, end = text.find('<code>'), text.find('</code>')
    if start == -1 or end == -1:
        return text[-100:], False
    return text[start + 6:end], '<think>' in text and '</think>' in text


class AppWorldHarness(AgentHarness):
    def __init__(self, config, model_backend, generation_config):
        self.backend = model_backend
        self.generation = dict(generation_config)
        self.history_length = int(config.get('history_length', 4))
        if self.history_length < 1:
            raise ValueError('AppWorld history_length must be positive')

    def build_prompt(self, trajectory, observation):
        pairs, action = [], None
        for event in trajectory.events:
            if event.type == 'action':
                action = event.content
            elif event.type == 'observation' and action is not None:
                pairs.append((action, event.content))
                action = None
        supervisor = observation['supervisor']
        values = {f'supervisor_{key}': supervisor[key]
                  for key in ('first_name', 'last_name', 'email', 'phone_number')}
        values['task_description'] = observation['task_description']
        if not pairs:
            return APPWORLD_TEMPLATE_NO_HIS.format(**values)
        recent = pairs[-self.history_length:]
        history = ''.join(f'\nCode {i}: \n{code}\n\nResult {i}: \n{obs}\n'
                          for i, (code, obs) in enumerate(recent, len(pairs)-len(recent)+1))
        if len(history) > 10000:
            history = '... ' + history[-10000:]
        return APPWORLD_TEMPLATE.format(**values, step_count=len(pairs), history_length=len(recent),
                                        action_history=history.strip(), current_step=len(pairs)+1,
                                        current_observation=observation['observation'])

    async def next_action(self, trajectory, observation):
        messages = [{'role': 'user', 'content': self.build_prompt(trajectory, observation)}]
        trajectory.events.append(TrajectoryEvent(type='model_request', role='user', content=messages))
        response = await self.backend.generate(messages, self.generation)
        code, valid = parse_code(response.content or '')
        return {'action': code, 'is_action_valid': valid, 'content': response.content,
                'reasoning': response.reasoning, 'usage': response.usage,
                'finish_reason': response.finish_reason}


HARNESSES.register('appworld', lambda config, **kwargs: AppWorldHarness(
    config, kwargs['model_backend'], kwargs.get('generation_config', {})))
