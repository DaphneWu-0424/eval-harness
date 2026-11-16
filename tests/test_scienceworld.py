import asyncio
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from eval_harness.backends.fake import FakeBackend
from eval_harness.core.models import EvaluationResult, TaskSample, Trajectory
from eval_harness.datasets.scienceworld import ScienceWorldManifest
from eval_harness.environments.scienceworld import ScienceWorldAdapter
from eval_harness.evaluators.scienceworld import ScienceWorldEvaluator
from eval_harness.executors.agent import AgentExecutor
from eval_harness.harnesses.scienceworld import ScienceWorldHarness, parse_action
from eval_harness.metrics import compute_metrics
from eval_harness.reports.generate import render_text_report


def sample(name="boil", seed=23):
    episode = dict(episode_id=f"{name}:{seed}", task_name=name, task_type=name, variation_id=seed, env_seed=seed)
    return TaskSample(episode["episode_id"], episode, source=name, metadata={"task_type": name})


class FakeWorker:
    def __init__(self, horizon, scores=None):
        self.horizon, self.steps = horizon, 0
        self.scores = scores or [5, -100]
        self.closed = False
    def info(self, score):
        return dict(scienceworld_score=score, taskDesc="Boil water.", look="kitchen", inventory="empty",
                    action_templates=["look", "move OBJ to OBJ"], admissible_commands=["look"], won=score==100)
    def reset(self, episode):
        random.seed(episode["env_seed"])
        self.steps = 0
        return "initial", self.info(0)
    def step(self, action):
        self.steps += 1
        raw = self.scores[min(self.steps-1, len(self.scores)-1)]
        return f"observation-{self.steps}", 0.05 if self.steps==1 else 0.0, raw < 0 or raw==100 or self.steps>=self.horizon, self.info(raw)
    def close(self):
        self.closed = True


class ScienceWorldTests(unittest.IsolatedAsyncioTestCase):
    async def test_negative_native_score_preserves_raw_retained_and_return(self):
        workers=[]
        def factory(horizon):
            worker=FakeWorker(horizon);workers.append(worker);return worker
        env = ScienceWorldAdapter({"max_steps":50}, worker_factory=factory)
        backend=FakeBackend({"responses":["<think>Inspect.</think><action>look</action>"]})
        harness=ScienceWorldHarness({},backend,{})
        trajectory=await AgentExecutor(backend,env,harness,{},50).run(sample(),0)
        self.assertEqual(trajectory.status,"completed")
        self.assertEqual(trajectory.metadata["raw_score"],-100)
        self.assertEqual(trajectory.metadata["retained_score"],5)
        self.assertAlmostEqual(trajectory.metadata["episode_return"],0.05)
        self.assertEqual(trajectory.metadata["termination"],"native_failure")
        self.assertTrue(workers[0].closed)
        result=await ScienceWorldEvaluator({}).evaluate(sample(),trajectory)
        self.assertEqual(result.reward,0)
        self.assertEqual(result.scores["task_score"],0)
        self.assertEqual(result.scores["retained_score"],5)
        self.assertTrue(any(e.metadata.get("raw_score")==5 for e in trajectory.events))
        self.assertEqual(sum(e.type=="model_request" for e in trajectory.events),2)

    async def test_concurrent_episode_context_and_rng_are_isolated(self):
        env=ScienceWorldAdapter({"max_steps":2}, worker_factory=lambda h:FakeWorker(h,[0]))
        before=random.getstate()
        async def episode(seed):
            await env.reset(sample(seed=seed))
            await asyncio.sleep(0)
            step=await env.step("look")
            await env.close()
            return step.metadata["trajectory_metadata"]["env_seed"]
        seeds=await asyncio.gather(episode(23),episode(24))
        self.assertEqual(seeds,[23,24])
        self.assertEqual(random.getstate(),before)

    async def test_prompt_matches_colleague_template_and_history(self):
        env=ScienceWorldAdapter({"max_steps":50},worker_factory=lambda h:FakeWorker(h,[0]))
        observation=await env.reset(sample())
        harness=ScienceWorldHarness({},FakeBackend({}),{})
        trajectory=Trajectory("id",0,[],None,"running")
        prompt=harness.build_prompt(trajectory,observation)
        expected=("Task: Boil water.\nRecent interaction history:\n\nCurrent observation: initial\nCurrent room: kitchen\nInventory: empty\n"
                  "Admissible action templates:\n 'look'\n 'move OBJ to OBJ'\nRespond in <think>reasoning</think><action>one action</action> format. "
                  "Fill template object slots using objects visible in the current room or inventory.")
        self.assertEqual(prompt,expected)
        await env.close()

    async def test_score_100_is_the_only_completion(self):
        evaluator=ScienceWorldEvaluator({})
        for raw,success in [(0,False),(99,False),(100,True),(-100,False)]:
            trajectory=Trajectory("id",0,[],None,"completed",metadata={"raw_score":raw})
            result=await evaluator.evaluate(sample(),trajectory)
            self.assertEqual(result.labels["success"],success)
        trajectory.status="failed"
        self.assertIsNone((await evaluator.evaluate(sample(),trajectory)).reward)

    async def test_prompt_overflow_fails_without_executing_environment_action(self):
        backend=FakeBackend({"responses":["<think>x</think><action>look</action>"]})
        env=ScienceWorldAdapter({},worker_factory=lambda h:FakeWorker(h))
        harness=ScienceWorldHarness({"max_prompt_tokens":1},backend,{})
        trajectory=await AgentExecutor(backend,env,harness,{},50).run(sample(),0)
        self.assertEqual(trajectory.status,"failed")
        self.assertEqual(trajectory.metadata["episode_length"],0)
        self.assertIn("frozen budget",trajectory.error)

    def test_frozen_manifest_counts_and_seed_validation(self):
        for name,size in [("scienceworld-claim.json",271),("scienceworld-in-run.json",149)]:
            samples=ScienceWorldManifest({"path":f"configs/manifests/{name}"}).load()
            self.assertEqual(len(samples),size)
            self.assertEqual(len({s.source for s in samples}),30)
        data=json.loads(Path('configs/manifests/scienceworld-in-run.json').read_text())
        data['episodes'][0]['env_seed']+=1
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'seed'):
                ScienceWorldManifest({'path':str(path)}).load()

    def test_macro_score_is_not_episode_mean_and_failure_is_not_zero(self):
        samples=[sample("a",0),sample("a",1),sample("b",0)]
        evaluations=[EvaluationResult(s.sample_id,0,'completed',float(v==100),{'task_score':v,'success':float(v==100)}, {},{},'science') for s,v in zip(samples,[100,0,100])]
        trajectories=[Trajectory(s.sample_id,0,[],None,'completed') for s in samples]
        result=compute_metrics(samples,trajectories,evaluations,[{'type':'scienceworld'}],1)
        self.assertEqual(result['scienceworld_score'],75)
        self.assertEqual(result['accuracy'],0.75)
        self.assertIn('75.00/100',render_text_report('science',result))
        evaluations.pop()
        result=compute_metrics(samples,trajectories,evaluations,[{'type':'scienceworld'}],1)
        self.assertIsNone(result['scienceworld_score'])
        self.assertIsNone(result['accuracy'])

    def test_format_validity_does_not_require_a_concrete_action_pool(self):
        self.assertEqual(parse_action('<think>x</think><action>MOVE cup TO table</action>'),('move cup to table',True))
        self.assertFalse(parse_action('<action>look</action>')[1])


if __name__=='__main__':
    unittest.main()
