from unittest import TestCase

from physical_ai_agent.policies.lerobot_policy_runner import (
    LeRobotPolicyRunner,
    _align_processor_stats_to_declared_features,
)


class _FakePolicy:
    def __init__(self) -> None:
        self.observations = []

    def select_action(self, observation):
        self.observations.append(observation)
        return "raw_action"


class _Step:
    def __init__(self, name, output):
        self.name = name
        self.output = output
        self.inputs = []

    def __call__(self, value):
        self.inputs.append(value)
        return self.output


class _Feature:
    def __init__(self, shape):
        self.shape = shape


class _Processor:
    def __init__(self, steps):
        self.steps = steps


class LeRobotPolicyRunnerTest(TestCase):
    def test_select_action_applies_lerobot_processor_order(self) -> None:
        policy = _FakePolicy()
        env_preprocessor = _Step("env_pre", {"env": "preprocessed"})
        preprocessor = _Step("pre", {"policy": "batch"})
        postprocessor = _Step("post", "unnormalized_action")

        def env_postprocessor(transition):
            return {"action": f"env_{transition['action']}"}

        runner = LeRobotPolicyRunner(
            policy=policy,
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )

        action = runner.select_action({"raw": "observation"})

        self.assertEqual(action, "env_unnormalized_action")
        self.assertEqual(env_preprocessor.inputs, [{"raw": "observation"}])
        self.assertEqual(preprocessor.inputs, [{"env": "preprocessed"}])
        self.assertEqual(policy.observations, [{"policy": "batch"}])
        self.assertEqual(postprocessor.inputs, ["raw_action"])

    def test_align_processor_stats_trims_to_declared_feature_shape(self) -> None:
        step = _Step("normalizer", {})
        step.features = {"observation.state": _Feature((6,))}
        step.stats = {
            "observation.state": {
                "mean": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "std": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        }

        _align_processor_stats_to_declared_features(_Processor([step]))

        self.assertEqual(step.stats["observation.state"]["mean"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(step.stats["observation.state"]["std"], [1, 1, 1, 1, 1, 1])

    def test_select_action_uses_policy_processors_without_env_processors(self) -> None:
        policy = _FakePolicy()
        preprocessor = _Step("pre", {"policy": "batch"})
        postprocessor = _Step("post", "unnormalized_action")

        runner = LeRobotPolicyRunner(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

        action = runner.select_action({"raw": "observation"})

        self.assertEqual(action, "unnormalized_action")
        self.assertEqual(preprocessor.inputs, [{"raw": "observation"}])
        self.assertEqual(policy.observations, [{"policy": "batch"}])
        self.assertEqual(postprocessor.inputs, ["raw_action"])
