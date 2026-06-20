from unittest import TestCase
from tempfile import TemporaryDirectory


class SO101VisualRLPolicyTest(TestCase):
    def test_visual_actor_critic_forward_and_act(self) -> None:
        try:
            import gymnasium as gym
            import numpy as np
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"visual RL policy dependencies are not available: {exc}")

        from physical_ai_agent.policies.so101_visual_actor_critic import (
            make_so101_visual_actor_critic,
        )

        observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(low=0, high=255, shape=(3, 32, 32), dtype=np.uint8),
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        model = make_so101_visual_actor_critic(
            observation_space=observation_space,
            action_space=action_space,
        )
        observation = {
            "image": torch.zeros((2, 3, 32, 32), dtype=torch.uint8),
            "state": torch.zeros((2, 6), dtype=torch.float32),
        }

        output = model(observation)
        action_packet = model.act(observation, deterministic=True)

        self.assertEqual(tuple(output.action_mean.shape), (2, 6))
        self.assertEqual(tuple(output.value.shape), (2,))
        self.assertEqual(tuple(action_packet["action"].shape), (2, 6))
        self.assertTrue(bool(torch.all(action_packet["action"] <= 1.0)))
        self.assertTrue(bool(torch.all(action_packet["action"] >= -1.0)))

    def test_discounted_returns(self) -> None:
        from scripts.so101_visual_rl_policy_smoke import _discounted_returns

        self.assertEqual(_discounted_returns([1.0, 2.0, 3.0], gamma=0.5), [2.75, 3.5, 3.0])

    def test_visual_actor_critic_checkpoint_round_trip(self) -> None:
        try:
            import gymnasium as gym
            import numpy as np
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"visual RL policy dependencies are not available: {exc}")

        from physical_ai_agent.policies.so101_visual_actor_critic import (
            load_so101_visual_actor_critic_checkpoint,
            make_so101_visual_actor_critic,
            save_so101_visual_actor_critic_checkpoint,
        )

        observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(low=0, high=255, shape=(3, 32, 32), dtype=np.uint8),
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        model = make_so101_visual_actor_critic(
            observation_space=observation_space,
            action_space=action_space,
        )
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = __import__("pathlib").Path(tmpdir) / "policy.pt"
            save_so101_visual_actor_critic_checkpoint(
                path=checkpoint_path,
                model=model,
                observation_space=observation_space,
                action_space=action_space,
                metadata={"config": {"camera_name": "wrist_cam"}},
            )
            loaded, metadata = load_so101_visual_actor_critic_checkpoint(checkpoint_path)

        observation = {
            "image": torch.zeros((1, 3, 32, 32), dtype=torch.uint8),
            "state": torch.zeros((1, 6), dtype=torch.float32),
        }
        self.assertEqual(metadata["config"]["camera_name"], "wrist_cam")
        self.assertEqual(tuple(loaded(observation).action_mean.shape), (1, 6))
