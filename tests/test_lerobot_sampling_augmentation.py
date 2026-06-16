from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal stdlib test envs.
    torch = None  # type: ignore[assignment]

from physical_ai_agent.lerobot_sampling_augmentation import (
    SamplingAugmentationConfig,
    SamplingAugmentedDataset,
    augment_state_tensor,
)


class FakeDataset:
    num_frames = 1
    num_episodes = 1
    episodes = None

    def __init__(self) -> None:
        self.meta = object()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        return {
            "observation.state": torch.zeros(6, dtype=torch.float32),
            "action": torch.arange(6, dtype=torch.float32),
        }


class SamplingAugmentationTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_state_jitter_happens_on_sample_and_keeps_action_clean(self) -> None:
        torch.manual_seed(123)
        dataset = SamplingAugmentedDataset(
            FakeDataset(),
            SamplingAugmentationConfig(state_jitter_std=0.1, state_jitter_arm_only=True, enabled=True),
        )

        item = dataset[0]

        self.assertFalse(torch.equal(item["observation.state"], torch.zeros(6)))
        self.assertEqual(float(item["observation.state"][5]), 0.0)
        self.assertTrue(torch.equal(item["action"], torch.arange(6, dtype=torch.float32)))
        self.assertEqual(dataset.num_frames, 1)

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_state_dropout_can_keep_gripper_channel(self) -> None:
        state = torch.ones(6, dtype=torch.float32)

        augmented = augment_state_tensor(
            state,
            SamplingAugmentationConfig(
                state_dropout_prob=1.0,
                state_dropout_keep_gripper=True,
                enabled=True,
            ),
        )

        self.assertTrue(torch.equal(augmented[:5], torch.zeros(5)))
        self.assertEqual(float(augmented[5]), 1.0)

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_action_dropout_happens_on_sample(self) -> None:
        dataset = SamplingAugmentedDataset(
            FakeDataset(),
            SamplingAugmentationConfig(action_dropout_prob=1.0, enabled=True),
        )

        item = dataset[0]

        self.assertTrue(torch.equal(item["action"], torch.zeros(6, dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
