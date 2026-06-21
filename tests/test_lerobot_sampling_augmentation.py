from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal stdlib test envs.
    torch = None  # type: ignore[assignment]

from physical_ai_agent.lerobot_sampling_augmentation import (
    SamplingAugmentationConfig,
    SamplingAugmentedDataset,
    augment_batch_on_device,
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
    def test_patch_mask_ratio_masks_patch_grid(self) -> None:
        batch = {
            "observation.images.camera1": torch.ones((2, 3, 256, 256), dtype=torch.float32),
        }

        augment_batch_on_device(
            batch,
            SamplingAugmentationConfig(
                image_patch_mask_ratio=0.25,
                gpu_image_augmentation=True,
                enabled=True,
            ),
        )

        image = batch["observation.images.camera1"]
        self.assertGreater(int((image == 0.0).sum().item()), 0)
        self.assertLess(int((image == 0.0).sum().item()), image.numel())

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_affine_augmentation_runs_on_input_device(self) -> None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        image = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=device)
        image[:, :, 8:24, 8:24] = 1.0
        batch = {"observation.images.camera1": image.clone()}

        torch.manual_seed(123)
        augment_batch_on_device(
            batch,
            SamplingAugmentationConfig(
                image_affine_degrees=5.0,
                image_affine_translate=0.05,
                gpu_image_augmentation=True,
                enabled=True,
            ),
        )

        augmented = batch["observation.images.camera1"]
        self.assertEqual(augmented.device.type, device.type)
        self.assertEqual(tuple(augmented.shape), tuple(image.shape))
        self.assertFalse(torch.equal(augmented.cpu(), image.cpu()))


if __name__ == "__main__":
    unittest.main()
