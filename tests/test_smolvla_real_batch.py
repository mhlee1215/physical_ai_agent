from unittest import TestCase
from unittest.mock import patch

from physical_ai_agent.policies.smolvla_real import _build_batch_for_policy


class _FakeFeature:
    shape = (3, 16, 16)


class _FakeStateFeature:
    shape = (6,)


class _FakePolicyConfig:
    robot_state_feature = _FakeStateFeature()
    image_features = {
        "observation.images.camera1": _FakeFeature(),
        "observation.images.camera2": _FakeFeature(),
        "observation.images.camera3": _FakeFeature(),
    }
    device = "cpu"
    vlm_model_name = "fake-vlm"
    pad_language_to = "longest"
    tokenizer_max_length = 48


class _FakePolicy:
    config = _FakePolicyConfig()


class _FakeTokenizer:
    def __call__(self, text, *, padding, truncation, max_length, return_tensors):
        import torch

        self.last_call = {
            "text": text,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "return_tensors": return_tensors,
        }
        return {
            "input_ids": torch.tensor([[11, 22, 33, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        }


class SmolVLARealBatchTest(TestCase):
    def test_so101_policy_batch_maps_camera1_to_egocentric_and_camera2_to_wrist(self) -> None:
        import numpy as np

        camera_pixels = {
            "egocentric_cam": np.full((8, 8, 3), 255, dtype=np.uint8),
            "wrist_cam": np.full((8, 8, 3), 127, dtype=np.uint8),
        }

        batch, mapping = _build_batch_for_policy(
            _FakePolicy(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            camera_pixels,
            instruction=None,
            local_files_only=True,
        )

        self.assertEqual(mapping["observation.images.camera1"], "egocentric_cam")
        self.assertEqual(mapping["observation.images.camera2"], "wrist_cam")
        self.assertEqual(mapping["observation.images.camera3"], "wrist_cam")
        self.assertGreater(float(batch["observation.images.camera1"].mean()), float(batch["observation.images.camera2"].mean()))

    def test_instruction_is_tokenized_when_provided(self) -> None:
        fake_tokenizer = _FakeTokenizer()

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer) as loader:
            batch, _mapping = _build_batch_for_policy(
                _FakePolicy(),
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                {},
                instruction="Pick up the green Android figure.",
                local_files_only=True,
            )

        loader.assert_called_once_with("fake-vlm", local_files_only=True)
        self.assertEqual(fake_tokenizer.last_call["text"], ["Pick up the green Android figure.\n"])
        self.assertEqual(fake_tokenizer.last_call["padding"], "longest")
        self.assertEqual(fake_tokenizer.last_call["max_length"], 48)
        self.assertEqual(tuple(batch["observation.language.tokens"].shape), (1, 4))
        self.assertEqual(int(batch["observation.language.attention_mask"].sum().item()), 3)

    def test_placeholder_language_tokens_are_preserved_without_instruction(self) -> None:
        batch, _mapping = _build_batch_for_policy(
            _FakePolicy(),
            [1.0, 2.0, 3.0],
            {},
            instruction=None,
            local_files_only=True,
        )

        self.assertEqual(tuple(batch["observation.language.tokens"].shape), (1, 4))
        self.assertEqual(int(batch["observation.language.attention_mask"].sum().item()), 4)
