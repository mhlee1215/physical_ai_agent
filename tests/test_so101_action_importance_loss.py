from __future__ import annotations

import unittest

import torch

from scripts.lerobot_train_so101_lightning import (
    _teacher_action_importance_weights,
    _terminal_valid_mask,
)


class SO101ActionImportanceLossTest(unittest.TestCase):
    def test_terminal_mask_uses_last_valid_steps_not_fixed_phase(self) -> None:
        action_is_pad = torch.tensor(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, False, False],
            ]
        )

        mask = _terminal_valid_mask(
            action_is_pad,
            batch_size=2,
            horizon=6,
            steps=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(mask.tolist(), [[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]])

    def test_teacher_importance_weights_follow_action_dynamics(self) -> None:
        actions = torch.zeros((1, 5, 6), dtype=torch.float32)
        actions[0, :, 0] = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.5])
        actions[0, :, 5] = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0])
        losses = torch.ones_like(actions)

        weights, metrics = _teacher_action_importance_weights(
            actions,
            losses=losses,
            actions_is_pad=None,
            delta_weight=1.0,
            gripper_transition_weight=2.0,
            terminal_steps=1,
            terminal_weight=1.5,
        )

        self.assertGreater(float(weights[0, 2, 0]), float(weights[0, 1, 0]))
        self.assertGreater(float(weights[0, 3, 5]), float(weights[0, 3, 0]))
        self.assertGreater(float(weights[0, 4, 0]), float(weights[0, 1, 0]))
        self.assertGreater(metrics["action_importance_weight_max"], 1.0)


if __name__ == "__main__":
    unittest.main()
