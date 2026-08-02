from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from scripts.lerobot_train_so101_lightning import (
    _add_valid_mask_auxiliary_loss,
    _aligned_requery_noise,
    _latest_retained_checkpoint,
    _predicted_action_circular_loss,
    _predicted_action_overlap_consistency_loss,
    _predicted_action_requery_consistency_loss,
    _smolvla_flow_model,
    _teacher_action_importance_weights,
    _terminal_valid_mask,
)


class SO101ActionImportanceLossTest(unittest.TestCase):
    def test_valid_mask_head_uses_predicted_chunk_not_teacher_chunk(self) -> None:
        class CapturingHead:
            config = SimpleNamespace(threshold=0.5, consecutive_invalid=2)

            def __init__(self) -> None:
                self.action = None

            def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                del state
                self.action = action
                return torch.zeros(action.shape[:2], dtype=action.dtype, device=action.device)

        head = CapturingHead()
        teacher = torch.zeros((1, 4, 2), dtype=torch.float32)
        predicted = torch.full((1, 4, 2), 0.75, dtype=torch.float32)
        total, metrics = _add_valid_mask_auxiliary_loss(
            (torch.tensor(1.0, requires_grad=True), {}),
            valid_mask_head=head,
            batch={
                "observation.state": torch.zeros((1, 2), dtype=torch.float32),
                "action": teacher,
                "action_is_pad": torch.tensor([[False, False, True, True]]),
            },
            weight=0.2,
            action_chunk=predicted,
        )

        self.assertTrue(torch.equal(head.action, predicted))
        self.assertIn("valid_mask_loss", metrics)
        self.assertGreater(float(total.detach()), 1.0)

    def test_valid_mask_head_rejects_silent_teacher_action_fallback(self) -> None:
        class Head:
            config = SimpleNamespace(threshold=0.5, consecutive_invalid=2)

        with self.assertRaisesRegex(ValueError, "predicted action chunk"):
            _add_valid_mask_auxiliary_loss(
                (torch.tensor(1.0), {}),
                valid_mask_head=Head(),
                batch={
                    "observation.state": torch.zeros((1, 2), dtype=torch.float32),
                    "action": torch.zeros((1, 4, 2), dtype=torch.float32),
                    "action_is_pad": torch.zeros((1, 4), dtype=torch.bool),
                },
                weight=0.2,
            )

    def test_latest_retained_checkpoint_uses_saved_step_not_alias_priority(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, step in (
                ("best_closed_loop", 2586),
                ("best_train_loss", 5172),
                ("best_val_loss", 6465),
            ):
                state = root / name / "training_state"
                state.mkdir(parents=True)
                (state / "training_step.json").write_text(f'{{"step": {step}}}\n', encoding="utf-8")

            self.assertEqual(_latest_retained_checkpoint(root), root / "best_val_loss")

    def test_overlap_consistency_uses_future_teacher_action(self) -> None:
        class Policy:
            @staticmethod
            def prepare_action(batch: dict[str, torch.Tensor]) -> torch.Tensor:
                return batch["action"]

        current = torch.zeros((1, 5, 2), dtype=torch.float32)
        current[:, 2:4] = torch.tensor([[[0.25, -0.5], [0.5, -0.25]]])
        future_teacher = torch.tensor(
            [[[0.25, -0.5], [0.5, -0.25], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
        )

        loss, metrics = _predicted_action_overlap_consistency_loss(
            Policy(),
            overlap_batch={
                "action": future_teacher,
                "action_is_pad": torch.zeros((1, 5), dtype=torch.bool),
            },
            current_action_hat=current,
            current_actions_is_pad=torch.zeros((1, 5), dtype=torch.bool),
            overlap_valid=torch.tensor([True]),
            offset=2,
            horizon=2,
        )

        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertEqual(metrics["action_overlap_consistency_target"], "teacher")

    def test_smolvla_flow_model_resolves_through_peft_wrapper(self) -> None:
        class FlowModel:
            def sample_noise(self) -> None:
                return None

        class BasePolicy:
            model = FlowModel()

        class PeftPolicy:
            def get_base_model(self) -> BasePolicy:
                return BasePolicy()

        self.assertIsInstance(_smolvla_flow_model(BasePolicy()), FlowModel)
        self.assertIsInstance(_smolvla_flow_model(PeftPolicy()), FlowModel)

    def test_requery_consistency_compares_current_tail_with_future_prefix(self) -> None:
        current = torch.zeros((1, 6, 2), dtype=torch.float32)
        future = torch.zeros((1, 6, 2), dtype=torch.float32)
        current[:, 2:5] = torch.tensor([[[0.2, -0.3], [0.4, -0.1], [0.1, 0.5]]])
        future[:, :3] = current[:, 2:5]

        loss, metrics = _predicted_action_requery_consistency_loss(
            current_action_hat=current,
            future_action_hat=future,
            current_actions_is_pad=torch.zeros((1, 6), dtype=torch.bool),
            future_actions_is_pad=torch.zeros((1, 6), dtype=torch.bool),
            overlap_valid=torch.tensor([True]),
            offset=2,
            horizon=3,
        )

        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertEqual(metrics["action_requery_consistency_target"], "future_observation_prediction")
        self.assertEqual(metrics["action_requery_consistency_pairs"], 3)

    def test_requery_consistency_detaches_future_prediction_target(self) -> None:
        current = torch.zeros((1, 4, 1), dtype=torch.float32, requires_grad=True)
        future = torch.ones((1, 4, 1), dtype=torch.float32, requires_grad=True)

        loss, _metrics = _predicted_action_requery_consistency_loss(
            current_action_hat=current,
            future_action_hat=future,
            current_actions_is_pad=None,
            future_actions_is_pad=None,
            overlap_valid=torch.tensor([True]),
            offset=1,
            horizon=2,
        )

        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertIsNone(future.grad)

    def test_requery_noise_aligns_current_tail_with_future_prefix(self) -> None:
        class Model:
            @staticmethod
            def sample_noise(shape, device):
                return torch.full(shape, 9.0, device=device)

        current_noise = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        future_actions = torch.zeros((1, 6, 2), dtype=torch.float32)

        aligned = _aligned_requery_noise(
            model=Model(),
            future_actions=future_actions,
            current_noise=current_noise,
            offset=2,
            horizon=3,
        )

        self.assertTrue(torch.equal(aligned[:, :3], current_noise[:, 2:5]))
        self.assertTrue(torch.equal(aligned[:, 3:], torch.full((1, 3, 2), 9.0)))

    def test_wrist_roll_circular_loss_wraps_at_full_turn(self) -> None:
        target = torch.zeros((1, 2, 6), dtype=torch.float32)
        prediction = target.clone()
        prediction[:, :, 4] = 2.0 * torch.pi

        loss = _predicted_action_circular_loss(
            prediction,
            target,
            actions_is_pad=None,
            joint_index=4,
            period_normalized=2.0 * torch.pi,
        )

        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_wrist_roll_circular_loss_masks_padded_steps(self) -> None:
        target = torch.zeros((1, 2, 6), dtype=torch.float32)
        prediction = target.clone()
        prediction[0, 1, 4] = torch.pi

        loss = _predicted_action_circular_loss(
            prediction,
            target,
            actions_is_pad=torch.tensor([[False, True]]),
            joint_index=4,
            period_normalized=2.0 * torch.pi,
        )

        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

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
