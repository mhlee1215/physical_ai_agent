from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from physical_ai_agent.so101_tensorboard_loss_layout import (
    add_so101_loss_custom_scalars,
    should_log_repeated_scalar_metric,
    so101_loss_custom_scalars_layout,
)


class SO101TensorBoardLossLayoutTest(unittest.TestCase):
    def test_layout_collects_real_losses_without_config_or_debug_noise(self) -> None:
        layout = so101_loss_custom_scalars_layout()
        patterns = [
            pattern
            for _chart_type, chart_patterns in layout["Loss"].values()
            for pattern in chart_patterns
        ]

        included = (
            "train/loss",
            "val/loss",
            "train/action_loss",
            "train/action_requery_consistency_loss",
            "train/datasets/approach/loss",
            "val/datasets/grip_lift/loss",
            "val/datasets/grip_lift/valid_mask_loss",
        )
        excluded = (
            "important/train_loss",
            "important/val_loss",
            "extra/train/losses_after_forward",
            "train/action_smoothness_loss_weight",
            "train/loss_prefix_steps",
        )

        for tag in included:
            self.assertTrue(any(re.fullmatch(pattern, tag) for pattern in patterns), tag)
        for tag in excluded:
            self.assertFalse(any(re.fullmatch(pattern, tag) for pattern in patterns), tag)

    def test_repeated_scalar_filter_drops_config_and_old_debug_values(self) -> None:
        self.assertFalse(should_log_repeated_scalar_metric("losses_after_forward"))
        self.assertFalse(should_log_repeated_scalar_metric("action_smoothness_loss_weight"))
        self.assertFalse(should_log_repeated_scalar_metric("loss_prefix_steps"))
        self.assertFalse(should_log_repeated_scalar_metric("action_overlap_consistency_horizon"))
        self.assertTrue(should_log_repeated_scalar_metric("action_loss"))
        self.assertTrue(should_log_repeated_scalar_metric("action_smoothness_loss"))
        self.assertTrue(should_log_repeated_scalar_metric("valid_mask_boundary_mae_steps"))
        self.assertTrue(should_log_repeated_scalar_metric("valid_mask_accuracy"))

    def test_layout_is_written_as_tensorboard_plugin_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            with SummaryWriter(log_dir=str(log_dir)) as writer:
                writer.add_scalar("train/loss", 0.5, global_step=1)
                add_so101_loss_custom_scalars(writer)

            accumulator = EventAccumulator(str(log_dir))
            accumulator.Reload()

            self.assertIn("custom_scalars__config__", accumulator.Tags().get("tensors", []))

    def test_lightning_training_registers_layout_and_filters_repeated_noise(self) -> None:
        source = Path("scripts/lerobot_train_so101_lightning.py").read_text(encoding="utf-8")

        self.assertIn("add_so101_loss_custom_scalars(logger.experiment)", source)
        self.assertEqual(source.count("not should_log_repeated_scalar_metric(key)"), 2)


if __name__ == "__main__":
    unittest.main()
