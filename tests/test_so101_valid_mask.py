from __future__ import annotations

import unittest
from unittest import TestCase

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only on no-torch interpreters.
    torch = None

if torch is not None:
    from physical_ai_agent.policies.so101_valid_mask import (
        execution_horizon_from_valid_probs,
        first_invalid_step,
        valid_labels_from_action_is_pad,
    )


@unittest.skipIf(torch is None, "torch is required for SO101 valid-mask tensor tests")
class SO101ValidMaskTest(TestCase):
    def test_action_is_pad_becomes_valid_labels(self) -> None:
        labels = valid_labels_from_action_is_pad(torch.tensor([[False, False, True, True]]))

        self.assertEqual(labels.tolist(), [[1.0, 1.0, 0.0, 0.0]])

    def test_first_invalid_step_requires_consecutive_low_probs(self) -> None:
        step = first_invalid_step([0.9, 0.2, 0.8, 0.4, 0.3, 0.1], threshold=0.5, consecutive=2)

        self.assertEqual(step, 3)

    def test_execution_horizon_never_returns_zero_steps(self) -> None:
        horizon, reason = execution_horizon_from_valid_probs(
            [0.1, 0.1, 0.9],
            max_horizon=15,
            threshold=0.5,
            consecutive=2,
        )

        self.assertEqual(horizon, 1)
        self.assertEqual(reason, "valid_mask_stop")

    def test_execution_horizon_uses_max_when_no_stop(self) -> None:
        horizon, reason = execution_horizon_from_valid_probs(
            [0.9, 0.8, 0.7],
            max_horizon=15,
            threshold=0.5,
            consecutive=2,
        )

        self.assertEqual(horizon, 15)
        self.assertEqual(reason, "max_horizon")
