from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPTS_PATH = str(Path("scripts").resolve())
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from evaluate_so101_picklift_smolvla_policy import (  # noqa: E402
    _action_chunk_overlap_rmse,
    _resolve_action_chunk_inference_mode,
    _rtc_policy_predict_kwargs,
    _rtc_processed_overlap_rmse,
    _rtc_unbatched_chunk,
    _temporal_ensemble_action,
    _validate_rtc_settings,
)
from monitor_so101_training_dashboard import (  # noqa: E402
    _closed_loop_action_chunk_inference_args,
    _summarize_requery_boundary,
)


class SO101TemporalEnsembleTest(unittest.TestCase):
    def test_temporal_ensemble_blends_all_chunks_covering_current_step(self) -> None:
        old = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        new = np.asarray([[10.0], [11.0], [12.0]], dtype=float)

        action, source_count = _temporal_ensemble_action(
            [(0, old), (2, new)],
            step=2,
            decay=0.0,
        )

        self.assertEqual(source_count, 2)
        self.assertAlmostEqual(float(action[0]), 6.0)

    def test_overlap_rmse_compares_previous_tail_with_new_prefix(self) -> None:
        previous = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        new = np.asarray([[2.0], [4.0]], dtype=float)

        rmse = _action_chunk_overlap_rmse(
            previous_start=0,
            previous_chunk=previous,
            new_start=2,
            new_chunk=new,
            max_horizon=2,
        )

        self.assertAlmostEqual(float(rmse), np.sqrt(0.5))

    def test_boundary_summary_prefers_explicit_inference_marker(self) -> None:
        records = [
            {"action": [0.0, 0.0], "policy_inference": True},
            {"action": [1.0, 0.0], "policy_inference": False},
            {"action": [3.0, 0.0], "policy_inference": True, "requery_overlap_rmse": 0.25},
            {"action": [4.0, 0.0], "policy_inference": False},
        ]

        summary = _summarize_requery_boundary(records, n_action_steps=15)

        self.assertEqual(summary["requery_boundary_count"], 1)
        self.assertEqual(summary["non_boundary_step_count"], 2)
        self.assertAlmostEqual(float(summary["requery_overlap_rmse_mean"]), 0.25)

    def test_rtc_mode_requires_every_explicit_setting(self) -> None:
        with self.assertRaisesRegex(ValueError, "debug_maxlen"):
            _validate_rtc_settings(
                mode="rtc",
                prefix_attention_schedule="EXP",
                max_guidance_weight=10.0,
                execution_horizon=10,
                inference_delay=0,
                debug=False,
                debug_maxlen=None,
            )

    def test_rtc_raw_leftover_is_forwarded_without_postprocessing(self) -> None:
        raw_chunk = _rtc_unbatched_chunk(np.arange(24, dtype=np.float32).reshape(1, 4, 6))
        settings = {
            "prefix_attention_schedule": "EXP",
            "max_guidance_weight": 10.0,
            "execution_horizon": 3,
            "inference_delay": 0,
            "debug": False,
            "debug_maxlen": 100,
        }

        kwargs = _rtc_policy_predict_kwargs(
            previous_leftover=raw_chunk[2:],
            settings=settings,
        )

        self.assertEqual(tuple(kwargs["prev_chunk_left_over"].shape), (2, 6))
        self.assertEqual(kwargs["execution_horizon"], 3)
        self.assertEqual(kwargs["inference_delay"], 0)

    def test_rtc_gradient_correction_can_reenable_grad_under_no_grad(self) -> None:
        with torch.no_grad():
            with torch.enable_grad():
                value = torch.tensor(2.0, requires_grad=True)
                result = value.square()
                gradient = torch.autograd.grad(result, value)[0]

        self.assertAlmostEqual(float(gradient), 4.0)

    def test_rtc_overlap_compares_previous_processed_tail(self) -> None:
        previous = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        new = np.asarray([[1.0], [4.0], [8.0]], dtype=np.float32)

        rmse = _rtc_processed_overlap_rmse(
            previous_leftover=previous,
            new_chunk=new,
            max_horizon=2,
        )

        self.assertAlmostEqual(float(rmse), np.sqrt(2.0))

    def test_rtc_monitor_command_is_config_complete_and_exclusive(self) -> None:
        args = SimpleNamespace(
            closed_loop_inference_mode="rtc",
            closed_loop_temporal_ensemble=False,
            closed_loop_temporal_ensemble_decay=0.01,
            closed_loop_rtc_prefix_attention_schedule="EXP",
            closed_loop_rtc_max_guidance_weight=10.0,
            closed_loop_rtc_execution_horizon=10,
            closed_loop_rtc_inference_delay=0,
            closed_loop_rtc_debug=False,
            closed_loop_rtc_debug_maxlen=100,
        )

        command = _closed_loop_action_chunk_inference_args(args)

        self.assertIn("rtc", command)
        self.assertIn("--no-temporal-ensemble", command)
        self.assertIn("--no-rtc-debug", command)
        self.assertNotIn("--temporal-ensemble", command)

    def test_explicit_rtc_conflicts_with_legacy_temporal_ensemble_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _resolve_action_chunk_inference_mode(
                "rtc",
                legacy_temporal_ensemble=True,
            )


if __name__ == "__main__":
    unittest.main()
