from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

SCRIPTS_PATH = str(Path("scripts").resolve())
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from evaluate_so101_picklift_smolvla_policy import (  # noqa: E402
    _action_chunk_overlap_rmse,
    _temporal_ensemble_action,
)
from monitor_so101_training_dashboard import _summarize_requery_boundary  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
