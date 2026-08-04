from __future__ import annotations

import copy
import unittest

from scripts.render_mycobot280_policy_progression_figure import (
    build_progression_series,
)


def _summary() -> dict:
    return {
        "operation": "summarize_mycobot280_policy_claims",
        "status": "passed",
        "evaluation_contract": {
            "candidate_selection": "direct_unfiltered_draws_without_teacher_rejection"
        },
        "matched_schedule_rows": {
            "base": {
                "episodes": 11,
                "strict_successes": 0,
                "pickup_hold_successes": 0,
            },
            "deterministic_finetuned": {"strict_successes": 3},
            "randomized_finetuned": {"strict_successes": 5},
        },
        "fresh_randomized_multiseed": {
            "training_seeds": [20260731, 20260732, 20260733],
            "deterministic": {
                "episodes": 33,
                "strict_successes_per_training_seed": [3, 1, 0],
            },
            "randomized": {
                "episodes": 33,
                "strict_successes_per_training_seed": [5, 5, 4],
            },
            "positive_training_seed_count": 3,
            "exact_two_sided_sign_test_p": 0.25,
        },
    }


class PolicyProgressionFigureTests(unittest.TestCase):
    def test_builds_shared_base_and_three_training_pairs(self) -> None:
        series = build_progression_series(_summary())

        self.assertEqual(series["shared_base"]["model_count"], 1)
        self.assertEqual(series["shared_base"]["strict_successes"], 0)
        self.assertEqual(len(series["training_pairs"]), 3)
        self.assertEqual(
            [row["deterministic_strict_successes"] for row in series["training_pairs"]],
            [3, 1, 0],
        )
        self.assertEqual(
            [row["randomized_strict_successes"] for row in series["training_pairs"]],
            [5, 5, 4],
        )

    def test_rejects_three_implied_baselines_from_wrong_episode_count(self) -> None:
        summary = copy.deepcopy(_summary())
        summary["matched_schedule_rows"]["base"]["episodes"] = 33

        with self.assertRaisesRegex(ValueError, "matched 11-episode schedule"):
            build_progression_series(summary)

    def test_rejects_unanchored_first_training_pair(self) -> None:
        summary = copy.deepcopy(_summary())
        summary["fresh_randomized_multiseed"]["deterministic"][
            "strict_successes_per_training_seed"
        ][0] = 2

        with self.assertRaisesRegex(ValueError, "does not anchor the first seed"):
            build_progression_series(summary)


if __name__ == "__main__":
    unittest.main()
