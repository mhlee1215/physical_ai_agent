from __future__ import annotations

import copy
import unittest

from scripts.summarize_mycobot280_policy_claims import build_claim_summary


def _report(strict: int, pickup_hold: int) -> dict:
    schedule = [
        {"episode": index, "seed": 100 + index, "torch_seed": 200 + index}
        for index in range(3)
    ]
    episodes = []
    for index, item in enumerate(schedule):
        functional = index < pickup_hold
        success = index < strict
        episodes.append(
            {
                "seed": item["seed"],
                "success": success,
                "failed_gates": []
                if success
                else ["max_pad_cube_penetration_exceeded"]
                if functional
                else ["final_cube_lift_below_threshold"],
                "final_cube_lift_m": 0.06 if functional else 0.03,
                "final_pad_cube_contacted_pads": 2 if functional else 0,
                "lift_best_sustained_two_pad_steps": 80 if functional else 0,
                "post_lift_hold_best_sustained_two_pad_steps": 300
                if functional
                else 0,
                "post_lift_hold_min_cube_lift_m": 0.05 if functional else 0.02,
                "candidate": {"spawn_seed": item["seed"], "cube_mass_kg": 0.03},
            }
        )
    return {
        "status": "completed",
        "schedule": schedule,
        "steps_per_episode": 530,
        "episode_summaries": episodes,
        "environment": {
            "render_camera_profile": "ground_pickup_closeup",
            "object_physics": "randomized_from_audited_source_manifest",
        },
        "policy_runtime": {
            "contract": {"feature_contract": {"exact_7d_state_action": True}}
        },
        "schedule_contract": {
            "candidate_selection": "direct_unfiltered_draws_without_teacher_rejection",
            "source_seed_overlap_count": 0,
        },
    }


def _aggregate(strict: list[int], pickup: list[int], penetration: list[int]) -> dict:
    episodes = 3 * len(strict)
    return {
        "episodes_total": episodes,
        "strict_successes_total": sum(strict),
        "pooled_strict_success_rate": sum(strict) / episodes,
        "strict_successes_per_training_seed": strict,
        "pickup_hold_successes_total": sum(pickup),
        "pooled_pickup_hold_success_rate": sum(pickup) / episodes,
        "penetration_only_failures_total": sum(penetration),
    }


def _multiseed() -> dict:
    return {
        "status": "passed",
        "training_seeds": [11, 12, 13],
        "training_seed_count": 3,
        "aggregate_rows": {
            "deterministic_fresh_randomized": _aggregate(
                [1, 0, 0], [3, 3, 2], [2, 3, 2]
            ),
            "randomized_fresh_randomized": _aggregate(
                [2, 2, 1], [3, 3, 3], [1, 1, 2]
            ),
        },
        "paired_comparisons": {
            "fresh_randomized_training_data_effect": {
                "left": "deterministic_fresh_randomized",
                "right": "randomized_fresh_randomized",
                "strict_success_count_deltas": [1, 2, 1],
                "positive_training_seed_count": 3,
                "exact_two_sided_sign_test_p": 0.25,
                "mean_paired_penetration_delta_mm": -0.08,
            }
        },
        "source_distribution_caveat": {
            "accepted_attempts": 60,
            "total_attempts": 73,
            "highest_mass_quartile_accepted": 2,
            "highest_mass_quartile_attempted": 15,
            "highest_mass_quartile_validation_examples": 0,
            "validation_empty_friction_quartiles": [1, 2],
        },
    }


class PolicyClaimsSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reports = {
            "base": _report(strict=0, pickup_hold=0),
            "deterministic_finetuned": _report(strict=1, pickup_hold=3),
            "randomized_finetuned": _report(strict=2, pickup_hold=3),
        }

    def test_separates_functional_and_strict_claims(self) -> None:
        summary = build_claim_summary(self.reports, _multiseed())

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["matched_schedule_rows"]["base"]["pickup_hold_successes"], 0)
        self.assertEqual(summary["matched_schedule_rows"]["randomized_finetuned"]["strict_successes"], 2)
        self.assertEqual(summary["fresh_randomized_multiseed"]["positive_training_seed_count"], 3)
        self.assertEqual(
            summary["supported_claims"]["randomized_improves_functional_success"]["status"],
            "not_supported_as_a_material_advantage",
        )
        self.assertEqual(
            summary["teacher_filtering"]["does_not_apply_to"],
            "closed-loop policy evaluation rollouts",
        )

    def test_rejects_filtered_policy_evaluation(self) -> None:
        reports = copy.deepcopy(self.reports)
        reports["randomized_finetuned"]["schedule_contract"][
            "candidate_selection"
        ] = "teacher_accepted_only"

        with self.assertRaisesRegex(ValueError, "not an unfiltered direct draw"):
            build_claim_summary(reports, _multiseed())

    def test_rejects_mismatched_candidates(self) -> None:
        reports = copy.deepcopy(self.reports)
        reports["deterministic_finetuned"]["episode_summaries"][0]["candidate"][
            "cube_mass_kg"
        ] = 0.04

        with self.assertRaisesRegex(ValueError, "candidates do not match"):
            build_claim_summary(reports, _multiseed())


if __name__ == "__main__":
    unittest.main()
