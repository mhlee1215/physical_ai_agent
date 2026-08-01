from __future__ import annotations

import copy
import unittest

from scripts.summarize_mycobot280_randomized_training_multiseed import (
    summarize_multiseed,
)


def _report(
    *,
    training_seed: int,
    physics: str,
    success_seeds: set[int],
    fresh: bool,
) -> dict:
    environment_seeds = [10, 11]
    schedule = [
        {
            "episode": index,
            "seed": seed,
            "torch_seed": training_seed + index,
            "yaw_delta_rad": -0.1 + 0.2 * index,
        }
        for index, seed in enumerate(environment_seeds)
    ]
    episodes = []
    for index, item in enumerate(schedule):
        success = item["seed"] in success_seeds
        episodes.append(
            {
                "episode": index,
                "seed": item["seed"],
                "success": success,
                "failed_gates": []
                if success
                else ["max_pad_cube_penetration_exceeded"],
                "final_cube_lift_m": 0.06,
                "final_pad_cube_contacted_pads": 2,
                "lift_best_sustained_two_pad_steps": 80,
                "post_lift_hold_best_sustained_two_pad_steps": 300,
                "post_lift_hold_min_cube_lift_m": 0.051,
                "max_pad_cube_penetration_m": 0.0029 + index * 0.0002,
                "candidate": {
                    "spawn_seed": item["seed"],
                    "cube_mass_kg": 0.035 if index == 0 else 0.03,
                }
                if fresh
                else None,
            }
        )
    return {
        "status": "completed",
        "schedule": schedule,
        "steps_per_episode": 530,
        "episode_summaries": episodes,
        "aggregate": {
            "failure_reason_counts": {
                "passed": len(success_seeds),
                "max_pad_cube_penetration_exceeded": 2 - len(success_seeds),
            }
        },
        "environment": {
            "render_camera_profile": "ground_pickup_closeup",
            "object_physics": physics,
        },
        "policy_runtime": {
            "contract": {
                "feature_contract": {
                    "exact_7d_state_action": True,
                    "state_shape": [7],
                    "action_shape": [7],
                }
            }
        },
        "schedule_contract": {
            "candidate_selection": "direct_unfiltered_draws_without_teacher_rejection",
            "source_seed_overlap_count": 0,
        }
        if fresh
        else {},
    }


def _report_set(training_seed: int, deterministic_successes: int) -> dict:
    deterministic_seeds = set([10, 11][:deterministic_successes])
    randomized_seeds = {10, 11}
    return {
        "deterministic_nominal": _report(
            training_seed=training_seed,
            physics="fixed",
            success_seeds=deterministic_seeds,
            fresh=False,
        ),
        "randomized_nominal": _report(
            training_seed=training_seed,
            physics="fixed",
            success_seeds=randomized_seeds,
            fresh=False,
        ),
        "deterministic_fresh_randomized": _report(
            training_seed=training_seed,
            physics="randomized_from_audited_source_manifest",
            success_seeds=deterministic_seeds,
            fresh=True,
        ),
        "randomized_fresh_randomized": _report(
            training_seed=training_seed,
            physics="randomized_from_audited_source_manifest",
            success_seeds=randomized_seeds,
            fresh=True,
        ),
    }


class RandomizedTrainingMultiseedSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report_sets = {
            101: _report_set(101, 1),
            102: _report_set(102, 0),
            103: _report_set(103, 1),
        }

    def test_summarizes_paired_training_seed_replication(self) -> None:
        summary = summarize_multiseed(self.report_sets)

        self.assertEqual(summary["status"], "passed")
        row = summary["aggregate_rows"]["deterministic_nominal"]
        self.assertEqual(row["strict_successes_per_training_seed"], [1, 0, 1])
        self.assertEqual(row["strict_successes_total"], 2)
        self.assertEqual(row["penetration_only_failures_per_training_seed"], [1, 2, 1])
        self.assertEqual(row["penetration_only_failures_total"], 4)
        self.assertEqual(row["other_strict_failures_total"], 0)
        paired = summary["paired_comparisons"]["nominal_training_data_effect"]
        self.assertEqual(paired["strict_success_count_deltas"], [1, 2, 1])
        self.assertEqual(paired["positive_training_seed_count"], 3)
        self.assertAlmostEqual(paired["exact_two_sided_sign_test_p"], 0.25)
        high_mass = summary["aggregate_rows"][
            "randomized_fresh_randomized"
        ]["high_mass_slice"]
        self.assertEqual(high_mass["strict_successes"], 3)
        self.assertEqual(high_mass["episodes"], 3)

    def test_rejects_cross_seed_schedule_drift(self) -> None:
        report_sets = copy.deepcopy(self.report_sets)
        for name in ("deterministic_nominal", "randomized_nominal"):
            report_sets[103][name]["schedule"][0]["yaw_delta_rad"] = -0.2

        with self.assertRaisesRegex(ValueError, "schedules differ across training seeds"):
            summarize_multiseed(report_sets)

    def test_rejects_mismatched_randomized_candidates(self) -> None:
        report_sets = copy.deepcopy(self.report_sets)
        report_sets[102]["randomized_fresh_randomized"]["episode_summaries"][0][
            "candidate"
        ]["cube_mass_kg"] = 0.04

        with self.assertRaisesRegex(ValueError, "candidates do not match"):
            summarize_multiseed(report_sets)

    def test_rejects_source_seed_overlap(self) -> None:
        report_sets = copy.deepcopy(self.report_sets)
        report_sets[102]["deterministic_fresh_randomized"]["schedule_contract"][
            "source_seed_overlap_count"
        ] = 1

        with self.assertRaisesRegex(ValueError, "overlaps source attempt seeds"):
            summarize_multiseed(report_sets)


if __name__ == "__main__":
    unittest.main()
