from __future__ import annotations

import copy
import unittest

from scripts.summarize_mycobot280_randomized_training_pilot import summarize_pilot


def _report(*, physics: str, success_seeds: set[int], fresh: bool) -> dict:
    schedule = [
        {"episode": 0, "seed": 10, "torch_seed": 20, "yaw_delta_rad": -0.1},
        {"episode": 1, "seed": 11, "torch_seed": 21, "yaw_delta_rad": 0.1},
    ]
    episodes = []
    for index, item in enumerate(schedule):
        success = item["seed"] in success_seeds
        episodes.append(
            {
                "episode": index,
                "seed": item["seed"],
                "success": success,
                "failed_gates": [] if success else ["max_pad_cube_penetration_exceeded"],
                "final_cube_lift_m": 0.06 + index * 0.001,
                "final_pad_cube_contacted_pads": 2,
                "lift_best_sustained_two_pad_steps": 80,
                "post_lift_hold_best_sustained_two_pad_steps": 300,
                "post_lift_hold_min_cube_lift_m": 0.051,
                "max_pad_cube_penetration_m": 0.0029 + index * 0.0002,
                "candidate": {"spawn_seed": item["seed"], "cube_mass_kg": 0.03}
                if fresh
                else None,
            }
        )
    return {
        "status": "completed",
        "schedule": schedule,
        "steps_per_episode": 530,
        "episode_summaries": episodes,
        "aggregate": {"failure_reason_counts": {"passed": len(success_seeds)}},
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


def _supervised(*, loss: float, rmse: float) -> dict:
    return {
        "operation": "evaluate_smolvla_supervised_loss",
        "batches_evaluated": 20,
        "loss_mean": loss,
        "postprocessed_action_rmse_mean": rmse,
        "contract": {"feature_contract": {"exact_7d_state_action": True}},
    }


class RandomizedTrainingPilotSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reports = {
            "deterministic_nominal": _report(
                physics="fixed", success_seeds={11}, fresh=False
            ),
            "randomized_nominal": _report(
                physics="fixed", success_seeds={10, 11}, fresh=False
            ),
            "deterministic_fresh_randomized": _report(
                physics="randomized_from_audited_source_manifest",
                success_seeds=set(),
                fresh=True,
            ),
            "randomized_fresh_randomized": _report(
                physics="randomized_from_audited_source_manifest",
                success_seeds={10},
                fresh=True,
            ),
        }
        self.supervised = {
            "base": _supervised(loss=6.0, rmse=0.10),
            "randomized_finetuned": _supervised(loss=5.0, rmse=0.11),
        }

    def test_summarizes_strict_functional_and_paired_results(self) -> None:
        summary = summarize_pilot(self.reports, self.supervised)

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["rows"]["randomized_nominal"]["strict_successes"], 2)
        self.assertEqual(summary["rows"]["randomized_nominal"]["pickup_hold_successes"], 2)
        paired = summary["paired_comparisons"]["fresh_randomized_training_data_effect"]
        self.assertEqual(paired["strict_success_count_delta"], 1)
        self.assertEqual(paired["added_strict_success_seeds"], [10])
        self.assertAlmostEqual(summary["heldout_supervised"]["loss_relative_change"], -1 / 6)
        self.assertAlmostEqual(summary["heldout_supervised"]["action_rmse_relative_change"], 0.1)

    def test_rejects_mismatched_randomized_candidates(self) -> None:
        reports = copy.deepcopy(self.reports)
        reports["randomized_fresh_randomized"]["episode_summaries"][0]["candidate"]["cube_mass_kg"] = 0.04

        with self.assertRaisesRegex(ValueError, "candidates do not match"):
            summarize_pilot(reports, self.supervised)

    def test_rejects_source_seed_overlap(self) -> None:
        reports = copy.deepcopy(self.reports)
        reports["deterministic_fresh_randomized"]["schedule_contract"]["source_seed_overlap_count"] = 1

        with self.assertRaisesRegex(ValueError, "overlaps source-dataset"):
            summarize_pilot(reports, self.supervised)


if __name__ == "__main__":
    unittest.main()
