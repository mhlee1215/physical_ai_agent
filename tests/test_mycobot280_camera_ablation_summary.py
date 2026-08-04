from __future__ import annotations

import copy
import unittest

from scripts.summarize_mycobot280_camera_ablation import summarize_reports


def _report(
    *,
    camera: str,
    success_seeds: set[int],
    lifts_m: list[float],
) -> dict:
    schedule = [
        {"episode": 0, "seed": 10, "torch_seed": 20, "yaw_delta_rad": -0.1},
        {"episode": 1, "seed": 11, "torch_seed": 21, "yaw_delta_rad": 0.1},
    ]
    episodes = []
    for index, (item, lift) in enumerate(zip(schedule, lifts_m, strict=True)):
        success = item["seed"] in success_seeds
        episodes.append(
            {
                "episode": index,
                "seed": item["seed"],
                "success": success,
                "final_cube_lift_m": lift,
                "post_lift_hold_min_cube_lift_m": lift - 0.005,
                "max_pad_cube_penetration_m": 0.0029 + 0.0001 * index,
                "clipped_action_values": 10 + index,
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
                "failed": len(episodes) - len(success_seeds),
            }
        },
        "environment": {
            "render_camera_profile": camera,
            "object_physics": "fixed",
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
    }


class CameraAblationSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reports = {
            "base_wide": _report(
                camera="full_robot",
                success_seeds=set(),
                lifts_m=[0.04, 0.05],
            ),
            "finetuned_wide": _report(
                camera="full_robot",
                success_seeds={11},
                lifts_m=[0.06, 0.07],
            ),
            "base_close": _report(
                camera="ground_pickup_closeup",
                success_seeds=set(),
                lifts_m=[0.03, 0.04],
            ),
            "finetuned_close": _report(
                camera="ground_pickup_closeup",
                success_seeds={10, 11},
                lifts_m=[0.065, 0.075],
            ),
        }

    def test_summarizes_rows_and_paired_seed_deltas(self) -> None:
        summary = summarize_reports(self.reports)

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["rows"]["finetuned_close"]["successful_episodes"], 2)
        self.assertAlmostEqual(
            summary["rows"]["finetuned_close"]["mean_final_cube_lift_mm"],
            70.0,
        )
        close_effect = summary["paired_comparisons"]["camera_effect_finetuned"]
        self.assertAlmostEqual(close_effect["mean_final_lift_delta_mm"], 5.0)
        self.assertEqual(close_effect["added_success_seeds"], [10])
        self.assertEqual(close_effect["lost_success_seeds"], [])

    def test_rejects_a_camera_contract_mismatch(self) -> None:
        reports = copy.deepcopy(self.reports)
        reports["finetuned_close"]["environment"]["render_camera_profile"] = "full_robot"

        with self.assertRaisesRegex(ValueError, "camera profile"):
            summarize_reports(reports)


if __name__ == "__main__":
    unittest.main()
