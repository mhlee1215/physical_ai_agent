from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import export_mycobot_280_ground_pickup_teacher_dataset as teacher_export


class MyCobot280GroundPickupTeacherDatasetLifecycleTest(unittest.TestCase):
    def test_export_reuses_one_environment_and_closes_it_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"
            fake_env = Mock()
            environments = []

            def fake_attempt(**kwargs):
                environments.append(kwargs["env"])
                index = len(environments) - 1
                return {
                    "split": kwargs["split"],
                    "split_episode_index": kwargs["split_episode_index"],
                    "episode_index": kwargs["global_episode_index"],
                    "attempt_index": kwargs["attempt_index"],
                    "seed": kwargs["seed"],
                    "yaw_delta_rad": kwargs["yaw_delta"],
                    "initial_cube_xy": [0.15 + index * 0.001, 0.01 + index * 0.001],
                    "trajectory_hash": f"trajectory-{index}",
                    "path": (
                        f"splits/{kwargs['split']}/episodes/"
                        f"episode_{kwargs['split_episode_index']:04d}.jsonl"
                    ),
                    "frames": 530,
                    "rendered_frames": 530,
                    "success": True,
                    "final_cube_lift_m": 0.055,
                    "lift_best_sustained_two_pad_steps": 80,
                    "post_lift_hold_best_sustained_two_pad_steps": 300,
                    "post_lift_hold_min_cube_lift_m": 0.046,
                    "max_pad_cube_penetration_m": 0.0028,
                    "max_lift_pad_cube_penetration_m": 0.002,
                }

            with (
                patch.object(teacher_export, "_make_env", return_value=fake_env) as make_env,
                patch.object(teacher_export, "_export_attempt", side_effect=fake_attempt) as export_attempt,
            ):
                manifest = teacher_export.export_dataset(
                    output_dir=output_dir,
                    train_episodes=2,
                    val_episodes=1,
                    seed=200,
                    asset_root=Path("_vendor/mycobot_mujoco"),
                    official_gripper_root=Path("_vendor/mycobot_ros"),
                    width=256,
                    height=256,
                    fps=30,
                    render_every=1,
                    max_attempts=3,
                    yaw_min=-0.2,
                    yaw_max=0.2,
                )

            self.assertEqual(manifest["accepted_episodes"], 3)
            self.assertEqual(manifest["failed_episodes"], [])
            make_env.assert_called_once()
            self.assertEqual(export_attempt.call_count, 3)
            self.assertEqual(environments, [fake_env, fake_env, fake_env])
            fake_env.close.assert_called_once_with()
            self.assertFalse((output_dir / "scene_cache" / "shared").exists())


if __name__ == "__main__":
    unittest.main()
