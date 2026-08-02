from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from physical_ai_agent.real_so100.dataset_episode_replay import (
    EpisodeTrajectory,
    ReplayConfig,
    build_home_return_plan,
    build_absolute_replay_plan,
    interpolate_raw_pose,
    load_home_pose_raw,
    load_episode_trajectory,
    runtime_from_config,
    sim_qpos_to_raw_targets,
)


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def calibration() -> dict[str, dict[str, int]]:
    return {joint: {"range_min": 0, "range_max": 4095} for joint in JOINTS}


class DatasetEpisodeReplayTest(unittest.TestCase):
    def test_json_config_requires_every_field_and_rejects_unknown_fields(self) -> None:
        payload = {
            "schema_version": 1,
            "dataset": {"root": "dataset", "episode": 48},
            "hardware": {
                "port": "/dev/robot",
                "calibration": "calibration.json",
                "home_pose": "home.json",
                "serial_num_retry": 3,
            },
            "trajectory": {
                "fps": None,
                "alignment": "calibrated-start-relative",
                "max_trajectory_step_raw": 50.0,
                "max_bridge_step_raw": 10.0,
                "max_tracking_error_raw": 45.0,
                "max_start_error_raw": 30.0,
                "start_range_tolerance_raw": 5.0,
                "bridge_step_seconds": 0.12,
                "hold_final_seconds": 1.0,
            },
            "execution": {
                "enabled": True,
                "operator_confirmed": True,
                "workspace_clear_confirmed": True,
                "direct_observer_confirmed": True,
                "require_typed_confirmation": True,
                "return_mode": "home",
                "disable_torque_after_run": True,
            },
            "recording": {"enabled": True, "camera_index": 1, "video_fps": 12.0},
            "output_dir": "output",
        }
        config = ReplayConfig.model_validate(payload)
        runtime = runtime_from_config(config, config_path=Path("replay.json"))
        self.assertEqual(runtime.episode, 48)
        self.assertEqual(runtime.return_mode, "home")

        with self.assertRaisesRegex(ValueError, "Extra inputs are not permitted"):
            ReplayConfig.model_validate({**payload, "unknown_setting": True})
        missing = json.loads(json.dumps(payload))
        del missing["trajectory"]["max_tracking_error_raw"]
        with self.assertRaisesRegex(ValueError, "Field required"):
            ReplayConfig.model_validate(missing)

    def test_plan_uses_first_state_only_for_start_and_actions_for_commands(self) -> None:
        trajectory = EpisodeTrajectory(
            dataset_root=Path("dataset"),
            episode=3,
            frame_indices=(0, 1),
            start_state=(0.0, 0.0, 0.0, 0.0, 0.0, math.radians(-10.0)),
            actions=(
                (0.1, 0.0, 0.0, 0.0, 0.0, math.radians(-10.0)),
                (0.2, 0.0, 0.0, 0.0, 0.0, math.radians(100.0)),
            ),
            dataset_fps=12.0,
        )
        plan = build_absolute_replay_plan(trajectory, calibration(), max_trajectory_step_raw=5000.0)

        self.assertAlmostEqual(plan["start_raw"]["shoulder_pan"], 2047.5)
        self.assertNotEqual(plan["actions"][0]["target_raw"]["shoulder_pan"], plan["start_raw"]["shoulder_pan"])
        self.assertAlmostEqual(plan["actions"][1]["target_raw"]["gripper"], 4095.0)
        self.assertEqual(plan["command_source"], "action at each frame")

    def test_rejects_absolute_target_outside_calibration_without_clipping(self) -> None:
        narrow = {joint: {"range_min": 1000, "range_max": 3000} for joint in JOINTS}
        with self.assertRaisesRegex(ValueError, "outside calibration"):
            sim_qpos_to_raw_targets((math.pi, 0, 0, 0, 0, 0), narrow)

    def test_calibrated_start_alignment_preserves_action_delta(self) -> None:
        narrow = {joint: {"range_min": 1000, "range_max": 3000} for joint in JOINTS}
        trajectory = EpisodeTrajectory(
            dataset_root=Path("dataset"),
            episode=7,
            frame_indices=(0, 1),
            start_state=(0.0, -0.2, 0.2, 0.0, 0.0, math.radians(-10.0)),
            actions=(
                (0.0, -0.19, 0.19, 0.0, 0.0, math.radians(-10.0)),
                (0.0, -0.18, 0.18, 0.0, 0.0, math.radians(-10.0)),
            ),
            dataset_fps=12.0,
        )
        plan = build_absolute_replay_plan(
            trajectory,
            narrow,
            max_trajectory_step_raw=100.0,
            alignment="calibrated-start-relative",
        )

        scale = 4095.0 / 360.0
        observed = (
            plan["actions"][1]["target_raw"]["shoulder_lift"]
            - plan["actions"][0]["target_raw"]["shoulder_lift"]
        )
        self.assertAlmostEqual(observed, math.degrees(0.01) * scale)
        self.assertEqual(plan["alignment"], "calibrated-start-relative")

    def test_interpolation_limits_every_joint_step(self) -> None:
        start = {joint: 0.0 for joint in JOINTS}
        target = {joint: 0.0 for joint in JOINTS}
        target["elbow_flex"] = 25.0
        steps = interpolate_raw_pose(start, target, max_step_raw=10.0)
        self.assertEqual(len(steps), 3)
        previous = start
        for step in steps:
            self.assertLessEqual(abs(step["elbow_flex"] - previous["elbow_flex"]), 10.0)
            previous = step
        self.assertEqual(steps[-1], target)

    def test_loader_requires_contiguous_full_episode_and_preserves_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "meta").mkdir()
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(json.dumps({"fps": 12}), encoding="utf-8")
            state = [0.0, 0.0, 0.0, 0.0, 0.0, math.radians(-10.0)]
            action0 = [0.1, 0.0, 0.0, 0.0, 0.0, math.radians(-10.0)]
            action1 = [0.2, 0.0, 0.0, 0.0, 0.0, math.radians(100.0)]
            table = pa.table(
                {
                    "episode_index": pa.array([4, 4], type=pa.int64()),
                    "frame_index": pa.array([0, 1], type=pa.int64()),
                    "observation.state": pa.array([state, state], type=pa.list_(pa.float32(), 6)),
                    "action": pa.array([action0, action1], type=pa.list_(pa.float32(), 6)),
                }
            )
            pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

            loaded = load_episode_trajectory(root, 4)

            self.assertEqual(loaded.frame_indices, (0, 1))
            self.assertAlmostEqual(loaded.actions[1][0], 0.2, places=6)
            self.assertEqual(loaded.dataset_fps, 12.0)

    def test_home_return_loads_canonical_target_raw_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "home.json"
            expected = {joint: 2000 + index for index, joint in enumerate(JOINTS)}
            path.write_text(json.dumps({"name": "home", "target_raw": expected}), encoding="utf-8")

            loaded = load_home_pose_raw(path, calibration())

            self.assertEqual(loaded, {joint: float(value) for joint, value in expected.items()})

    def test_home_return_reports_calibration_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "home.json"
            requested = {joint: 2000 for joint in JOINTS}
            requested["gripper"] = -10
            path.write_text(json.dumps({"name": "home", "target_raw": requested}), encoding="utf-8")

            plan = build_home_return_plan(path, calibration())

            self.assertEqual(plan["target_raw"]["gripper"], 0.0)
            self.assertEqual(plan["calibration_adjustments"]["gripper"]["requested"], -10.0)


if __name__ == "__main__":
    unittest.main()
