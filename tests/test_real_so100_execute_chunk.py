from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.real_so100_execute_chunk import execute_action_chunk


class RealSO100ExecuteChunkTest(TestCase):
    def test_execute_blocks_before_connection_without_confirmations(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            action = _write_action(tmp)

            with patch("scripts.real_so100_execute_chunk._make_so100_bus") as make_bus:
                report = execute_action_chunk(
                    port="/dev/cu.fake",
                    action=action,
                    output=tmp / "report.json",
                    calibration=None,
                    execute=True,
                    human_confirmed=False,
                    experimental_adapter_confirmed=False,
                    action_steps=10,
                    delta_scale_raw_ticks=2.0,
                    max_abs_delta_raw=4.0,
                    step_settle_seconds=0.0,
                    camera_index=None,
                    visual_output_dir=None,
                    record_video=False,
                    video_fps=12.0,
                )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["send_action_called"])
        make_bus.assert_not_called()

    def test_execute_deprecated_raw_scaling_requires_explicit_flag(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            action = _write_action(tmp)
            bus = _FakeBus()

            with (
                patch("scripts.real_so100_execute_chunk._make_so100_bus", return_value=(bus, {})),
                patch("scripts.real_so100_execute_chunk._capture_visual", side_effect=_fake_visual),
                patch("scripts.real_so100_execute_chunk._start_motion_video", return_value=(None, None, {"path": str(tmp / "motion.mp4")})),
                patch("scripts.real_so100_execute_chunk._probe_motion_video", return_value={"exists": False}),
            ):
                report = execute_action_chunk(
                    port="/dev/cu.fake",
                    action=action,
                    output=tmp / "report.json",
                    calibration=None,
                    execute=True,
                    human_confirmed=True,
                    experimental_adapter_confirmed=True,
                    action_steps=10,
                    delta_scale_raw_ticks=2.0,
                    max_abs_delta_raw=4.0,
                    step_settle_seconds=0.0,
                    camera_index=3,
                    visual_output_dir=tmp / "visual",
                    record_video=True,
                    video_fps=12.0,
                )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["send_action_called"])
        self.assertEqual(len(bus.writes), 0)
        self.assertIn("Deprecated raw tick scaling", " ".join(report["blockers"]))

    def test_execute_writes_ten_metadata_chunk_steps_with_mock_bus(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            action = _write_action(tmp)
            config = _write_config(tmp)
            stats = _write_stats(tmp)
            bus = _FakeBus()

            with (
                patch("scripts.real_so100_execute_chunk._make_so100_bus", return_value=(bus, {})),
                patch("scripts.real_so100_execute_chunk._capture_visual", side_effect=_fake_visual),
                patch("scripts.real_so100_execute_chunk._start_motion_video", return_value=(None, None, {"path": str(tmp / "motion.mp4")})),
                patch("scripts.real_so100_execute_chunk._probe_motion_video", return_value={"exists": False}),
            ):
                report = execute_action_chunk(
                    port="/dev/cu.fake",
                    action=action,
                    output=tmp / "report.json",
                    calibration=None,
                    execute=True,
                    human_confirmed=True,
                    experimental_adapter_confirmed=False,
                    action_steps=10,
                    delta_scale_raw_ticks=2.0,
                    max_abs_delta_raw=4.0,
                    step_settle_seconds=0.0,
                    camera_index=3,
                    visual_output_dir=tmp / "visual",
                    record_video=True,
                    video_fps=12.0,
                    metadata_config=config,
                    action_stats=stats,
                    action_semantics="joint_delta",
                    gripper_semantics="higher_raw_opens",
                    command_units="feetech_raw_ticks",
                    confirm_so100_joint_order=True,
                )

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["send_action_called"])
        self.assertTrue(report["policy_actions_executed"])
        self.assertEqual(report["executed_action_steps"], 10)
        self.assertEqual(len(bus.writes), 10)
        self.assertEqual(set(bus.writes[0]), {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"})

    def test_dry_metadata_plan_uses_calibration_for_range_checks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            action = _write_action(tmp)
            config = _write_config(tmp)
            stats = _write_stats(tmp)
            calibration = tmp / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "shoulder_pan": {"range_min": -1, "range_max": 1},
                        "shoulder_lift": {"range_min": -1, "range_max": 1},
                        "elbow_flex": {"range_min": -1, "range_max": 1},
                        "wrist_flex": {"range_min": -1, "range_max": 1},
                        "wrist_roll": {"range_min": -1, "range_max": 1},
                        "gripper": {"range_min": 0, "range_max": 100},
                    }
                ),
                encoding="utf-8",
            )

            report = execute_action_chunk(
                port="/dev/cu.fake",
                action=action,
                output=tmp / "report.json",
                calibration=calibration,
                execute=False,
                human_confirmed=False,
                experimental_adapter_confirmed=False,
                action_steps=10,
                delta_scale_raw_ticks=2.0,
                max_abs_delta_raw=4.0,
                step_settle_seconds=0.0,
                camera_index=None,
                visual_output_dir=None,
                record_video=False,
                video_fps=12.0,
                metadata_config=config,
                action_stats=stats,
                action_semantics="absolute_joint_position",
                gripper_semantics="higher_raw_opens",
                command_units="lerobot_so100_position",
                confirm_so100_joint_order=True,
            )

        self.assertEqual(report["status"], "dry_run")
        self.assertIn("outside calibrated range", " ".join(report["dry_plan"]["blockers"]))


class _FakeBus:
    is_connected = False

    def __init__(self) -> None:
        self.state = {
            "shoulder_pan": 2400.0,
            "shoulder_lift": 2047.0,
            "elbow_flex": 2030.0,
            "wrist_flex": 1988.0,
            "wrist_roll": 2050.0,
            "gripper": 1865.0,
        }
        self.writes: list[dict[str, int]] = []

    def connect(self, *, handshake: bool) -> None:
        self.is_connected = True

    def disconnect(self, *, disable_torque: bool) -> None:
        self.is_connected = False

    def sync_read(self, _register: str, *, normalize: bool):
        return dict(self.state)

    def sync_write(self, _register: str, target: dict[str, int], *, normalize: bool, num_retry: int) -> None:
        self.writes.append(dict(target))
        self.state.update(target)


def _fake_visual(*, camera_index, output_dir, label, before_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{label}.jpg"
    image_path.write_text("fake", encoding="utf-8")
    return {"camera_index": camera_index, "image_path": str(image_path), "visual_motion_detected": label == "after"}


def _write_action(tmp: Path) -> Path:
    action = tmp / "action.json"
    action.write_text(
        json.dumps(
            {
                "raw_action": [0, 0, 0, 0, 0, 0],
                "raw_action_chunk": [[0.5, 0.25, -0.5, 0.25, 0.0, -0.25] for _index in range(12)],
            }
        ),
        encoding="utf-8",
    )
    return action


def _write_config(tmp: Path) -> Path:
    config = tmp / "config.json"
    config.write_text(
        json.dumps(
            {
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
                "normalization_mapping": {"ACTION": "MEAN_STD"},
                "chunk_size": 50,
                "n_action_steps": 50,
            }
        ),
        encoding="utf-8",
    )
    return config


def _write_stats(tmp: Path) -> Path:
    stats = tmp / "stats.json"
    stats.write_text(
        json.dumps(
            {
                "action": {
                    "mean": [0, 0, 0, 0, 0, 0],
                    "std": [1, 1, 1, 1, 1, 1],
                }
            }
        ),
        encoding="utf-8",
    )
    return stats
