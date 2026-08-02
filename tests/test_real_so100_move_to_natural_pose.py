from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_move_to_natural_pose import move_to_natural_pose


class _FakeBus:
    def __init__(self) -> None:
        self.is_connected = False
        self.calls: list[tuple] = []
        self.positions = {
            "shoulder_pan": 2000,
            "shoulder_lift": 2100,
            "elbow_flex": 800,
            "wrist_flex": 1900,
            "wrist_roll": 2000,
            "gripper": 2100,
        }

    def connect(self, *, handshake: bool) -> None:
        self.calls.append(("connect", handshake))
        self.is_connected = True

    def sync_read(self, register: str, *, normalize: bool):
        self.calls.append(("sync_read", register, normalize))
        return dict(self.positions)

    def sync_write(self, register: str, values: dict[str, int], **kwargs) -> None:
        self.calls.append(("sync_write", register, dict(values), kwargs))
        if register == "Goal_Position":
            self.positions.update(values)

    def enable_torque(self, **kwargs) -> None:
        self.calls.append(("enable_torque", kwargs))

    def disconnect(self, *, disable_torque: bool) -> None:
        self.calls.append(("disconnect", disable_torque))
        self.is_connected = False


class _Releasable:
    def release(self) -> None:
        pass


class MoveToNaturalPoseTest(unittest.TestCase):
    def test_preloads_current_pose_before_enabling_torque_and_moving(self) -> None:
        bus = _FakeBus()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        joint: {"range_min": 0, "range_max": 4095}
                        for joint in SO100_JOINT_ORDER
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "scripts.real_so100_move_to_natural_pose._make_so100_bus",
                    return_value=(bus, {}),
                ),
                patch(
                    "scripts.real_so100_move_to_natural_pose._capture_visual",
                    return_value={"image_path": str(root / "frame.jpg")},
                ),
                patch(
                    "scripts.real_so100_move_to_natural_pose._start_motion_video",
                    return_value=(_Releasable(), _Releasable(), {"path": str(root / "motion.mp4")}),
                ),
                patch("scripts.real_so100_move_to_natural_pose._record_motion_video"),
                patch("scripts.real_so100_move_to_natural_pose._probe_motion_video", return_value={}),
            ):
                report = move_to_natural_pose(
                    port="fake",
                    calibration=calibration,
                    output=root / "report.json",
                    execute=True,
                    human_confirmed=True,
                    workspace_clear_confirmed=True,
                    max_abs_delta_raw=100,
                    step_settle_seconds=0,
                    camera_index=0,
                    visual_output_dir=root / "visual",
                    record_video=True,
                    video_fps=12,
                    target_overrides={joint: 2200 for joint in SO100_JOINT_ORDER},
                )

        preload_index = next(
            index
            for index, call in enumerate(bus.calls)
            if call[0:2] == ("sync_write", "Goal_Position")
        )
        enable_index = next(index for index, call in enumerate(bus.calls) if call[0] == "enable_torque")
        movement_index = next(
            index
            for index, call in enumerate(bus.calls[enable_index + 1 :], start=enable_index + 1)
            if call[0:2] == ("sync_write", "Goal_Position")
        )
        self.assertLess(preload_index, enable_index)
        self.assertLess(enable_index, movement_index)
        self.assertTrue(report["torque_enabled_for_move"])
        self.assertTrue(report["post_task_torque_disabled"])


if __name__ == "__main__":
    unittest.main()
