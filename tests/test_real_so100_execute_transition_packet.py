from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.real_so100_execute_transition_packet import execute_transition_packet


class RealSO100ExecuteTransitionPacketTest(TestCase):
    def test_execute_blocks_before_connection_when_packet_not_ready(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            packet = _write_packet(tmp, ready=False)

            with patch("scripts.real_so100_execute_transition_packet._make_so100_bus") as make_bus:
                report = execute_transition_packet(
                    packet=packet,
                    output=tmp / "report.json",
                    port="/dev/cu.fake",
                    execute=True,
                    human_confirmed=True,
                    workspace_clear_confirmed=True,
                    observer_camera_index=3,
                    visual_output_dir=tmp / "visual",
                    record_video=True,
                    step_settle_seconds=0.0,
                )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        make_bus.assert_not_called()

    def test_ready_packet_dry_run_does_not_connect(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            packet = _write_packet(tmp, ready=True)

            with patch("scripts.real_so100_execute_transition_packet._make_so100_bus") as make_bus:
                report = execute_transition_packet(
                    packet=packet,
                    output=tmp / "report.json",
                    port="/dev/cu.fake",
                    execute=False,
                )

        self.assertEqual(report["status"], "dry_run")
        self.assertFalse(report["send_action_called"])
        make_bus.assert_not_called()

    def test_ready_packet_executes_twenty_steps_with_mock_bus(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            packet = _write_packet(tmp, ready=True)
            bus = _FakeBus()

            with (
                patch("scripts.real_so100_execute_transition_packet._make_so100_bus", return_value=(bus, {})),
                patch("scripts.real_so100_execute_transition_packet._capture_visual", side_effect=_fake_visual),
                patch("scripts.real_so100_execute_transition_packet._start_motion_video", return_value=(None, None, {"path": str(tmp / "motion.mp4")})),
                patch("scripts.real_so100_execute_transition_packet._probe_motion_video", return_value={"exists": False}),
            ):
                report = execute_transition_packet(
                    packet=packet,
                    output=tmp / "report.json",
                    port="/dev/cu.fake",
                    execute=True,
                    human_confirmed=True,
                    workspace_clear_confirmed=True,
                    observer_camera_index=3,
                    visual_output_dir=tmp / "visual",
                    record_video=True,
                    step_settle_seconds=0.0,
                )

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["send_action_called"])
        self.assertTrue(report["policy_actions_executed"])
        self.assertTrue(report["physical_robot_motion"])
        self.assertEqual(report["executed_action_steps"], 20)
        self.assertEqual(len(bus.writes), 20)
        self.assertEqual(set(bus.writes[0]), {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"})


class _FakeBus:
    is_connected = False

    def __init__(self) -> None:
        self.state = {
            "shoulder_pan": 1000.0,
            "shoulder_lift": 1000.0,
            "elbow_flex": 1000.0,
            "wrist_flex": 1000.0,
            "wrist_roll": 1000.0,
            "gripper": 1000.0,
        }
        self.writes: list[dict[str, float]] = []

    def connect(self, *, handshake: bool) -> None:
        self.is_connected = True

    def disconnect(self, *, disable_torque: bool) -> None:
        self.is_connected = False

    def sync_read(self, _register: str, *, normalize: bool):
        return dict(self.state)

    def sync_write(self, _register: str, target: dict[str, float], *, normalize: bool, num_retry: int) -> None:
        self.writes.append(dict(target))
        self.state.update(target)


def _fake_visual(*, camera_index, output_dir, label, before_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{label}.jpg"
    image_path.write_text("fake", encoding="utf-8")
    return {"camera_index": camera_index, "image_path": str(image_path), "visual_motion_detected": label == "after"}


def _write_packet(tmp: Path, *, ready: bool) -> Path:
    packet = tmp / "packet.json"
    steps = []
    step_index = 0
    chunks = []
    for chunk_index in range(2):
        chunk_steps = []
        for local_index in range(10):
            chunk_steps.append(
                {
                    "step_index": step_index,
                    "step_index_in_chunk": local_index,
                    "target_command": {
                        joint: 10.0 + step_index
                        for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    },
                    "target_raw_estimate": {
                        joint: 1000.0 + step_index
                        for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    },
                    "write_normalize": True,
                }
            )
            step_index += 1
        chunks.append({"chunk_index": chunk_index, "steps": chunk_steps})
    packet.write_text(
        json.dumps(
            {
                "status": "ready_for_observer_backed_execution" if ready else "blocked",
                "execution_ready": ready,
                "send_action_called": False,
                "physical_robot_motion": False,
                "chunks": chunks,
            }
        ),
        encoding="utf-8",
    )
    return packet
