from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_transition_execution_packet import build_transition_execution_packet


class RealSO100TransitionExecutionPacketTest(TestCase):
    def test_blocks_when_refresh_preflight_is_not_ready(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            refresh = _write_bundle(tmp, ready=False)

            report = build_transition_execution_packet(
                refresh_report=refresh,
                output=tmp / "packet.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["execution_ready"])
        self.assertTrue(any("refresh is not ready" in blocker for blocker in report["blockers"]))
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])

    def test_builds_ready_packet_from_ready_refresh_bundle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            refresh = _write_bundle(tmp, ready=True)

            report = build_transition_execution_packet(
                refresh_report=refresh,
                output=tmp / "packet.json",
            )

        self.assertEqual(report["status"], "ready_for_observer_backed_execution")
        self.assertTrue(report["execution_ready"])
        self.assertTrue(report["live_readback_regenerated"])
        self.assertEqual(report["transition_chunk_count"], 2)
        self.assertEqual(report["transition_step_count"], 20)
        self.assertEqual(report["chunks"][0]["steps"][0]["target_command"]["shoulder_pan"], 1000.0)
        self.assertEqual(report["next_agentic_layer_step"]["type"], "execute_packet_with_camera_3_recording")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])

    def test_blocks_when_chunk_is_not_ten_steps(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            refresh = _write_bundle(tmp, ready=True, steps_per_chunk=9)

            report = build_transition_execution_packet(
                refresh_report=refresh,
                output=tmp / "packet.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("expected 10" in blocker for blocker in report["blockers"]))


def _write_bundle(root: Path, *, ready: bool, steps_per_chunk: int = 10) -> Path:
    regen = root / "regen.json"
    gate = root / "gate.json"
    preflight = root / "preflight.json"
    refresh = root / "refresh.json"
    regen.write_text(json.dumps(_regen_payload(ready=ready, steps_per_chunk=steps_per_chunk)), encoding="utf-8")
    gate.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    preflight.write_text(
        json.dumps({"status": "ready_for_observer_backed_execution_gate" if ready else "blocked", "observer_camera_status": "available"}),
        encoding="utf-8",
    )
    refresh.write_text(
        json.dumps(
            {
                "status": "ready_for_execution_gate" if ready else "blocked",
                "policy_camera_indexes": ["0", "1"],
                "regenerated_transition_path": str(regen),
                "transition_gate_path": str(gate),
                "observer_preflight_path": str(preflight),
            }
        ),
        encoding="utf-8",
    )
    return refresh


def _regen_payload(*, ready: bool, steps_per_chunk: int) -> dict:
    steps = []
    step_index = 0
    for chunk_index in range(2):
        for step_index_in_chunk in range(steps_per_chunk):
            steps.append(
                {
                    "step_index": step_index,
                    "chunk_index": chunk_index,
                    "step_index_in_chunk": step_index_in_chunk,
                    "joint_targets": [
                        {
                            "joint": joint,
                            "target_command_value": 1000.0 + step_index,
                            "target_raw": 2000.0 + step_index,
                        }
                        for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    ],
                }
            )
            step_index += 1
    return {
        "status": "passed",
        "live_readback_regenerated": ready,
        "transition_steps": steps,
    }
