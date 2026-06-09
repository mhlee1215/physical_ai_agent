from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.real_so100_observer_return_refresh import run_observer_return_refresh


class RealSO100ObserverReturnRefreshTest(TestCase):
    def test_replay_refresh_runs_chain_but_stays_blocked(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            readback = tmp / "episode.jsonl"
            bridge.write_text(json.dumps(_bridge_report()), encoding="utf-8")
            readback.write_text(json.dumps({"frame_index": 0, "observation": {"state": _state(1000.0)}}) + "\n", encoding="utf-8")

            report = run_observer_return_refresh(
                bridge_report=bridge,
                output_dir=tmp / "refresh",
                port="/dev/null",
                mode="replay",
                replay_readback=readback,
                frame_index=0,
                observer_camera_status="off",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["readback_source"], "replay")
        self.assertFalse(report["live_readback_regenerated"])
        self.assertEqual(report["regenerated_transition_status"], "passed")
        self.assertEqual(report["transition_gate_status"], "passed")
        self.assertEqual(report["observer_preflight_status"], "blocked")
        self.assertEqual(report["next_agentic_layer_step"]["type"], "rerun_refresh_in_live_readonly_mode_when_camera_3_returns")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])

    def test_live_readonly_refresh_can_reach_execution_gate_preflight(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            bridge.write_text(json.dumps(_bridge_report()), encoding="utf-8")

            def fake_probe(_port: str, output: Path) -> dict:
                payload = {"ok": True, "positions_raw": _state(1000.0)}
                output.write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with patch("scripts.real_so100_observer_return_refresh.run_probe", side_effect=fake_probe):
                report = run_observer_return_refresh(
                    bridge_report=bridge,
                    output_dir=tmp / "refresh",
                    port="/dev/cu.fake",
                    mode="live_readonly",
                    observer_camera_status="available",
                    user_confirmed=True,
                    workspace_clear_confirmed=True,
                )

        self.assertEqual(report["status"], "ready_for_execution_gate")
        self.assertTrue(report["live_readback_regenerated"])
        self.assertEqual(report["observer_preflight_status"], "ready_for_observer_backed_execution_gate")
        self.assertEqual(report["next_agentic_layer_step"]["type"], "build_observer_backed_execution_report")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])

    def test_replay_mode_requires_replay_readback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            bridge.write_text(json.dumps(_bridge_report()), encoding="utf-8")

            with self.assertRaises(ValueError):
                run_observer_return_refresh(
                    bridge_report=bridge,
                    output_dir=tmp / "refresh",
                    port="/dev/null",
                    mode="replay",
                )


def _state(raw: float) -> dict[str, float]:
    return {
        "shoulder_pan": raw,
        "shoulder_lift": raw,
        "elbow_flex": raw,
        "wrist_flex": raw,
        "wrist_roll": raw,
        "gripper": raw,
    }


def _bridge_report(target_raw: float = 1100.0) -> dict:
    return {
        "status": "passed",
        "policy_camera_indexes": ["0", "1"],
        "observer_camera_indexes": [],
        "observer_camera_status": "temporarily_unavailable",
        "camera_3_status": "off",
        "bridge_target_joints": [
            {
                "joint": joint,
                "finite": True,
                "projected_raw": target_raw,
                "range_min": 0.0,
                "range_max": 4095.0,
                "was_out_of_range": False,
            }
            for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        ],
    }
