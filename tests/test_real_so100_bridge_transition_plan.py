from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_bridge_transition_plan import build_bridge_transition_plan


class RealSO100BridgeTransitionPlanTest(TestCase):
    def test_builds_bounded_no_actuation_transition_to_bridge_pose(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            episode = tmp / "episode.jsonl"
            bridge.write_text(json.dumps(_bridge_report()), encoding="utf-8")
            episode.write_text(json.dumps({"frame_index": 0, "observation": {"state": _state(1000.0)}}) + "\n", encoding="utf-8")

            report = build_bridge_transition_plan(
                bridge_report=bridge,
                episode=episode,
                frame_index=0,
                output=tmp / "transition.json",
                step_count=10,
                max_abs_raw_delta_per_step=80.0,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["transition_step_count"], 10)
        self.assertEqual(len(report["transition_steps"]), 10)
        self.assertEqual(report["transition_steps"][-1]["joint_targets"][0]["target_raw"], 1100.0)
        self.assertTrue(all(target["raw_target_in_calibrated_range"] for target in report["transition_steps"][-1]["joint_targets"]))
        self.assertEqual(report["next_agentic_layer_step"]["type"], "run_projection_and_trajectory_diagnostics_on_transition_candidate")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])

    def test_blocks_when_step_delta_is_too_large(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            episode = tmp / "episode.jsonl"
            bridge.write_text(json.dumps(_bridge_report(target_raw=1900.0)), encoding="utf-8")
            episode.write_text(json.dumps({"frame_index": 0, "observation": {"state": _state(1000.0)}}) + "\n", encoding="utf-8")

            report = build_bridge_transition_plan(
                bridge_report=bridge,
                episode=episode,
                frame_index=0,
                output=tmp / "transition.json",
                step_count=10,
                max_abs_raw_delta_per_step=80.0,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("above limit", report["blockers"][0])
        self.assertFalse(report["send_action_called"])

    def test_auto_chunks_preserves_ten_step_chunks_while_reducing_delta(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            episode = tmp / "episode.jsonl"
            bridge.write_text(json.dumps(_bridge_report(target_raw=1900.0)), encoding="utf-8")
            episode.write_text(json.dumps({"frame_index": 0, "observation": {"state": _state(1000.0)}}) + "\n", encoding="utf-8")

            report = build_bridge_transition_plan(
                bridge_report=bridge,
                episode=episode,
                frame_index=0,
                output=tmp / "transition.json",
                step_count=10,
                max_abs_raw_delta_per_step=80.0,
                auto_chunks=True,
                chunk_size=10,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["transition_chunk_count"], 2)
        self.assertEqual(report["transition_step_count"], 20)
        self.assertEqual(report["transition_steps"][10]["chunk_index"], 1)
        self.assertEqual(report["transition_steps"][10]["step_index_in_chunk"], 0)
        self.assertLessEqual(abs(report["delta_summary"]["shoulder_pan"]["per_step_delta_raw"]), 80.0)
        self.assertFalse(report["send_action_called"])


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
