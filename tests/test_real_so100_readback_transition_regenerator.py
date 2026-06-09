from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_readback_transition_regenerator import regenerate_transition_from_readback


class RealSO100ReadbackTransitionRegeneratorTest(TestCase):
    def test_regenerates_transition_from_read_only_probe_payload(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            readback = tmp / "readback.json"
            bridge.write_text(json.dumps(_bridge_report()), encoding="utf-8")
            readback.write_text(json.dumps({"positions_raw": _state(1000.0)}), encoding="utf-8")

            report = regenerate_transition_from_readback(
                bridge_report=bridge,
                readback=readback,
                output=tmp / "regen.json",
                readback_source="live",
                max_abs_raw_delta_per_step=80.0,
                chunk_size=10,
            )

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["live_readback_regenerated"])
        self.assertEqual(report["transition_chunk_count"], 1)
        self.assertEqual(report["transition_step_count"], 10)
        self.assertEqual(report["next_agentic_layer_step"]["type"], "run_transition_candidate_gate_on_live_readback_plan")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])

    def test_replay_readback_requires_live_rerun_before_execution(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            readback = tmp / "readback.json"
            bridge.write_text(json.dumps(_bridge_report(target_raw=1900.0)), encoding="utf-8")
            readback.write_text(json.dumps({"observation": {"state": _state(1000.0)}}), encoding="utf-8")

            report = regenerate_transition_from_readback(
                bridge_report=bridge,
                readback=readback,
                output=tmp / "regen.json",
                readback_source="replay",
                max_abs_raw_delta_per_step=80.0,
                chunk_size=10,
            )

        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["live_readback_regenerated"])
        self.assertEqual(report["transition_chunk_count"], 2)
        self.assertEqual(report["next_agentic_layer_step"]["type"], "rerun_with_live_readback_before_execution")

    def test_blocks_missing_readback_joint(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            readback = tmp / "readback.json"
            state = _state(1000.0)
            state.pop("gripper")
            bridge.write_text(json.dumps(_bridge_report()), encoding="utf-8")
            readback.write_text(json.dumps({"positions_raw": state}), encoding="utf-8")

            report = regenerate_transition_from_readback(
                bridge_report=bridge,
                readback=readback,
                output=tmp / "regen.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("missing joints" in blocker for blocker in report["blockers"]))
        self.assertFalse(report["send_action_called"])

    def test_reads_jsonl_episode_frame(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bridge = tmp / "bridge.json"
            readback = tmp / "episode.jsonl"
            bridge.write_text(json.dumps(_bridge_report()), encoding="utf-8")
            readback.write_text(
                json.dumps({"frame_index": 0, "observation": {"state": _state(900.0)}})
                + "\n"
                + json.dumps({"frame_index": 1, "observation": {"state": _state(1000.0)}})
                + "\n",
                encoding="utf-8",
            )

            report = regenerate_transition_from_readback(
                bridge_report=bridge,
                readback=readback,
                output=tmp / "regen.json",
                frame_index=1,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["source_current_raw"]["shoulder_pan"], 1000.0)


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
