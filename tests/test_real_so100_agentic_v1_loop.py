from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.real_so100_agentic_v1_loop import (
    build_feedback_packet,
    run_agentic_v1_loop,
    validate_pseudo_llm_decision,
)


class RealSO100AgenticV1LoopTest(TestCase):
    def test_validates_pseudo_llm_decision_without_rule_semantics(self) -> None:
        decision = _decision()

        result = validate_pseudo_llm_decision(decision)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["blockers"], [])

    def test_blocks_decision_when_limited_delta_exceeds_bound(self) -> None:
        decision = _decision()
        decision["limited_step"]["manual_delta_raw"] = 8.0
        decision["limited_step"]["max_abs_delta_raw"] = 2.0

        result = validate_pseudo_llm_decision(decision)

        self.assertEqual(result["status"], "blocked")
        self.assertIn("manual_delta_raw exceeds max_abs_delta_raw", result["blockers"])

    def test_loop_records_pre_step_post_and_feedback_contract(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            decision_path = tmp / "decision.json"
            decision_path.write_text(json.dumps(_decision()), encoding="utf-8")

            with patch("scripts.real_so100_agentic_v1_loop.record_observation") as observe, patch(
                "scripts.real_so100_agentic_v1_loop.run_micro_step"
            ) as micro:
                observe.side_effect = [
                    {
                        "ok": True,
                        "episode_jsonl": str(tmp / "pre" / "episode.jsonl"),
                        "frames_recorded": 2,
                        "send_action_called": False,
                    },
                    {
                        "ok": True,
                        "episode_jsonl": str(tmp / "post" / "episode.jsonl"),
                        "frames_recorded": 2,
                        "send_action_called": False,
                    },
                ]
                micro.return_value = {
                    "status": "passed",
                    "joint": "shoulder_pan",
                    "planned_delta_raw": -5.0,
                    "observed_delta_raw": -4.0,
                    "target_error_raw": -1.0,
                    "send_action_called": True,
                    "camera_index": 3,
                    "visual_check": {
                        "before": {"image_path": str(tmp / "observer" / "before.jpg")},
                        "after": {
                            "image_path": str(tmp / "observer" / "after.jpg"),
                            "mean_absdiff": 3.5,
                            "visual_motion_detected": True,
                        },
                    },
                    "motion_video": {"path": str(tmp / "observer" / "motion.mp4")},
                }

                manifest = run_agentic_v1_loop(
                    decision=decision_path,
                    output_dir=tmp / "run",
                    port="/dev/cu.fake",
                    calibration_file=tmp / "calibration.json",
                    execute=True,
                    human_confirmed=True,
                    policy_camera_indexes=[0, 1],
                    observer_camera_index=3,
                )

                self.assertEqual(manifest["status"], "passed")
                self.assertTrue(manifest["send_action_called"])
                self.assertTrue(manifest["physical_robot_motion"])
                self.assertEqual(observe.call_count, 2)
                self.assertEqual(micro.call_args.kwargs["camera_index"], 3)
                self.assertEqual(micro.call_args.kwargs["visual_output_dir"], tmp / "run" / "observer_camera_3")
                self.assertTrue(micro.call_args.kwargs["record_video"])
                self.assertEqual(manifest["feedback_summary"]["observer_visual_motion_detected"], True)
                self.assertTrue((tmp / "run" / "feedback_packet.json").exists())

    def test_feedback_packet_keeps_semantic_judgment_for_llm(self) -> None:
        feedback = build_feedback_packet(
            decision=_decision(),
            micro_report={
                "status": "passed",
                "joint": "shoulder_pan",
                "planned_delta_raw": -5.0,
                "observed_delta_raw": -4.0,
                "target_error_raw": -1.0,
                "send_action_called": True,
                "camera_index": 3,
                "visual_check": {"after": {"visual_motion_detected": True}},
            },
            pre_observe={"episode_jsonl": "pre.jsonl", "frames_recorded": 2},
            post_observe={"episode_jsonl": "post.jsonl", "frames_recorded": 2},
        )

        self.assertTrue(feedback["summary"]["needs_pseudo_llm_or_on_device_llm_feedback"])
        self.assertEqual(
            feedback["next_version_input"]["semantic_judgment_owner"],
            "pseudo_llm_during_development_on_device_llm_at_runtime",
        )


def _decision() -> dict:
    return {
        "backend": "codex_pseudo_llm_v0",
        "target_runtime": "on_device_llm_or_vlm",
        "agentic_layer_version": "v1",
        "task": "Pick up the green Android figure and move it to the right.",
        "scene_interpretation": {
            "camera_0": "target is clipped in wrist view",
            "camera_1": "target and arm are visible in global view",
        },
        "selected_subgoal": "improve_wrist_view_target_framing",
        "smolvla_prompt": "Move the arm slightly so the green figure becomes more centered in the wrist camera.",
        "limited_step": {
            "joint": "shoulder_pan",
            "manual_delta_raw": -5.0,
            "max_abs_delta_raw": 5.0,
            "non_contact_confirmed": True,
            "contact_ok_for_gripper": False,
        },
        "expected_observer_evidence": {
            "camera": "3",
            "description": "arm changes pose without contacting the object",
        },
    }
