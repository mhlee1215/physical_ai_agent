from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_policy_camera_feedback import build_policy_camera_feedback


class BuildRealSO100PolicyCameraFeedbackTest(TestCase):
    def test_builds_observer_off_pseudo_llm_feedback_for_smolvla(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text(
                json.dumps(
                    {
                        "frame_index": 2,
                        "observation": {
                            "state": {"gripper": 1788},
                            "state_available": True,
                            "state_source": "live_so100_readback",
                            "images": {"0": "camera_0.jpg", "1": "camera_1.jpg"},
                            "image_shapes": {"0": [1080, 1920, 3], "1": [1080, 1920, 3]},
                            "camera_roles": {"0": "wrist_cam", "1": "egocentric_wide_context"},
                            "policy_camera_indexes": [0, 1],
                            "observer_camera_indexes": [],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = tmp / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "episode_jsonl": str(episode),
                        "policy_camera_indexes": [0, 1],
                        "observer_camera_indexes": [],
                        "send_action_called": False,
                        "policy_actions_executed": False,
                    }
                ),
                encoding="utf-8",
            )

            report = build_policy_camera_feedback(
                observation_manifest=manifest,
                output=tmp / "feedback.json",
                observations=["camera 1 sees the target and robot context"],
                diagnosis=["use camera 1 for coarse approach context"],
                next_prompt="Use the wide context view to approach the green figure before grasping.",
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["camera_contract"]["policy_camera_indexes"], [0, 1])
        self.assertEqual(report["camera_contract"]["observer_camera_indexes"], [])
        self.assertFalse(report["camera_contract"]["camera_3_is_policy_input"])
        self.assertTrue(report["pseudo_llm_feedback"]["does_not_prompt_operator"])
        self.assertEqual(report["pseudo_llm_feedback"]["target"], "in_loop_agent_or_smolvla")
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "run_no_actuation_smolvla_from_policy_camera_feedback",
        )
        self.assertFalse(report["execution_outcome"]["send_action_called"])
        self.assertFalse(report["task_success_claim_allowed"])

    def test_blocks_without_feedback_text(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = tmp / "episode.jsonl"
            episode.write_text("", encoding="utf-8")
            manifest = tmp / "manifest.json"
            manifest.write_text(
                json.dumps({"ok": True, "episode_jsonl": str(episode), "policy_camera_indexes": [0, 1]}),
                encoding="utf-8",
            )

            report = build_policy_camera_feedback(
                observation_manifest=manifest,
                output=tmp / "feedback.json",
                observations=[],
                diagnosis=[],
                next_prompt="",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["next_agentic_layer_step"]["type"], "repair_policy_camera_feedback_capture")
