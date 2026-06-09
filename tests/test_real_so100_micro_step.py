import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_micro_step import _probe_motion_video, run_micro_step


class RealSO100MicroStepTest(TestCase):
    def test_dry_run_does_not_connect_or_send_action(self) -> None:
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            output = Path(tmpdir) / "report.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "ready_for_execution": False,
                        "joint_plans": [
                            {
                                "joint": "wrist_roll",
                                "current_raw": 2050,
                                "target_raw": 2051,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_micro_step(
                port="/dev/cu.fake",
                command_plan=plan_path,
                joint="wrist_roll",
                output=output,
                execute=False,
                human_confirmed=False,
                non_contact_confirmed=False,
                contact_ok_for_gripper=False,
                max_abs_delta_raw=2.0,
                settle_seconds=0.0,
            )
            output_exists = output.exists()

        self.assertEqual(report["status"], "dry_run")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["policy_actions_executed"])
        self.assertTrue(output_exists)

    def test_execute_is_blocked_before_connection_when_plan_not_ready(self) -> None:
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            output = Path(tmpdir) / "report.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "ready_for_execution": False,
                        "joint_plans": [
                            {
                                "joint": "wrist_roll",
                                "current_raw": 2050,
                                "target_raw": 2051,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_micro_step(
                port="/dev/cu.fake",
                command_plan=plan_path,
                joint="wrist_roll",
                output=output,
                execute=True,
                human_confirmed=True,
                non_contact_confirmed=True,
                contact_ok_for_gripper=False,
                max_abs_delta_raw=2.0,
                settle_seconds=0.0,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["send_action_called"])
        self.assertIn("Command plan is not marked ready_for_execution=true.", report["blockers"])

    def test_large_delta_blocks_before_connection(self) -> None:
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            output = Path(tmpdir) / "report.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "ready_for_execution": True,
                        "joint_plans": [
                            {
                                "joint": "wrist_roll",
                                "current_raw": 2050,
                                "target_raw": 2080,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_micro_step(
                port="/dev/cu.fake",
                command_plan=plan_path,
                joint="wrist_roll",
                output=output,
                execute=True,
                human_confirmed=True,
                non_contact_confirmed=True,
                contact_ok_for_gripper=False,
                max_abs_delta_raw=2.0,
                settle_seconds=0.0,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["send_action_called"])
        self.assertIn("exceeds max_abs_delta_raw", report["blockers"][0])

    def test_execute_requires_video_and_visual_evidence_before_connection(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.json"

            report = run_micro_step(
                port="/dev/cu.fake",
                command_plan=None,
                joint="wrist_roll",
                output=output,
                execute=True,
                human_confirmed=True,
                non_contact_confirmed=True,
                contact_ok_for_gripper=False,
                max_abs_delta_raw=2.0,
                settle_seconds=0.0,
                manual_delta_raw=1.0,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["send_action_called"])
        self.assertTrue(report["visual_check_required"])
        self.assertTrue(report["motion_video_required"])
        self.assertIn("Executed real robot movements must pass --record-video.", report["blockers"])
        self.assertIn("Executed real robot movements must pass --camera-index for visual/video evidence.", report["blockers"])
        self.assertIn("Executed real robot movements must pass --visual-output-dir for visual/video evidence.", report["blockers"])

    def test_manual_delta_dry_run_does_not_need_command_plan(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.json"

            report = run_micro_step(
                port="/dev/cu.fake",
                command_plan=None,
                joint="wrist_roll",
                output=output,
                execute=False,
                human_confirmed=True,
                non_contact_confirmed=True,
                contact_ok_for_gripper=False,
                max_abs_delta_raw=1.0,
                settle_seconds=0.0,
                manual_delta_raw=1.0,
            )

        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["manual_delta_raw"], 1.0)
        self.assertFalse(report["send_action_called"])

    def test_gripper_contact_probe_can_bypass_non_contact_confirmation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.json"

            report = run_micro_step(
                port="/dev/cu.fake",
                command_plan=None,
                joint="gripper",
                output=output,
                execute=True,
                human_confirmed=True,
                non_contact_confirmed=False,
                contact_ok_for_gripper=True,
                max_abs_delta_raw=5.0,
                settle_seconds=0.0,
                manual_delta_raw=-10.0,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("exceeds max_abs_delta_raw", report["blockers"][0])
        self.assertNotIn("Non-contact workspace confirmation flag is required.", report["blockers"])
        self.assertTrue(report["contact_probe_allowed"])

    def test_video_recording_request_is_reported_without_dry_run_capture(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.json"

            report = run_micro_step(
                port="/dev/cu.fake",
                command_plan=None,
                joint="wrist_roll",
                output=output,
                execute=False,
                human_confirmed=True,
                non_contact_confirmed=True,
                contact_ok_for_gripper=False,
                max_abs_delta_raw=1.0,
                settle_seconds=0.0,
                manual_delta_raw=1.0,
                record_video=True,
            )

        self.assertEqual(report["status"], "dry_run")
        self.assertTrue(report["record_video_requested"])
        self.assertFalse(report["motion_video_required"])
        self.assertNotIn("motion_video", report)

    def test_probe_motion_video_records_codec_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "motion.mp4"
            _write_video(video)

            result = _probe_motion_video(video)

        self.assertTrue(result["exists"])
        self.assertIn("actual_codec", result)
        self.assertGreaterEqual(result["actual_frame_count"], 1)
        self.assertTrue(result["first_frame_readable"])
        self.assertIn("browser_preview_recommended", result)


def _write_video(path: Path) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (80, 60))
    if not writer.isOpened():
        raise RuntimeError("failed to open test video writer")
    for index in range(4):
        frame = np.full((60, 80, 3), 240, dtype=np.uint8)
        cv2.rectangle(frame, (10 + index * 5, 20), (35 + index * 5, 45), (0, 190, 55), thickness=-1)
        writer.write(frame)
    writer.release()
