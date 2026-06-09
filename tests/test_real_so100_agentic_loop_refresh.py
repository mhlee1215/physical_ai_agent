from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_agentic_loop_refresh import refresh_agentic_loop


class RealSO100AgenticLoopRefreshTest(TestCase):
    def test_refreshes_no_actuation_loop_from_episode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = _write_episode(tmp)
            contract = _write_contract_bundle(tmp)

            manifest = refresh_agentic_loop(
                contract=contract,
                output_dir=tmp / "gate",
                reports_dir=tmp / "reports",
                label="fixture_move_right",
                episode=episode,
                frame_index=1,
                calibration_file=tmp / "calibration.json",
                policy_camera_indexes=[0, 1],
                observer_camera_indexes=[3],
            )

            report_html = Path(manifest["loop_report_html"]).read_text(encoding="utf-8")
            refreshed_contract = json.loads(Path(manifest["contract"]).read_text(encoding="utf-8"))
            gate_exists = Path(manifest["gate_manifest"]).exists()
            contract_exists = Path(manifest["contract"]).exists()
            next_plan_exists = Path(manifest["next_plan"]).exists()

        self.assertEqual(manifest["status"], "passed")
        self.assertFalse(manifest["physical_robot_motion"])
        self.assertFalse(manifest["send_action_called"])
        self.assertEqual(manifest["next_stage"], "smolvla_proposal_only")
        self.assertTrue(gate_exists)
        self.assertTrue(contract_exists)
        self.assertTrue(next_plan_exists)
        self.assertIn("external setup blocker", report_html)
        self.assertIn("metadata_with_stats", refreshed_contract["adapter_and_safety"]["action_metadata_path"])
        self.assertTrue(refreshed_contract["adapter_and_safety"]["action_metadata"]["action_stats_available"])
        self.assertIn("execute_gate", refreshed_contract["adapter_and_safety"]["execute_gate_path"])
        self.assertFalse(refreshed_contract["adapter_and_safety"]["execute_gate_ready_for_execution"])


def _write_episode(root: Path) -> Path:
    camera_0 = root / "camera_0.jpg"
    camera_1 = root / "camera_1.jpg"
    camera_3 = root / "camera_3.jpg"
    _write_camera_0_clipped(camera_0)
    _write_camera_1_visible(camera_1)
    _write_camera_1_visible(camera_3)
    episode = root / "episode.jsonl"
    episode.write_text(
        json.dumps(
            {
                "frame_index": 1,
                "task": "Pick up the green Android figure and move it to the right.",
                "observation": {
                    "state": {
                        "shoulder_pan": 2376,
                        "shoulder_lift": 2047,
                        "elbow_flex": 2039,
                        "wrist_flex": 1988,
                        "wrist_roll": 2050,
                        "gripper": 1865,
                    },
                    "images": {"0": str(camera_0), "1": str(camera_1), "3": str(camera_3)},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return episode


def _write_contract_bundle(root: Path) -> Path:
    smolvla_report = _write_json(
        root / "smolvla_report.json",
        {
            "model_id": "lerobot/smolvla_base",
            "instruction": "Pick up the green Android figure and move it to the right.",
            "instruction_tokenized": True,
            "language_token_count": 14,
            "raw_action_dim": 6,
            "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
            "policy_camera_indexes": ["0", "1"],
            "observer_camera_indexes": ["3"],
            "observer_camera_role": "codex_debug_only_not_smolvla_input",
            "actuation_enabled": False,
            "send_action_called": False,
        },
    )
    smolvla_action = _write_json(
        root / "smolvla_action.json",
        {
            "instruction_tokenized": True,
            "raw_action": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "safe_to_execute": False,
        },
    )
    safety = _write_json(
        root / "safety.json",
        {
            "status": "passed",
            "execution_allowed": False,
            "human_confirmed": False,
            "blockers": ["Human confirmation is required before any real SO-100 action."],
        },
    )
    command = _write_json(
        root / "command.json",
        {
            "ready_for_execution": False,
            "human_confirmed": False,
            "blockers": ["Adapter semantics are not confirmed."],
        },
    )
    metadata = _write_json(
        root / "metadata_with_stats" / "smolvla_action_metadata_report.json",
        {
            "status": "blocked",
            "metadata": {
                "action_dim": 6,
                "action_normalization": "MEAN_STD",
                "action_stats_available": True,
                "output_is_normalized": True,
                "action_semantics": "absolute_joint_position",
                "joint_order": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ],
                "gripper_semantics": "higher_raw_opens",
                "stats_source": "lerobot_policy_postprocessor",
                "selected_action_stats_key": "so100.buffer",
                "command_units": None,
                "blockers": [
                    "Command units must be explicitly confirmed as feetech_raw_ticks before writing SO-100 Goal_Position."
                ],
            },
            "required_next_steps": [
                "Confirm the postprocessed action units and conversion to Feetech raw Goal_Position ticks."
            ],
        },
    )
    execute_gate = _write_json(
        root / "execute_gate" / "dry_gate.json",
        {
            "status": "dry_run",
            "command_units": "lerobot_so100_position",
            "blockers": ["Human confirmation flag is required."],
            "dry_plan": {
                "ready_for_execution": False,
                "blockers": [
                    "Step 0 joint shoulder_lift maps to raw target 4771.0 outside calibrated range [2001.0, 3695.0]."
                ],
            },
        },
    )
    grasp = _write_json(root / "grasp.json", {"grasp_outcome": "grasp_failed_object_stationary"})
    pack = _write_json(
        root / "pack.json",
        {
            "movement_report_html": "movement.html",
            "gate_report_html": "gate.html",
            "agentic_lessons": [],
        },
    )
    return _write_json(
        root / "contract.json",
        {
            "policy": {
                "instruction": "Pick up the green Android figure and move it to the right.",
                "instruction_tokenized": True,
                "report_path": str(smolvla_report),
                "action_path": str(smolvla_action),
            },
            "task_goal": {
                "instruction": "Pick up the green Android figure and move it to the right.",
            },
            "adapter_and_safety": {
                "safety_report_path": str(safety),
                "command_plan_path": str(command),
                "action_metadata_path": str(metadata),
                "execute_gate_path": str(execute_gate),
            },
            "evidence": {
                "grasp_outcome": str(grasp),
                "pre_stage_pack": str(pack),
            },
        },
    )


def _write_camera_0_clipped(path: Path) -> None:
    import cv2
    import numpy as np

    image = np.full((180, 260, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (-10, -10), (50, 70), (0, 190, 55), thickness=-1)
    cv2.rectangle(image, (120, 130), (180, 178), (160, 80, 20), thickness=-1)
    cv2.imwrite(str(path), image)


def _write_camera_1_visible(path: Path) -> None:
    import cv2
    import numpy as np

    image = np.full((180, 260, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (105, 65), (155, 125), (0, 190, 55), thickness=-1)
    cv2.imwrite(str(path), image)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
