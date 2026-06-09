from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_agentic_iteration import run_agentic_iteration


class RealSO100AgenticIterationTest(TestCase):
    def test_runs_full_no_actuation_iteration_from_episode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = _write_episode(tmp)
            contract = _write_contract_bundle(tmp)
            state = tmp / "state.json"
            vla_prompt_packet = _write_json(tmp / "vla_prompt_packet.json", {"vla_prompt": {"target": "SmolVLA"}})

            manifest = run_agentic_iteration(
                prompt="녹색 인형을 집어서 오른쪽으로 옮겨줘",
                contract=contract,
                output_dir=tmp / "gate",
                reports_dir=tmp / "reports",
                label="fixture_iteration",
                state=state,
                iteration_index=3,
                episode=episode,
                frame_index=1,
                calibration_file=tmp / "calibration.json",
                policy_camera_indexes=[0, 1],
                observer_camera_indexes=[3],
                vla_prompt_packet=vla_prompt_packet,
            )

            prompt_iteration = json.loads(Path(manifest["prompt_iteration"]).read_text(encoding="utf-8"))
            stateful_next_plan = json.loads(Path(manifest["stateful_next_plan"]).read_text(encoding="utf-8"))
            policy_patch = json.loads(Path(manifest["agentic_policy_patch"]).read_text(encoding="utf-8"))
            state_payload = json.loads(state.read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "passed")
        self.assertFalse(manifest["physical_robot_motion"])
        self.assertFalse(manifest["send_action_called"])
        self.assertEqual(manifest["vla_prompt_packet"], str(vla_prompt_packet))
        self.assertEqual(stateful_next_plan["vla_prompt_packet"], str(vla_prompt_packet))
        self.assertEqual(prompt_iteration["agentic_policy_patch"], manifest["agentic_policy_patch"])
        self.assertIn("policy_input_quality_gate", manifest["policy_patch_rules"])
        self.assertEqual(policy_patch["prompt_contract"]["vla_prompt_target"], "SmolVLA")
        self.assertTrue(policy_patch["prompt_contract"]["does_not_prompt_operator"])
        self.assertEqual(manifest["gate_status"], "blocked")
        self.assertEqual(manifest["next_stage"], "smolvla_proposal_only")
        self.assertEqual(manifest["next_step_type"], "rerun_smolvla_dry")
        self.assertFalse(manifest["task_success_claim_allowed"])
        self.assertEqual(prompt_iteration["camera_contract"]["smolvla_policy_inputs"], ["0", "1"])
        self.assertEqual(prompt_iteration["camera_contract"]["observer_inputs"], ["3"])
        self.assertIn("jaw_object_framing_not_ready", manifest["failure_modes"])
        self.assertIn("external_setup_ready_before_contact", state_payload["active_constraints"])

    def test_runs_no_actuation_iteration_when_observer_camera_is_off(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = _write_episode(tmp, include_observer=False)
            contract = _write_contract_bundle(tmp, observer_indexes=[])
            state = tmp / "state.json"

            manifest = run_agentic_iteration(
                prompt="녹색 인형을 집어서 오른쪽으로 옮겨줘",
                contract=contract,
                output_dir=tmp / "gate",
                reports_dir=tmp / "reports",
                label="observer_off_iteration",
                state=state,
                iteration_index=4,
                episode=episode,
                frame_index=1,
                calibration_file=tmp / "calibration.json",
                policy_camera_indexes=[0, 1],
                observer_camera_indexes=[],
            )

            prompt_iteration = json.loads(Path(manifest["prompt_iteration"]).read_text(encoding="utf-8"))
            stateful_next_plan = json.loads(Path(manifest["stateful_next_plan"]).read_text(encoding="utf-8"))

        self.assertEqual(manifest["observer_camera_status"], "temporarily_unavailable")
        self.assertFalse(manifest["physical_robot_motion"])
        self.assertFalse(manifest["send_action_called"])
        self.assertEqual(prompt_iteration["camera_contract"]["smolvla_policy_inputs"], ["0", "1"])
        self.assertEqual(prompt_iteration["camera_contract"]["observer_inputs"], [])
        self.assertEqual(prompt_iteration["camera_contract"]["observer_camera_status"], "temporarily_unavailable")
        self.assertEqual(stateful_next_plan.get("observer_camera_status"), "temporarily_unavailable")
        self.assertEqual(stateful_next_plan["stage"], "smolvla_proposal_only")
        self.assertEqual(
            stateful_next_plan["autonomous_next_steps"][0]["observer_camera_indexes_excluded_from_policy"],
            [],
        )


def _write_episode(root: Path, *, include_observer: bool = True) -> Path:
    camera_0 = root / "camera_0.jpg"
    camera_1 = root / "camera_1.jpg"
    camera_3 = root / "camera_3.jpg"
    _write_camera_0_clipped(camera_0)
    _write_camera_1_visible(camera_1)
    images = {"0": str(camera_0), "1": str(camera_1)}
    if include_observer:
        _write_camera_1_visible(camera_3)
        images["3"] = str(camera_3)
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
                    "images": images,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return episode


def _write_contract_bundle(root: Path, *, observer_indexes: list[str] | None = None) -> Path:
    observer_indexes = ["3"] if observer_indexes is None else observer_indexes
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
            "observer_camera_indexes": observer_indexes,
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
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
