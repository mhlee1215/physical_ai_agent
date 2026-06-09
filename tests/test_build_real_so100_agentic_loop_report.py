from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_agentic_loop_report import build_agentic_loop_report


class BuildRealSO100AgenticLoopReportTest(TestCase):
    def test_builds_html_with_reframe_advice_and_success_evidence(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image = tmp / "camera_0.jpg"
            _write_image(image)
            pregrasp = _write_json(
                tmp / "pregrasp.json",
                {
                    "assessments": [
                        {
                            "camera": "0",
                            "image_path": str(image),
                            "bbox_xyxy": [0, 0, 40, 50],
                            "edge_clipped": True,
                            "usable_for_pregrasp": False,
                        }
                    ]
                },
            )
            advice = _write_json(
                tmp / "advice.json",
                {
                    "pregrasp_probe": str(pregrasp),
                    "actions": [
                        {
                            "diagnostic_summary": "camera 0 target detection is clipped; external setup blocker. Image-space diagnostic: shift target appearance about 32px right.",
                            "image_space_nudge": {
                                "recommended_shift_px": [32.0, 0.0],
                                "target_margin_px": 32,
                                "current_bbox_xyxy": [0.0, 102.0, 206.0, 512.0],
                            },
                            "agent_actionable": False,
                        }
                    ],
                },
            )
            contract = _write_json(
                tmp / "contract.json",
                {
                    "policy": {
                        "instruction": "Pick up the green Android figure and move it to the right.",
                        "policy_camera_indexes": ["0", "1"],
                        "observer_camera_indexes": ["3"],
                    },
                    "task_goal": {"transport_direction": "right"},
                    "agentic_layer": {
                        "blockers": ["camera 0 jaw/object framing gate is not ready"],
                        "verifier_contract": {
                            "pregrasp_gate_status": "passed",
                            "jaw_gate_status": "blocked",
                            "last_grasp_outcome": "grasp_failed_object_stationary",
                            "relocation_verifier_status": "not_run",
                        },
                    },
                },
            )
            plan = _write_json(
                tmp / "plan.json",
                {
                    "stage": "external_setup_blocked",
                    "physical_robot_motion": False,
                    "next_steps": [],
                    "external_setup_blocker": {
                        "type": "external_setup_blocker",
                        "agent_actionable": False,
                        "vla_prompt_allowed": False,
                        "why_not_agent_action": "camera/object framing is outside the robot policy action space",
                        "diagnostics": [
                            {
                                "diagnostic_summary": "camera 0 target detection is clipped; external setup blocker. Image-space diagnostic: shift target appearance about 32px right.",
                                "image_space_nudge": {
                                    "recommended_shift_px": [32.0, 0.0],
                                    "target_margin_px": 32,
                                    "current_bbox_xyxy": [0.0, 102.0, 206.0, 512.0],
                                },
                            }
                        ],
                    },
                    "post_external_setup_verification": [
                        {"type": "rerun_no_actuation_gate_after_external_setup_change", "physical_robot_motion": False}
                    ],
                    "required_evidence_before_success_claim": ["object_relocation_verifier"],
                    "guardrails": ["Do not execute raw SmolVLA actions directly."],
                },
            )
            prompt_iteration = _write_json(
                tmp / "prompt_iteration.json",
                {
                    "agentic_policy_patch": str(tmp / "policy_patch.json"),
                    "success_accounting": {"task_success_claim_allowed": False},
                    "next_iteration": {"stage": "external_setup_blocked", "next_step_type": None},
                },
            )
            policy_patch = _write_json(
                tmp / "policy_patch.json",
                {
                    "prompt_contract": {"vla_prompt_target": "SmolVLA", "does_not_prompt_operator": True},
                    "success_accounting": {"task_success_claim_allowed": False},
                    "rules": [
                        {
                            "id": "policy_input_quality_gate",
                            "action": "block SmolVLA prompting and contact execution",
                        }
                    ],
                    "legacy_normalizations": [
                        {"from": "observation_repair_policy", "to": "policy_input_quality_gate"}
                    ],
                },
            )
            ready_fixture = _write_json(
                tmp / "ready_fixture.json",
                {
                    "status": "passed",
                    "physical_robot_motion": False,
                    "next_step_types": [
                        "execute_video_backed_contact_probe",
                        "materialize_relocation_verifier_packet",
                    ],
                    "checks": [{"name": "step_order", "passed": True}],
                },
            )

            manifest = build_agentic_loop_report(
                contract=contract,
                next_plan=plan,
                reframe_advice=advice,
                output=tmp / "report.html",
                prompt_iteration=prompt_iteration,
                policy_patch=policy_patch,
                ready_path_fixture=ready_fixture,
            )
            html = (tmp / "report.html").read_text(encoding="utf-8")

        self.assertEqual(manifest["stage"], "external_setup_blocked")
        self.assertFalse(manifest["physical_robot_motion"])
        self.assertIn("external setup blocker", html)
        self.assertIn("image-space diagnostic only", html)
        self.assertIn("32px too far left", html)
        self.assertNotIn("shift target appearance", html)
        self.assertIn("VLA prompt allowed", html)
        self.assertIn("object_relocation_verifier", html)
        self.assertIn("camera 0 jaw/object framing gate is not ready", html)
        self.assertIn("policy_input_quality_gate", html)
        self.assertIn("SmolVLA", html)
        self.assertIn("materialize_relocation_verifier_packet", html)
        self.assertEqual(manifest["agentic_policy_patch"], str(Path(tmpdir) / "policy_patch.json"))
        self.assertEqual(manifest["policy_patch_rules"], ["policy_input_quality_gate"])
        self.assertEqual(manifest["ready_path_step_types"][1], "materialize_relocation_verifier_packet")
        self.assertFalse(manifest["task_success_claim_allowed"])


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_image(path: Path) -> None:
    import cv2
    import numpy as np

    image = np.full((80, 120, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (40, 50), (0, 190, 55), thickness=-1)
    cv2.imwrite(str(path), image)
