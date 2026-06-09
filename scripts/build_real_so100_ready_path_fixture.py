#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.real_so100_agentic_controller import build_agentic_next_plan


def build_ready_path_fixture(
    *,
    output_dir: Path,
    reports_dir: Path,
    vla_prompt_packet: Path,
    output: Path | None = None,
    contact_output_dir: str = "_workspace/real_so100/contact_probe_ready_path_fixture",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    contract = _write_json(output_dir / "ready_contract.json", _ready_contract())
    plan_path = reports_dir / "real_so100_agentic_next_plan_ready_path_fixture.json"
    plan = build_agentic_next_plan(
        contract=contract,
        output=plan_path,
        contact_output_dir=contact_output_dir,
        vla_prompt_packet=vla_prompt_packet,
    )
    checks = _check_ready_plan(plan=plan, contact_output_dir=contact_output_dir, vla_prompt_packet=vla_prompt_packet)
    manifest = {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "operation": "real_so100_ready_path_fixture",
        "purpose": "prove the non-hardware ready path from SmolVLA prompt packet to observer-frame relocation verifier",
        "contract": str(contract),
        "next_plan": str(plan_path),
        "vla_prompt_packet": str(vla_prompt_packet),
        "contact_output_dir": contact_output_dir,
        "physical_robot_motion": False,
        "send_action_called": False,
        "checks": checks,
        "stage": plan.get("stage"),
        "next_step_types": [step.get("type") for step in plan.get("next_steps", [])],
        "notes": [
            "This fixture does not execute the robot.",
            "It verifies the controller ordering and file-path contract used once the real gate becomes ready.",
        ],
    }
    manifest_path = output or (reports_dir / "real_so100_ready_path_fixture.json")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _ready_contract() -> dict[str, Any]:
    return {
        "manifest_path": "ready_contract.json",
        "policy": {
            "instruction": "Pick up the green Android figure and move it to the right.",
            "instruction_tokenized": True,
            "policy_camera_indexes": ["0", "1"],
            "observer_camera_indexes": ["3"],
            "observer_camera_role": "codex_debug_only_not_smolvla_input",
            "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
        },
        "task_goal": {
            "instruction": "Pick up the green Android figure and move it to the right.",
            "target_object": "green Android figure",
            "transport_direction": "right",
            "requires_grasp": True,
            "requires_transport": True,
            "final_success_verifier": "object_relocation_image_space",
        },
        "agentic_layer": {
            "decision": "ready_for_reframed_contact_probe",
            "next_agentic_action": {
                "type": "minimal_contact_probe",
                "joint": "gripper",
                "reason": "policy inputs ready; execute minimal observer-backed contact probe",
            },
            "verifier_contract": {
                "relocation_task_success_candidate": False,
            },
        },
        "adapter_and_safety": {
            "command_plan_ready_for_execution": True,
        },
        "evidence": {
            "grasp_outcome": "_workspace/real_so100/gripper_close_minus120_contact_probe_001/grasp_outcome.json",
        },
    }


def _check_ready_plan(*, plan: dict[str, Any], contact_output_dir: str, vla_prompt_packet: Path) -> list[dict[str, Any]]:
    steps = plan.get("next_steps", [])
    step_types = [step.get("type") for step in steps]
    joined_commands = [" ".join(step.get("command", [])) for step in steps]
    return [
        _check("stage_is_minimal_contact_probe", plan.get("stage") == "minimal_contact_probe", plan.get("stage")),
        _check(
            "step_order",
            step_types
            == [
                "execute_video_backed_contact_probe",
                "materialize_relocation_verifier_packet",
                "run_relocation_verifier",
                "rebuild_agentic_contract",
            ],
            step_types,
        ),
        _check(
            "contact_probe_uses_observer_camera_3",
            "--camera-index 3" in joined_commands[0],
            joined_commands[0] if joined_commands else None,
        ),
        _check(
            "contact_probe_writes_visual_subdir",
            f"--visual-output-dir {contact_output_dir}/visual" in joined_commands[0],
            joined_commands[0] if joined_commands else None,
        ),
        _check(
            "verifier_packet_materializes_from_execution_report",
            f"--vla-prompt-packet {vla_prompt_packet}" in joined_commands[1]
            and f"--execution-report {contact_output_dir}/report.json" in joined_commands[1],
            joined_commands[1] if len(joined_commands) > 1 else None,
        ),
        _check(
            "relocation_verifier_uses_visual_before_after",
            f"--before {contact_output_dir}/visual/before.jpg" in joined_commands[2]
            and f"--after {contact_output_dir}/visual/after.jpg" in joined_commands[2],
            joined_commands[2] if len(joined_commands) > 2 else None,
        ),
        _check(
            "success_evidence_requires_relocation",
            "object_relocation_verifier" in plan.get("required_evidence_before_success_claim", []),
            plan.get("required_evidence_before_success_claim", []),
        ),
    ]


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a non-hardware ready-path fixture for the real SO-100 agentic loop.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("_workspace/real_so100/reports"))
    parser.add_argument("--vla-prompt-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contact-output-dir", default="_workspace/real_so100/contact_probe_ready_path_fixture")
    args = parser.parse_args()
    print(
        json.dumps(
            build_ready_path_fixture(
                output_dir=args.output_dir,
                reports_dir=args.reports_dir,
                vla_prompt_packet=args.vla_prompt_packet,
                output=args.output,
                contact_output_dir=args.contact_output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
