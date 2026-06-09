#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def decide_next_action(
    *,
    pregrasp_probe: Path,
    jaw_readiness: Path,
    grasp_outcome: Path | None = None,
    output: Path | None = None,
    object_view_camera: str = "1",
    jaw_camera: str = "0",
) -> dict[str, Any]:
    pregrasp = _load_json(pregrasp_probe)
    jaw = _load_json(jaw_readiness)
    grasp = _load_json(grasp_outcome) if grasp_outcome is not None else None

    blockers: list[str] = []
    evidence: dict[str, Any] = {
        "pregrasp_probe": str(pregrasp_probe),
        "jaw_readiness": str(jaw_readiness),
        "grasp_outcome": str(grasp_outcome) if grasp_outcome else None,
        "pregrasp_status": pregrasp.get("status"),
        "primary_camera": pregrasp.get("primary_camera"),
        "usable_cameras": pregrasp.get("usable_cameras", []),
        "object_view_camera": object_view_camera,
        "jaw_camera": jaw_camera,
        "jaw_status": jaw.get("status"),
        "jaw_blockers": jaw.get("blockers", []),
        "last_grasp_outcome": grasp.get("grasp_outcome") if grasp else None,
    }

    if pregrasp.get("status") != "passed" or object_view_camera not in [str(item) for item in pregrasp.get("usable_cameras", [])]:
        blockers.append(f"camera {object_view_camera} object-view pregrasp probe is not passed")
    if jaw.get("status") != "ready":
        blockers.append(f"camera {jaw_camera} jaw/object framing gate is not ready")
    if grasp and grasp.get("grasp_outcome") == "grasp_failed_object_stationary":
        evidence["last_failure_requires_reframe"] = True
    else:
        evidence["last_failure_requires_reframe"] = False

    if blockers:
        recommended_action = f"reframe_camera_{jaw_camera}_or_camera_{object_view_camera}_or_object"
        allowed_physical_action = None
        status = "blocked"
    elif evidence["last_failure_requires_reframe"]:
        recommended_action = "contact_probe_allowed_after_reframe"
        allowed_physical_action = {
            "joint": "gripper",
            "contact_ok_for_gripper": True,
            "requires_object_view_before_after": True,
            "object_view_camera": object_view_camera,
            "requires_grasp_outcome_verifier": True,
        }
        status = "ready"
    else:
        recommended_action = "contact_probe_allowed"
        allowed_physical_action = {
            "joint": "gripper",
            "contact_ok_for_gripper": True,
            "requires_object_view_before_after": True,
            "object_view_camera": object_view_camera,
            "requires_grasp_outcome_verifier": True,
        }
        status = "ready"

    vla_prompt_gate = _vla_prompt_gate(
        pregrasp=pregrasp,
        jaw=jaw,
        object_view_camera=object_view_camera,
        jaw_camera=jaw_camera,
    )
    result = {
        "status": status,
        "recommended_action": recommended_action,
        "vla_prompt_allowed": vla_prompt_gate["status"] == "ready",
        "vla_prompt_gate": vla_prompt_gate,
        "allowed_physical_action": allowed_physical_action,
        "physical_execution_gate": {
            "status": status,
            "allowed_physical_action": allowed_physical_action,
            "blockers": blockers,
        },
        "blockers": blockers,
        "evidence": evidence,
        "notes": [
            "Next-action gate only; it does not execute robot actions.",
            "SmolVLA prompt/proposal readiness is separate from physical execution readiness.",
            f"A physical close probe still requires explicit micro-step confirmations and camera-{object_view_camera} before/after evidence.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _vla_prompt_gate(
    *,
    pregrasp: dict[str, Any],
    jaw: dict[str, Any],
    object_view_camera: str,
    jaw_camera: str,
) -> dict[str, Any]:
    object_view_usable = (
        pregrasp.get("status") == "passed"
        and object_view_camera in [str(item) for item in pregrasp.get("usable_cameras", [])]
    )
    jaw_marker_visible = bool(jaw.get("jaw_marker_candidate"))
    blockers = []
    if not object_view_usable:
        blockers.append(f"camera {object_view_camera} does not provide usable object-view policy input")
    if not jaw_marker_visible:
        blockers.append(f"camera {jaw_camera} does not provide a visible jaw marker")
    if blockers:
        return {
            "status": "blocked",
            "proposal_only": True,
            "blockers": blockers,
            "reason": "SmolVLA proposal input is incomplete",
        }
    return {
        "status": "ready",
        "proposal_only": True,
        "blockers": [],
        "reason": (
            f"camera {object_view_camera} has usable object context and camera {jaw_camera} has jaw-marker evidence; "
            "physical execution can still remain blocked by stricter jaw/object framing"
        ),
    }


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide the next safe real SO-100 agentic action.")
    parser.add_argument("--pregrasp-probe", type=Path, required=True)
    parser.add_argument("--jaw-readiness", type=Path, required=True)
    parser.add_argument("--grasp-outcome", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--object-view-camera", default="1")
    parser.add_argument("--jaw-camera", default="0")
    args = parser.parse_args()
    print(
        json.dumps(
            decide_next_action(
                pregrasp_probe=args.pregrasp_probe,
                jaw_readiness=args.jaw_readiness,
                grasp_outcome=args.grasp_outcome,
                output=args.output,
                object_view_camera=args.object_view_camera,
                jaw_camera=args.jaw_camera,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
