#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_agentic_prestage_pack(
    *,
    output: Path,
    movement_report_manifest: Path,
    next_action_gate: Path,
    grasp_outcome: Path,
    gate_report_manifest: Path | None = None,
    title: str = "Real SO-100 Agentic Pre-stage Evidence Pack",
) -> dict[str, Any]:
    movement = _load_json(movement_report_manifest)
    gate = _load_json(next_action_gate)
    grasp = _load_json(grasp_outcome)
    gate_report = _load_json(gate_report_manifest) if gate_report_manifest is not None else {}
    lessons = _derive_lessons(movement=movement, gate=gate, grasp=grasp)
    pack = {
        "status": "passed",
        "operation": "real_so100_agentic_prestage_pack",
        "title": title,
        "purpose": "pre-stage evidence for improving the agentic layer, not benchmark success",
        "movement_report_manifest": str(movement_report_manifest),
        "movement_report_html": movement.get("output_html"),
        "gate_report_manifest": str(gate_report_manifest) if gate_report_manifest else None,
        "gate_report_html": gate_report.get("output_html") if gate_report else None,
        "next_action_gate": str(next_action_gate),
        "grasp_outcome": str(grasp_outcome),
        "current_gate_status": gate.get("status"),
        "recommended_action": gate.get("recommended_action"),
        "allowed_physical_action": gate.get("allowed_physical_action"),
        "last_grasp_outcome": grasp.get("grasp_outcome"),
        "video_count": movement.get("video_count", 0),
        "legacy_without_video_count": movement.get("legacy_without_video_count", 0),
        "agentic_lessons": lessons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")
    return pack


def _derive_lessons(*, movement: dict[str, Any], gate: dict[str, Any], grasp: dict[str, Any]) -> list[dict[str, str]]:
    lessons: list[dict[str, str]] = []
    if grasp.get("grasp_outcome") == "grasp_failed_object_stationary":
        lessons.append(
            {
                "observation": "The gripper moved, but the green object bbox and center stayed stable.",
                "agentic_update": "Treat this as a failed contact/grasp attempt and do not keep closing from the same pose.",
            }
        )
    if gate.get("recommended_action") == "reframe_camera_0_or_object":
        lessons.append(
            {
                "observation": "The next-action gate blocks physical action because camera 0 object/jaw framing is not ready.",
                "agentic_update": "Acquire a better camera-0/end-effector view before issuing another contact probe.",
            }
        )
    if int(movement.get("legacy_without_video_count", 0)) > 0:
        lessons.append(
            {
                "observation": "Some movement evidence predates mandatory motion video capture.",
                "agentic_update": "For all future real movements, require --record-video and include motion.mp4 in the human-review report.",
            }
        )
    if not lessons:
        lessons.append(
            {
                "observation": "No blocking failure pattern was detected in the supplied evidence.",
                "agentic_update": "Proceed only through the next-action gate and preserve video-backed movement reports.",
            }
        )
    return lessons


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real SO-100 agentic pre-stage evidence pack.")
    parser.add_argument("--movement-report-manifest", type=Path, required=True)
    parser.add_argument("--gate-report-manifest", type=Path)
    parser.add_argument("--next-action-gate", type=Path, required=True)
    parser.add_argument("--grasp-outcome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Real SO-100 Agentic Pre-stage Evidence Pack")
    args = parser.parse_args()
    print(
        json.dumps(
            build_agentic_prestage_pack(
                output=args.output,
                movement_report_manifest=args.movement_report_manifest,
                gate_report_manifest=args.gate_report_manifest,
                next_action_gate=args.next_action_gate,
                grasp_outcome=args.grasp_outcome,
                title=args.title,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
