#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = "_workspace/real_so100/calibration/so100_local.json"


def build_next_runbook(
    *,
    pre_stage_pack: Path,
    output: Path,
    port: str = DEFAULT_PORT,
    calibration_file: str = DEFAULT_CALIBRATION,
) -> dict[str, Any]:
    pack = _load_json(pre_stage_pack)
    gate = _load_json(Path(pack["next_action_gate"]))
    movement_manifest = _load_json(Path(pack["movement_report_manifest"]))
    lines = _render_runbook(
        pack=pack,
        gate=gate,
        movement_manifest=movement_manifest,
        pre_stage_pack=pre_stage_pack,
        port=port,
        calibration_file=calibration_file,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "passed",
        "operation": "real_so100_next_runbook",
        "output_markdown": str(output),
        "pre_stage_pack": str(pre_stage_pack),
        "current_gate_status": pack.get("current_gate_status"),
        "recommended_action": pack.get("recommended_action"),
        "allowed_physical_action": pack.get("allowed_physical_action"),
        "movement_report_html": pack.get("movement_report_html"),
        "gate_report_html": pack.get("gate_report_html"),
        "video_count": movement_manifest.get("video_count", 0),
        "contains_physical_command": gate.get("status") == "ready",
        "purpose": "operator handoff for the SO-100 agentic-layer pre-stage",
    }
    manifest_path = output.with_suffix(".json")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _render_runbook(
    *,
    pack: dict[str, Any],
    gate: dict[str, Any],
    movement_manifest: dict[str, Any],
    pre_stage_pack: Path,
    port: str,
    calibration_file: str,
) -> list[str]:
    status = str(pack.get("current_gate_status"))
    recommended_action = str(pack.get("recommended_action"))
    lines = [
        "# Real SO-100 Next Runbook",
        "",
        "This is an operator handoff for the agentic-layer pre-stage. It is not a benchmark success report.",
        "",
        "## Current Evidence",
        "",
        f"- Pre-stage pack: `{pre_stage_pack}`",
        f"- Movement report: `{pack.get('movement_report_html')}`",
        f"- Gate report: `{pack.get('gate_report_html')}`",
        f"- Movement videos in report: `{movement_manifest.get('video_count', 0)}`",
        f"- Current gate status: `{status}`",
        f"- Recommended action: `{recommended_action}`",
    ]
    blockers = gate.get("blockers", [])
    if blockers:
        lines.append(f"- Blockers: `{'; '.join(str(item) for item in blockers)}`")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
        ]
    )
    if status == "ready" and gate.get("allowed_physical_action"):
        lines.extend(_ready_lines(port=port, calibration_file=calibration_file, pack=pack))
    else:
        lines.extend(_blocked_lines(port=port, calibration_file=calibration_file, pack=pack))
    lines.extend(
        [
            "",
            "## Refresh Reports After Any Movement",
            "",
            "After an executed movement, rebuild the human-review report and pre-stage pack so the new `motion.mp4` is preserved:",
            "",
            "```bash",
            "PYTHONPATH=src:. .venv/bin/python -B scripts/build_real_so100_movement_report.py \\",
            "  --report _workspace/real_so100/gripper_close_minus50_contact_probe_001/report.json \\",
            "  --report _workspace/real_so100/gripper_close_minus80_contact_probe_001/report.json \\",
            "  --report _workspace/real_so100/gripper_close_minus120_contact_probe_001/report.json \\",
            "  --report _workspace/real_so100/reframe_shoulder_pan_plus15_camera0_gate_001/report.json \\",
            "  --report _workspace/real_so100/reframe_shoulder_pan_minus30_camera0_gate_001/report.json \\",
            "  --output _workspace/real_so100/reports/real_so100_reframe_movement_report_next.html",
            "```",
            "",
            "```bash",
            "PYTHONPATH=src:. .venv/bin/python -B scripts/build_real_so100_agentic_prestage_pack.py \\",
            "  --movement-report-manifest _workspace/real_so100/reports/real_so100_reframe_movement_report_next.json \\",
            "  --next-action-gate _workspace/real_so100/<latest_gate_dir>/next_action_gate.json \\",
            "  --grasp-outcome _workspace/real_so100/gripper_close_minus120_contact_probe_001/grasp_outcome.json \\",
            "  --output _workspace/real_so100/reports/real_so100_agentic_prestage_pack_next.json",
            "```",
        ]
    )
    return lines


def _blocked_lines(*, port: str, calibration_file: str, pack: dict[str, Any]) -> list[str]:
    return [
        "The physical action gate is blocked, so do not close the gripper yet.",
        "",
        "1. Reframe camera 0/1 or the target object so the green object and gripper are usable in the two Innomaker policy views.",
        "2. If `scripts/real_so100_reframe_advisor.py` reports left/top clipping in camera `0`, move the target appearance rightward/downward in that image while keeping the jaw marker visible.",
        "3. Keep SmolVLA policy inputs restricted to Innomaker camera indexes `0` and `1`; keep iPhone camera index `3` as Codex observer/debug only.",
        "4. Run the no-actuation gate again:",
        "",
        "```bash",
        "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_checkpoint_26_gate.py \\",
        "  --output-dir _workspace/real_so100/checkpoint_26_gate_after_manual_reframe_next \\",
        f"  --port {port} \\",
        f"  --calibration-file {calibration_file} \\",
        "  --duration-seconds 1.0 \\",
        "  --fps 2.0 \\",
        "  --policy-camera-index 0 \\",
        "  --policy-camera-index 1 \\",
        "  --observer-camera-index 3 \\",
        "  --wrist-camera-index 0 \\",
        "  --egocentric-camera-index 1 \\",
        "  --task green_doll_after_manual_reframe \\",
        f"  --grasp-outcome {pack.get('grasp_outcome')}",
        "```",
        "",
        "Only continue to a contact probe if that gate returns `status=ready` and `allowed_physical_action` is non-null.",
    ]


def _ready_lines(*, port: str, calibration_file: str, pack: dict[str, Any]) -> list[str]:
    action = pack.get("allowed_physical_action", {})
    joint = action.get("joint", "gripper")
    return [
        "The physical action gate is ready. Use a minimal, video-backed contact probe.",
        "",
        "```bash",
        "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_micro_step.py \\",
        f"  --port {port} \\",
        f"  --joint {joint} \\",
        "  --manual-delta-raw -30 \\",
        "  --output _workspace/real_so100/gripper_contact_probe_next/report.json \\",
        "  --execute \\",
        "  --human-confirmed \\",
        "  --contact-ok-for-gripper \\",
        "  --max-abs-delta-raw 30 \\",
        "  --settle-seconds 1.5 \\",
        "  --camera-index 3 \\",
        "  --visual-output-dir _workspace/real_so100/gripper_contact_probe_next/visual \\",
        "  --record-video \\",
        "  --video-fps 12",
        "```",
        "",
        "Then materialize the observer-frame relocation verifier packet and run the relocation verifier before any further close command.",
        "",
        "```bash",
        "PYTHONPATH=src:. .venv/bin/python -B scripts/build_real_so100_relocation_verifier_packet.py \\",
        "  --vla-prompt-packet _workspace/real_so100/reports/real_so100_vla_prompt_packet_move_right_u20cam_003.json \\",
        "  --execution-report _workspace/real_so100/gripper_contact_probe_next/report.json \\",
        "  --output _workspace/real_so100/gripper_contact_probe_next/relocation_verifier_packet_right.json \\",
        "  --relocation-output _workspace/real_so100/gripper_contact_probe_next/object_relocation_right.json",
        "```",
        "",
        "```bash",
        "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_object_relocation.py \\",
        "  --before _workspace/real_so100/gripper_contact_probe_next/visual/before.jpg \\",
        "  --after _workspace/real_so100/gripper_contact_probe_next/visual/after.jpg \\",
        "  --target-direction right \\",
        "  --min-delta-px 40 \\",
        "  --output _workspace/real_so100/gripper_contact_probe_next/object_relocation_right.json",
        "```",
        "",
        "Before asking SmolVLA for another proposal, refresh the no-actuation policy observation with the two Innomaker inputs:",
        "",
        "```bash",
        "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_checkpoint_26_gate.py \\",
        "  --output-dir _workspace/real_so100/checkpoint_26_gate_after_contact_probe_next \\",
        f"  --port {port} \\",
        f"  --calibration-file {calibration_file} \\",
        "  --duration-seconds 1.0 \\",
        "  --fps 2.0 \\",
        "  --policy-camera-index 0 \\",
        "  --policy-camera-index 1 \\",
        "  --observer-camera-index 3 \\",
        "  --wrist-camera-index 0 \\",
        "  --egocentric-camera-index 1 \\",
        "  --task green_doll_after_contact_probe \\",
        f"  --grasp-outcome {pack.get('grasp_outcome')}",
        "```",
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the next real SO-100 operator runbook.")
    parser.add_argument("--pre-stage-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration-file", default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    print(
        json.dumps(
            build_next_runbook(
                pre_stage_pack=args.pre_stage_pack,
                output=args.output,
                port=args.port,
                calibration_file=args.calibration_file,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
