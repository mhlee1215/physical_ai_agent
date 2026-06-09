#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.build_real_so100_agentic_layer_contract import build_agentic_layer_contract
from scripts.build_real_so100_agentic_loop_report import build_agentic_loop_report
from scripts.real_so100_agentic_controller import build_agentic_next_plan
from scripts.real_so100_checkpoint_26_gate import run_checkpoint_26_gate
from scripts.real_so100_reframe_advisor import build_reframe_advice


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = "_workspace/real_so100/calibration/so100_local.json"


def refresh_agentic_loop(
    *,
    contract: Path,
    output_dir: Path,
    reports_dir: Path,
    label: str,
    port: str | None = None,
    episode: Path | None = None,
    frame_index: int | None = None,
    duration_seconds: float = 1.0,
    fps: float = 2.0,
    calibration_file: Path | None = None,
    policy_camera_indexes: list[int] | None = None,
    observer_camera_indexes: list[int] | None = None,
    wrist_camera_index: str = "0",
    egocentric_camera_index: str = "1",
) -> dict[str, Any]:
    base_contract = _load_json(contract)
    task = str(base_contract.get("task_goal", {}).get("instruction") or base_contract.get("policy", {}).get("instruction") or label)
    evidence = base_contract.get("evidence", {})
    policy = base_contract.get("policy", {})
    adapter = base_contract.get("adapter_and_safety", {})
    grasp_outcome = Path(evidence["grasp_outcome"])
    calibration = calibration_file or Path(DEFAULT_CALIBRATION)

    gate = run_checkpoint_26_gate(
        output_dir=output_dir,
        port=port,
        episode=episode,
        frame_index=frame_index,
        grasp_outcome=grasp_outcome,
        calibration_file=calibration,
        duration_seconds=duration_seconds,
        fps=fps,
        task=task,
        policy_camera_indexes=policy_camera_indexes,
        observer_camera_indexes=observer_camera_indexes,
        wrist_camera_index=wrist_camera_index,
        egocentric_camera_index=egocentric_camera_index,
    )
    advice_path = output_dir / "reframe_advice.json"
    advice = build_reframe_advice(
        pregrasp_probe=Path(gate["pregrasp_probe"]),
        jaw_readiness=Path(gate["jaw_readiness"]),
        output=advice_path,
        jaw_camera=wrist_camera_index,
        object_view_camera=egocentric_camera_index,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    refreshed_contract_path = reports_dir / f"real_so100_agentic_layer_contract_{label}.json"
    refreshed_contract_md = reports_dir / f"real_so100_agentic_layer_contract_{label}.md"
    refreshed_contract = build_agentic_layer_contract(
        smolvla_report=Path(policy["report_path"]),
        smolvla_action=Path(policy["action_path"]),
        safety_report=Path(adapter["safety_report_path"]),
        command_plan=Path(adapter["command_plan_path"]),
        next_action_gate=Path(gate["next_action_gate"]),
        grasp_outcome=grasp_outcome,
        pre_stage_pack=Path(evidence["pre_stage_pack"]),
        action_metadata_report=Path(adapter["action_metadata_path"]) if adapter.get("action_metadata_path") else None,
        execute_gate_report=Path(adapter["execute_gate_path"]) if adapter.get("execute_gate_path") else None,
        output=refreshed_contract_path,
        output_markdown=refreshed_contract_md,
    )
    if gate.get("observer_camera_status") == "temporarily_unavailable":
        refreshed_contract.setdefault("policy", {})["observer_camera_indexes"] = []
        refreshed_contract.setdefault("policy", {})["observer_camera_status"] = "temporarily_unavailable"
        refreshed_contract.setdefault("policy", {})["observer_camera_note"] = gate.get("observer_camera_note")
        refreshed_contract_path.write_text(json.dumps(refreshed_contract, indent=2, sort_keys=True), encoding="utf-8")
        if refreshed_contract_md.exists():
            refreshed_contract_md.write_text(_render_observer_off_contract_markdown(refreshed_contract), encoding="utf-8")

    next_plan_path = reports_dir / f"real_so100_agentic_next_plan_{label}.json"
    next_plan = build_agentic_next_plan(
        contract=refreshed_contract_path,
        reframe_advice=advice_path,
        output=next_plan_path,
        port=port or DEFAULT_PORT,
        calibration_file=str(calibration),
        next_output_dir=str(output_dir),
        contact_output_dir=f"_workspace/real_so100/contact_probe_{label}",
    )

    loop_report_path = reports_dir / f"real_so100_agentic_loop_report_{label}.html"
    loop_report = build_agentic_loop_report(
        contract=refreshed_contract_path,
        next_plan=next_plan_path,
        reframe_advice=advice_path,
        output=loop_report_path,
        title=f"Real SO-100 Agentic Loop Report: {label}",
    )
    manifest = {
        "status": "passed",
        "operation": "real_so100_agentic_loop_refresh",
        "label": label,
        "task": task,
        "physical_robot_motion": False,
        "send_action_called": False,
        "observer_camera_status": gate.get("observer_camera_status"),
        "observer_camera_note": gate.get("observer_camera_note"),
        "gate_manifest": gate["manifest_path"],
        "gate_status": gate["status"],
        "reframe_advice": str(advice_path),
        "reframe_actions": advice.get("actions", []),
        "contract": str(refreshed_contract_path),
        "next_plan": str(next_plan_path),
        "loop_report_html": str(loop_report_path),
        "next_stage": next_plan.get("stage"),
        "loop_report_manifest": loop_report["manifest_path"],
        "agentic_decision": refreshed_contract.get("agentic_layer", {}).get("decision"),
    }
    manifest_path = reports_dir / f"real_so100_agentic_loop_refresh_{label}.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_observer_off_contract_markdown(contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Real SO-100 Agentic Layer Contract",
            "",
            f"- Status: `{contract.get('status')}`",
            f"- Decision: `{contract.get('agentic_layer', {}).get('decision')}`",
            f"- Policy cameras: `{contract.get('policy', {}).get('policy_camera_indexes')}`",
            f"- Observer cameras: `{contract.get('policy', {}).get('observer_camera_indexes')}`",
            f"- Observer status: `{contract.get('policy', {}).get('observer_camera_status')}`",
            f"- Send action called: `{contract.get('policy', {}).get('send_action_called')}`",
            "",
            "Observer camera is temporarily unavailable; this contract supports no-actuation agentic-layer development only.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the real SO-100 no-actuation agentic loop in one command.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("_workspace/real_so100/reports"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--port")
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--duration-seconds", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--policy-camera-index", type=int, action="append", default=[])
    parser.add_argument("--observer-camera-index", type=int, action="append", default=[])
    parser.add_argument("--wrist-camera-index", default="0")
    parser.add_argument("--egocentric-camera-index", default="1")
    args = parser.parse_args()
    if args.port is None and args.episode is None:
        raise SystemExit("either --port or --episode is required")
    print(
        json.dumps(
            refresh_agentic_loop(
                contract=args.contract,
                output_dir=args.output_dir,
                reports_dir=args.reports_dir,
                label=args.label,
                port=args.port,
                episode=args.episode,
                frame_index=args.frame_index,
                duration_seconds=args.duration_seconds,
                fps=args.fps,
                calibration_file=args.calibration_file,
                policy_camera_indexes=args.policy_camera_index or None,
                observer_camera_indexes=args.observer_camera_index,
                wrist_camera_index=args.wrist_camera_index,
                egocentric_camera_index=args.egocentric_camera_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
