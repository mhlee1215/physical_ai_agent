#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_real_so100_agentic_log import analyze_agentic_log
from scripts.build_real_so100_agentic_policy_patch import build_agentic_policy_patch
from scripts.build_real_so100_prompt_iteration import build_prompt_iteration
from scripts.real_so100_agentic_controller import DEFAULT_CALIBRATION, DEFAULT_PORT, build_agentic_next_plan
from scripts.real_so100_agentic_loop_refresh import refresh_agentic_loop
from scripts.update_real_so100_agentic_state import update_agentic_state


def run_agentic_iteration(
    *,
    prompt: str,
    contract: Path,
    output_dir: Path,
    reports_dir: Path,
    label: str,
    state: Path,
    iteration_index: int | None = None,
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
    vla_prompt_packet: Path | None = None,
) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    calibration = calibration_file or Path(DEFAULT_CALIBRATION)
    refresh = refresh_agentic_loop(
        contract=contract,
        output_dir=output_dir,
        reports_dir=reports_dir,
        label=label,
        port=port,
        episode=episode,
        frame_index=frame_index,
        duration_seconds=duration_seconds,
        fps=fps,
        calibration_file=calibration,
        policy_camera_indexes=policy_camera_indexes,
        observer_camera_indexes=observer_camera_indexes,
        wrist_camera_index=wrist_camera_index,
        egocentric_camera_index=egocentric_camera_index,
    )

    analysis_path = reports_dir / f"real_so100_agentic_log_analysis_{label}.json"
    analysis = analyze_agentic_log(
        refresh_manifest=Path(refresh["manifest_path"]),
        output=analysis_path,
    )

    updated_state = update_agentic_state(
        analysis=analysis_path,
        state=state,
        output=state,
    )

    stateful_next_plan_path = reports_dir / f"real_so100_agentic_next_plan_{label}_stateful.json"
    stateful_next_plan = build_agentic_next_plan(
        contract=Path(refresh["contract"]),
        reframe_advice=Path(refresh["reframe_advice"]),
        agentic_state=state,
        output=stateful_next_plan_path,
        port=port or DEFAULT_PORT,
        calibration_file=str(calibration),
        next_output_dir=str(output_dir),
        contact_output_dir=f"_workspace/real_so100/contact_probe_{label}",
        vla_prompt_packet=vla_prompt_packet,
    )

    refreshed_contract = _load_json(Path(refresh["contract"]))
    smolvla_report = Path(refreshed_contract["policy"]["report_path"])
    prompt_iteration_json = reports_dir / f"real_so100_prompt_iteration_{label}.json"
    prompt_iteration_md = reports_dir / f"real_so100_prompt_iteration_{label}.md"
    policy_patch_path = reports_dir / f"real_so100_agentic_policy_patch_{label}.json"
    prompt_iteration = build_prompt_iteration(
        prompt=prompt,
        smolvla_report=smolvla_report,
        refresh_manifest=Path(refresh["manifest_path"]),
        analysis=analysis_path,
        agentic_state=state,
        next_plan=stateful_next_plan_path,
        output_json=prompt_iteration_json,
        output_md=prompt_iteration_md,
        iteration_index=iteration_index,
        vla_prompt_packet=vla_prompt_packet,
        agentic_policy_patch=policy_patch_path,
    )
    policy_patch = build_agentic_policy_patch(
        analysis=analysis_path,
        agentic_state=state,
        prompt_iteration=prompt_iteration_json,
        output=policy_patch_path,
    )

    manifest = {
        "status": "passed",
        "operation": "real_so100_agentic_iteration",
        "label": label,
        "prompt": prompt,
        "iteration_index": iteration_index,
        "physical_robot_motion": False,
        "send_action_called": False,
        "observer_camera_status": refresh.get("observer_camera_status"),
        "observer_camera_note": refresh.get("observer_camera_note"),
        "camera_contract": prompt_iteration.get("camera_contract", {}),
        "refresh_manifest": refresh["manifest_path"],
        "gate_status": refresh.get("gate_status"),
        "analysis": str(analysis_path),
        "failure_modes": [item.get("type") for item in analysis.get("failure_modes", [])],
        "agentic_state": str(state),
        "active_constraints": updated_state.get("active_constraints", []),
        "stateful_next_plan": str(stateful_next_plan_path),
        "vla_prompt_packet": str(vla_prompt_packet) if vla_prompt_packet else None,
        "next_stage": stateful_next_plan.get("stage"),
        "next_step_type": prompt_iteration.get("next_iteration", {}).get("next_step_type"),
        "prompt_iteration": str(prompt_iteration_json),
        "prompt_iteration_markdown": str(prompt_iteration_md),
        "agentic_policy_patch": str(policy_patch_path),
        "policy_patch_rules": [item.get("id") for item in policy_patch.get("rules", [])],
        "task_success_claim_allowed": prompt_iteration.get("success_accounting", {}).get(
            "task_success_claim_allowed"
        ),
    }
    manifest_path = reports_dir / f"real_so100_agentic_iteration_{label}.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real SO-100 SmolVLA agentic iteration.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("_workspace/real_so100/reports"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--state", type=Path, default=Path("_workspace/real_so100/reports/real_so100_agentic_state.json"))
    parser.add_argument("--iteration-index", type=int)
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
    parser.add_argument("--vla-prompt-packet", type=Path)
    args = parser.parse_args()
    if args.port is None and args.episode is None:
        raise SystemExit("either --port or --episode is required")
    print(
        json.dumps(
            run_agentic_iteration(
                prompt=args.prompt,
                contract=args.contract,
                output_dir=args.output_dir,
                reports_dir=args.reports_dir,
                label=args.label,
                state=args.state,
                iteration_index=args.iteration_index,
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
                vla_prompt_packet=args.vla_prompt_packet,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
