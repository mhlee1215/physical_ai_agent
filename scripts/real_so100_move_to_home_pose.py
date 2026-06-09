#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_move_to_natural_pose import move_to_natural_pose


DEFAULT_HOME_POSE = Path("_workspace/real_so100/home_pose/canonical_home_pose_2026_06_07.json")


def move_to_home_pose(
    *,
    port: str,
    calibration: Path,
    home_pose: Path,
    output: Path,
    execute: bool,
    human_confirmed: bool,
    workspace_clear_confirmed: bool,
    max_abs_delta_raw: float,
    step_settle_seconds: float,
    camera_index: int | None,
    visual_output_dir: Path | None,
    record_video: bool,
    video_fps: float,
) -> dict[str, Any]:
    home_payload = _load_home_pose(home_pose)
    report = move_to_natural_pose(
        port=port,
        calibration=calibration,
        output=output,
        execute=execute,
        human_confirmed=human_confirmed,
        workspace_clear_confirmed=workspace_clear_confirmed,
        max_abs_delta_raw=max_abs_delta_raw,
        step_settle_seconds=step_settle_seconds,
        camera_index=camera_index,
        visual_output_dir=visual_output_dir,
        record_video=record_video,
        video_fps=video_fps,
        target_overrides=home_payload["target_raw"],
    )
    report["operation"] = "real_so100_move_to_home_pose"
    report["home_pose"] = str(home_pose)
    report["home_pose_name"] = home_payload.get("name")
    report["target_home_raw"] = home_payload["target_raw"]
    report["target_natural_raw"] = None
    _write_json(output, report)
    return report


def _load_home_pose(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_raw = payload.get("target_raw")
    if not isinstance(target_raw, dict):
        raise ValueError(f"home pose missing target_raw: {path}")
    missing = [joint for joint in SO100_JOINT_ORDER if joint not in target_raw]
    if missing:
        raise ValueError(f"home pose target_raw missing joints: {missing}")
    payload["target_raw"] = {joint: float(target_raw[joint]) for joint in SO100_JOINT_ORDER}
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move the real SO-100 follower to the user-defined home pose.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--home-pose", type=Path, default=DEFAULT_HOME_POSE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    parser.add_argument("--max-abs-delta-raw", type=float, default=80.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.12)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    args = parser.parse_args()

    print(
        json.dumps(
            move_to_home_pose(
                port=args.port,
                calibration=args.calibration,
                home_pose=args.home_pose,
                output=args.output,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
                max_abs_delta_raw=args.max_abs_delta_raw,
                step_settle_seconds=args.step_settle_seconds,
                camera_index=args.camera_index,
                visual_output_dir=args.visual_output_dir,
                record_video=args.record_video,
                video_fps=args.video_fps,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
