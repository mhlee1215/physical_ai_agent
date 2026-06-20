#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_micro_step import _capture_visual, _make_so100_bus
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")
DEFAULT_MODEL = Path(
    "_workspace/real_so100/body_map_collection_20260611/session_20260611_184136/body_map/body_map_knn_model.json"
)


def run_body_map_micro_step(
    *,
    model_path: Path,
    output_dir: Path,
    port: str,
    calibration: Path,
    desired_camera: int,
    desired_marker_id: int,
    desired_delta_px: tuple[float, float],
    execute: bool,
    human_confirmed: bool,
    workspace_clear_confirmed: bool,
    max_abs_delta_raw: float,
    scale: float,
    step_settle_seconds: float,
    camera_index: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    calibration_payload = _load_calibration(calibration)
    selected = _select_transition(
        model=model,
        desired_camera=desired_camera,
        desired_marker_id=desired_marker_id,
        desired_delta_px=desired_delta_px,
    )
    proposed_delta = {
        joint: float(value) * scale for joint, value in selected["joint_delta_raw"].items() if joint in SO100_JOINT_ORDER
    }
    limited_delta = {
        joint: _clip(value, -max_abs_delta_raw, max_abs_delta_raw)
        for joint, value in proposed_delta.items()
    }
    blockers = []
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    if not workspace_clear_confirmed:
        blockers.append("Workspace clear confirmation flag is required.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_body_map_micro_step",
        "model": str(model_path),
        "port": port,
        "calibration": str(calibration),
        "desired_camera": desired_camera,
        "desired_marker_id": desired_marker_id,
        "desired_delta_px": list(desired_delta_px),
        "selected_transition_index": selected["transition_index"],
        "selected_transition_marker_delta": selected.get("selected_marker_delta"),
        "selected_transition_joint_delta_raw": selected["joint_delta_raw"],
        "scale": scale,
        "proposed_delta_raw": proposed_delta,
        "limited_delta_raw": limited_delta,
        "max_abs_delta_raw": max_abs_delta_raw,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "camera_index": camera_index,
        "send_action_called": False,
        "policy_actions_executed": False,
        "post_task_torque_disabled": False,
        "home_return_status": None,
        "blockers": blockers,
        "status": "blocked" if blockers else ("ready" if execute else "dry_run"),
    }
    if blockers or not execute:
        _write_json(output_dir / "body_map_micro_step_report.json", report)
        return report

    bus, _motors = _make_so100_bus(port)
    try:
        before_visual = _capture_visual(
            camera_index=camera_index,
            output_dir=output_dir / "visual",
            label="before",
            before_path=None,
        )
        report["visual_check"] = {"before": before_visual}
        before_image = Path(before_visual["image_path"])

        bus.connect(handshake=True)
        before_state = {
            joint: float(value)
            for joint, value in bus.sync_read("Present_Position", normalize=False).items()
            if joint in SO100_JOINT_ORDER
        }
        target = {}
        for joint in SO100_JOINT_ORDER:
            raw = before_state[joint] + limited_delta.get(joint, 0.0)
            target[joint] = int(round(_clip(raw, calibration_payload[joint]["range_min"], calibration_payload[joint]["range_max"])))
        bus.sync_write("Goal_Position", target, normalize=False, num_retry=3)
        report["send_action_called"] = True
        report["policy_actions_executed"] = True
        time.sleep(step_settle_seconds)
        after_state = {
            joint: float(value)
            for joint, value in bus.sync_read("Present_Position", normalize=False).items()
            if joint in SO100_JOINT_ORDER
        }
        report["readback_before_raw"] = before_state
        report["commanded_target_raw"] = target
        report["readback_after_raw"] = after_state
        report["observed_delta_raw"] = {
            joint: round(after_state[joint] - before_state[joint], 4)
            for joint in SO100_JOINT_ORDER
        }
        report["visual_check"]["after"] = _capture_visual(
            camera_index=camera_index,
            output_dir=output_dir / "visual",
            label="after",
            before_path=before_image,
        )
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)

    home_report = move_to_home_pose(
        port=port,
        calibration=calibration,
        home_pose=DEFAULT_HOME_POSE,
        output=output_dir / "home_return_after_body_map_micro_step.json",
        execute=True,
        human_confirmed=human_confirmed,
        workspace_clear_confirmed=workspace_clear_confirmed,
        max_abs_delta_raw=120.0,
        step_settle_seconds=0.10,
        camera_index=camera_index,
        visual_output_dir=output_dir / "home_return_visual",
        record_video=True,
        video_fps=8.0,
    )
    report["home_return_report"] = str(output_dir / "home_return_after_body_map_micro_step.json")
    report["home_return_status"] = home_report.get("status")
    report["post_task_torque_disabled"] = bool(home_report.get("post_task_torque_disabled"))
    _write_json(output_dir / "body_map_micro_step_report.json", report)
    return report


def _select_transition(
    *,
    model: dict[str, Any],
    desired_camera: int,
    desired_marker_id: int,
    desired_delta_px: tuple[float, float],
) -> dict[str, Any]:
    key = f"camera_{desired_camera}_id_{desired_marker_id}"
    candidates = []
    for row in model.get("rows", []):
        marker_delta = (row.get("marker_deltas") or {}).get(key)
        if marker_delta is None:
            continue
        delta = marker_delta["delta_px"]
        distance = math.sqrt((float(delta[0]) - desired_delta_px[0]) ** 2 + (float(delta[1]) - desired_delta_px[1]) ** 2)
        candidates.append((distance, row, marker_delta))
    if not candidates:
        raise ValueError(f"No model transitions for {key}.")
    distance, row, marker_delta = min(candidates, key=lambda item: item[0])
    result = dict(row)
    result["selected_marker_delta"] = marker_delta
    result["selection_distance_px"] = round(distance, 4)
    return result


def _load_calibration(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        joint: {
            "range_min": float(payload[joint]["range_min"]),
            "range_max": float(payload[joint]["range_max"]),
        }
        for joint in SO100_JOINT_ORDER
    }


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one SO-100 body-map approximate IK micro-step.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--desired-camera", type=int, required=True)
    parser.add_argument("--desired-marker-id", type=int, required=True)
    parser.add_argument("--desired-dx-px", type=float, required=True)
    parser.add_argument("--desired-dy-px", type=float, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-abs-delta-raw", type=float, default=35.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.45)
    parser.add_argument("--camera-index", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_body_map_micro_step(
                model_path=args.model,
                output_dir=args.output_dir,
                port=args.port,
                calibration=args.calibration,
                desired_camera=args.desired_camera,
                desired_marker_id=args.desired_marker_id,
                desired_delta_px=(args.desired_dx_px, args.desired_dy_px),
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
                max_abs_delta_raw=args.max_abs_delta_raw,
                scale=args.scale,
                step_settle_seconds=args.step_settle_seconds,
                camera_index=args.camera_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
