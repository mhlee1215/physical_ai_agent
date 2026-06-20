#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_apriltag_detection_check import check_apriltag_detection
from scripts.real_so100_micro_step import _make_so100_bus, _probe_motion_video, _record_motion_video, _start_motion_video
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")


def collect_keypoint_marker_trajectory(
    *,
    output_root: Path,
    port: str,
    calibration: Path,
    camera_indexes: list[int],
    observer_camera_index: int,
    execute: bool,
    human_confirmed: bool,
    workspace_clear_confirmed: bool,
    max_abs_delta_raw: float,
    step_settle_seconds: float,
    waypoint_settle_seconds: float,
    video_fps: float,
) -> dict[str, Any]:
    import cv2

    run_dir = output_root / time.strftime("trajectory_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "trajectory_report.json"
    calibration_payload = _load_calibration(calibration)
    blockers = []
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    if not workspace_clear_confirmed:
        blockers.append("Workspace clear confirmation flag is required.")
    if max_abs_delta_raw <= 0:
        blockers.append("max_abs_delta_raw must be positive.")
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_collect_keypoint_marker_trajectory",
        "run_dir": str(run_dir),
        "port": port,
        "calibration": str(calibration),
        "camera_indexes": camera_indexes,
        "observer_camera_index": observer_camera_index,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "max_abs_delta_raw": max_abs_delta_raw,
        "step_settle_seconds": step_settle_seconds,
        "waypoint_settle_seconds": waypoint_settle_seconds,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "post_task_torque_disabled": False,
        "blockers": blockers,
        "status": "blocked" if blockers else ("ready" if execute else "dry_run"),
        "waypoints": [],
    }
    if blockers or not execute:
        _write_json(report_path, report)
        return report

    bus, _motors = _make_so100_bus(port)
    video_capture = None
    video_writer = None
    try:
        bus.connect(handshake=True)
        current = _read_positions(bus)
        report["readback_before_raw"] = current
        bus.sync_write("Goal_Position", {joint: int(round(current[joint])) for joint in SO100_JOINT_ORDER}, normalize=False, num_retry=3)
        report["initial_hold_sent"] = True

        video_capture, video_writer, video_result = _start_motion_video(
            camera_index=observer_camera_index,
            output_dir=run_dir / "observer_video",
            fps=video_fps,
        )
        report["motion_video"] = video_result

        initial_packet = _capture_waypoint_packet(
            name="waypoint_000_start",
            role="start_pose_before_coupled_trajectory",
            run_dir=run_dir,
            camera_indexes=camera_indexes,
            state=_read_positions(bus),
        )
        report["waypoints"].append(initial_packet)

        targets = _default_coupled_targets(current=current, calibration=calibration_payload)
        report["planned_targets"] = targets
        executed_steps: list[dict[str, Any]] = []
        state = dict(current)
        for target_index, target_item in enumerate(targets, start=1):
            target = target_item["target_raw"]
            steps = _interpolate_targets(
                current=state,
                target=target,
                calibration=calibration_payload,
                max_abs_delta_raw=max_abs_delta_raw,
            )
            for step_index, step_target in enumerate(steps):
                bus.sync_write("Goal_Position", step_target, normalize=False, num_retry=3)
                report["send_action_called"] = True
                report["physical_robot_motion"] = True
                executed_steps.append(
                    {
                        "target_index": target_index,
                        "target_name": target_item["name"],
                        "step_index_in_target": step_index,
                        "target_raw": step_target,
                    }
                )
                _record_motion_video(
                    capture=video_capture,
                    writer=video_writer,
                    result=video_result,
                    duration_seconds=step_settle_seconds,
                    fps=video_fps,
                )
            time.sleep(waypoint_settle_seconds)
            state = _read_positions(bus)
            packet = _capture_waypoint_packet(
                name=f"waypoint_{target_index:03d}_{target_item['name']}",
                role=target_item["role"],
                run_dir=run_dir,
                camera_indexes=camera_indexes,
                state=state,
            )
            report["waypoints"].append(packet)

        after = _read_positions(bus)
        report["executed_steps"] = executed_steps
        report["executed_action_steps"] = len(executed_steps)
        report["readback_after_raw"] = after
        report["observed_delta_raw"] = {
            joint: round(after[joint] - current[joint], 4)
            for joint in SO100_JOINT_ORDER
        }
        report["policy_actions_executed"] = True
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        if video_writer is not None:
            video_writer.release()
        if video_capture is not None:
            video_capture.release()
        if isinstance(report.get("motion_video"), dict):
            report["motion_video"].update(_probe_motion_video(Path(report["motion_video"]["path"])))
            report["motion_video"]["preview_frames"] = _write_video_preview_frames(
                cv2=cv2,
                video_path=Path(report["motion_video"]["path"]),
                output_dir=Path(report["motion_video"]["path"]).parent / "preview_frames",
            )
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)

    home_report = move_to_home_pose(
        port=port,
        calibration=calibration,
        home_pose=DEFAULT_HOME_POSE,
        output=run_dir / "home_return_after_keypoint_trajectory.json",
        execute=True,
        human_confirmed=human_confirmed,
        workspace_clear_confirmed=workspace_clear_confirmed,
        max_abs_delta_raw=120.0,
        step_settle_seconds=0.10,
        camera_index=observer_camera_index,
        visual_output_dir=run_dir / "home_return_visual",
        record_video=True,
        video_fps=video_fps,
    )
    report["home_return_report"] = str(run_dir / "home_return_after_keypoint_trajectory.json")
    report["home_return_status"] = home_report.get("status")
    report["post_task_torque_disabled"] = bool(home_report.get("post_task_torque_disabled"))
    _write_json(report_path, report)
    return report


def _default_coupled_targets(*, current: dict[str, float], calibration: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    raw_targets = [
        (
            "lifted_center",
            "safe lifted center pose between table anchors",
            {
                "shoulder_pan": 2055,
                "shoulder_lift": 2450,
                "elbow_flex": 1750,
                "wrist_flex": 1750,
                "wrist_roll": current["wrist_roll"],
                "gripper": current["gripper"],
            },
        ),
        (
            "workspace_left",
            "coupled left workspace visit near table keypoint markers",
            {
                "shoulder_pan": 1860,
                "shoulder_lift": 2520,
                "elbow_flex": 1640,
                "wrist_flex": 1620,
                "wrist_roll": current["wrist_roll"],
                "gripper": current["gripper"],
            },
        ),
        (
            "workspace_right",
            "coupled right workspace visit near table keypoint markers",
            {
                "shoulder_pan": 2310,
                "shoulder_lift": 2520,
                "elbow_flex": 1640,
                "wrist_flex": 1620,
                "wrist_roll": current["wrist_roll"],
                "gripper": current["gripper"],
            },
        ),
        (
            "forward_center",
            "moderate forward center pose for workspace coverage",
            {
                "shoulder_pan": 2055,
                "shoulder_lift": 2750,
                "elbow_flex": 1380,
                "wrist_flex": 1400,
                "wrist_roll": current["wrist_roll"],
                "gripper": current["gripper"],
            },
        ),
        (
            "low_forward_center",
            "lower pregrasp-height center pose near table keypoint markers",
            {
                "shoulder_pan": 2055,
                "shoulder_lift": 3100,
                "elbow_flex": 1100,
                "wrist_flex": 1120,
                "wrist_roll": current["wrist_roll"],
                "gripper": current["gripper"],
            },
        ),
    ]
    return [
        {
            "name": name,
            "role": role,
            "target_raw": {
                joint: int(round(_clip(float(target[joint]), calibration[joint]["range_min"], calibration[joint]["range_max"])))
                for joint in SO100_JOINT_ORDER
            },
        }
        for name, role, target in raw_targets
    ]


def _capture_waypoint_packet(
    *,
    name: str,
    role: str,
    run_dir: Path,
    camera_indexes: list[int],
    state: dict[str, float],
) -> dict[str, Any]:
    waypoint_dir = run_dir / name
    detection = check_apriltag_detection(
        camera_indexes=camera_indexes,
        output_dir=waypoint_dir,
        dictionaries=["tag36h11"],
        warmup_frames=6,
    )
    packet = {
        "name": name,
        "role": role,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "passed" if detection.get("ok") else "blocked",
        "joint_positions_raw": state,
        "camera_detection_report": str(waypoint_dir / "detection_report.json"),
        "raw_images": {
            camera: report.get("raw_image")
            for camera, report in detection.get("cameras", {}).items()
        },
        "marker_observations": _marker_observations(detection),
        "send_action_called": False,
        "writes_intended": False,
    }
    packet_path = waypoint_dir / "waypoint_packet.json"
    packet["waypoint_packet"] = str(packet_path)
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    return packet


def _marker_observations(detection: dict[str, Any]) -> list[dict[str, Any]]:
    observations = []
    for camera, camera_report in detection.get("cameras", {}).items():
        for detection_item in camera_report.get("detections", []):
            for marker in detection_item.get("markers", []):
                min_side = float(marker.get("min_side_px", 0.0))
                observations.append(
                    {
                        "camera": int(camera),
                        "marker_id": int(marker["id"]),
                        "center_px": marker["center_px"],
                        "min_side_px": marker["min_side_px"],
                        "quality": "strong" if min_side >= 45 else "usable" if min_side >= 24 else "weak",
                    }
                )
    return observations


def _read_positions(bus: Any) -> dict[str, float]:
    values = bus.sync_read("Present_Position", normalize=False)
    return {
        joint: float(values[joint])
        for joint in SO100_JOINT_ORDER
    }


def _load_calibration(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        joint: {
            "range_min": float(payload[joint]["range_min"]),
            "range_max": float(payload[joint]["range_max"]),
        }
        for joint in SO100_JOINT_ORDER
    }


def _interpolate_targets(
    *,
    current: dict[str, float],
    target: dict[str, int],
    calibration: dict[str, dict[str, float]],
    max_abs_delta_raw: float,
) -> list[dict[str, int]]:
    largest = max(abs(float(target[joint]) - float(current[joint])) for joint in SO100_JOINT_ORDER)
    steps = max(1, int(math.ceil(largest / max_abs_delta_raw)))
    result = []
    for index in range(1, steps + 1):
        fraction = index / steps
        result.append(
            {
                joint: int(
                    round(
                        _clip(
                            float(current[joint]) + ((float(target[joint]) - float(current[joint])) * fraction),
                            calibration[joint]["range_min"],
                            calibration[joint]["range_max"],
                        )
                    )
                )
                for joint in SO100_JOINT_ORDER
            }
        )
    return result


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _write_video_preview_frames(*, cv2: Any, video_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return {"status": "blocked", "reason": "video has no readable frame count"}
        frames: dict[str, str] = {}
        stats: dict[str, dict[str, float | int]] = {}
        for label, index in {
            "first": 0,
            "middle": max(0, frame_count // 2),
            "last": max(0, frame_count - 1),
        }.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame_path = output_dir / f"{label}_frame.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames[label] = str(frame_path)
            stats[label] = {
                "frame_index": index,
                "mean": round(float(frame.mean()), 4),
                "std": round(float(frame.std()), 4),
                "min": int(frame.min()),
                "max": int(frame.max()),
            }
        return {
            "status": "passed" if frames else "blocked",
            "frame_count": frame_count,
            "frames": frames,
            "stats": stats,
        }
    finally:
        capture.release()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SO-100 keypoint-marker trajectory samples with coupled safe waypoints.")
    parser.add_argument("--output-root", type=Path, default=Path("_workspace/real_so100/keypoint_marker_trajectory"))
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--camera-index", type=int, action="append", required=True)
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--max-abs-delta-raw", type=float, default=60.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.12)
    parser.add_argument("--waypoint-settle-seconds", type=float, default=0.45)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            collect_keypoint_marker_trajectory(
                output_root=args.output_root,
                port=args.port,
                calibration=args.calibration,
                camera_indexes=args.camera_index,
                observer_camera_index=args.observer_camera_index,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
                max_abs_delta_raw=args.max_abs_delta_raw,
                step_settle_seconds=args.step_settle_seconds,
                waypoint_settle_seconds=args.waypoint_settle_seconds,
                video_fps=args.video_fps,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
