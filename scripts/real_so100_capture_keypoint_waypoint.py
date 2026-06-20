#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.real_so100_apriltag_detection_check import check_apriltag_detection
from scripts.real_so100_motor_state_snapshot import read_motor_state_snapshot


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")


def capture_keypoint_waypoint(
    *,
    name: str,
    role: str,
    output_root: Path,
    port: str,
    calibration: Path,
    camera_indexes: list[int],
    dictionaries: list[str],
    warmup_frames: int,
    notes: list[str],
) -> dict[str, Any]:
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    detection = check_apriltag_detection(
        camera_indexes=camera_indexes,
        output_dir=output_dir,
        dictionaries=dictionaries,
        warmup_frames=warmup_frames,
    )
    motor = read_motor_state_snapshot(
        port=port,
        calibration=calibration,
        output=output_dir / "motor_state_snapshot.json",
    )
    packet = _build_packet(
        name=name,
        role=role,
        output_dir=output_dir,
        port=port,
        calibration=calibration,
        detection=detection,
        motor=motor,
        notes=notes,
    )
    packet_path = output_dir / "waypoint_packet.json"
    packet["waypoint_packet"] = str(packet_path)
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    _update_ledger(output_root=output_root, packet=packet)
    return packet


def _build_packet(
    *,
    name: str,
    role: str,
    output_dir: Path,
    port: str,
    calibration: Path,
    detection: dict[str, Any],
    motor: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    markers = []
    for camera, camera_report in detection.get("cameras", {}).items():
        for detection_item in camera_report.get("detections", []):
            for marker in detection_item.get("markers", []):
                min_side = float(marker.get("min_side_px", 0.0))
                markers.append(
                    {
                        "camera": int(camera),
                        "marker_id": int(marker["id"]),
                        "center_px": marker["center_px"],
                        "corners_px": marker["corners_px"],
                        "mean_side_px": marker["mean_side_px"],
                        "min_side_px": marker["min_side_px"],
                        "quality": "strong" if min_side >= 45 else "usable" if min_side >= 24 else "weak",
                    }
                )
    motors = motor.get("motors", {})
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_keypoint_waypoint_capture",
        "name": name,
        "role": role,
        "status": "passed" if detection.get("ok") and motor.get("ok") else "blocked",
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "writes_intended": False,
        "port": port,
        "calibration": str(calibration),
        "camera_detection_report": str(output_dir / "detection_report.json"),
        "motor_state_snapshot": str(output_dir / "motor_state_snapshot.json"),
        "raw_images": {
            camera: report.get("raw_image")
            for camera, report in detection.get("cameras", {}).items()
        },
        "overlay_images": {
            camera: report.get("overlay_image")
            for camera, report in detection.get("cameras", {}).items()
        },
        "marker_observations": markers,
        "joint_positions_raw": {
            joint: item.get("Present_Position")
            for joint, item in motors.items()
        },
        "joint_position_fraction_in_calibration": {
            joint: item.get("position_fraction_in_calibration")
            for joint, item in motors.items()
        },
        "torque_enable": {
            joint: item.get("Torque_Enable")
            for joint, item in motors.items()
        },
        "voltage_readback": {
            joint: item.get("Present_Voltage")
            for joint, item in motors.items()
        },
        "near_calibration_limits": _near_limits(motors),
        "notes": notes,
        "blockers": _blockers(detection=detection, motor=motor),
    }


def _near_limits(motors: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for joint, item in motors.items():
        fraction = item.get("position_fraction_in_calibration")
        if isinstance(fraction, (int, float)) and (fraction <= 0.05 or fraction >= 0.95):
            result.append(
                {
                    "joint": joint,
                    "fraction": fraction,
                    "position_raw": item.get("Present_Position"),
                }
            )
    return result


def _blockers(*, detection: dict[str, Any], motor: dict[str, Any]) -> list[str]:
    blockers = []
    if not detection.get("ok"):
        blockers.append("camera tag detection failed")
    if not motor.get("ok"):
        blockers.append("motor state snapshot failed")
    return blockers


def _update_ledger(*, output_root: Path, packet: dict[str, Any]) -> None:
    ledger_path = output_root / "waypoints.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(packet, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one read-only SO-100 keypoint waypoint sample.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("_workspace/real_so100/keypoint_pose_waypoints"))
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--camera-index", type=int, action="append", required=True)
    parser.add_argument("--dictionary", action="append", default=["tag36h11"])
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            capture_keypoint_waypoint(
                name=args.name,
                role=args.role,
                output_root=args.output_root,
                port=args.port,
                calibration=args.calibration,
                camera_indexes=args.camera_index,
                dictionaries=args.dictionary,
                warmup_frames=args.warmup_frames,
                notes=args.note,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
